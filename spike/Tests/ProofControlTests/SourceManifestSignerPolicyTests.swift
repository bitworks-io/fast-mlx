import CryptoKit
import Darwin
import Foundation
import XCTest
@testable import ProofControl

final class SourceManifestSignerPolicyTests: XCTestCase {
    private static let baselineCommit =
        "4400b8932df374945ebef2cc504782016297c0df"
    private static let baselineTree =
        "7c81bdf20225425e61efec24ce02835c9893fffa"
    private static let candidateCommit =
        "f8d86192e2c558605a8745c446598063aedaac36"
    private static let candidateTree =
        "f9ca3359d542ad621650fd968193e97051f56afb"
    private static let blobA = String(repeating: "1", count: 40)
    private static let blobB = String(repeating: "2", count: 40)
    private static let shaA = String(repeating: "a", count: 64)
    private static let shaB = String(repeating: "b", count: 64)
    private static let hex10 = String(repeating: "10", count: 32)
    private static let hex11 = String(repeating: "11", count: 32)
    private static let hex12 = String(repeating: "12", count: 32)
    private static let hex13 = String(repeating: "13", count: 32)
    private static let hex14 = String(repeating: "14", count: 32)
    private static let hex15 = String(repeating: "15", count: 32)
    private static let hex16 = String(repeating: "16", count: 32)
    private static let hex17 = String(repeating: "17", count: 32)
    private static let hex18 = String(repeating: "18", count: 32)
    private static let hex19 = String(repeating: "19", count: 32)
    private static let hex20 = String(repeating: "20", count: 32)
    private static let hex21 = String(repeating: "21", count: 32)
    private static let hex22 = String(repeating: "22", count: 32)
    private static let hex23 = String(repeating: "23", count: 32)
    private static let hex24 = String(repeating: "24", count: 32)
    private static let hex25 = String(repeating: "25", count: 32)
    private static let hex26 = String(repeating: "26", count: 32)

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
            "fast-mlx-source-policy-\(UUID().uuidString)"
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

    func testCanonicalPolicyBytesAdmitTypedSourceOnlyPolicy() throws {
        let fixture = try makeFixture()

        XCTAssertEqual(
            try SourceManifestSignerPolicyVerifier.policyBytes(
                fields: fixture.sourcePolicyFields
            ),
            fixture.sourcePolicyBytes
        )

        let sourcePolicy = try admitSourcePolicy(
            file: fixture.sourcePolicyFile,
            rootSignatureBase64: fixture.sourcePolicyRootSignatureBase64,
            root: fixture.sourcePolicyRoot,
            runKeyPolicy: fixture.runKeyPolicy
        )

        XCTAssertEqual(sourcePolicy.file, fixture.sourcePolicyFile)
        XCTAssertEqual(sourcePolicy.policySHA256, fixture.sourcePolicyFile.sha256)
        XCTAssertEqual(sourcePolicy.policyGeneration, 11)
        XCTAssertEqual(sourcePolicy.validFromUnixSeconds, 1_900_000_000)
        XCTAssertEqual(sourcePolicy.validUntilUnixSeconds, 2_100_000_000)
        XCTAssertEqual(sourcePolicy.rootKeyID, fixture.sourcePolicyRoot.keyID)
        XCTAssertEqual(
            sourcePolicy.activeSourceOperatorKeyID,
            fixture.sourceManifestSigner.keyID
        )
        XCTAssertEqual(
            sourcePolicy.activeSourceOperatorPublicKeyBase64,
            fixture.sourceManifestSigner.publicKeyBase64
        )
        XCTAssertEqual(
            sourcePolicy.activeSourceOperatorScope,
            SourceManifestSignerPolicyScope.sourceManifest
        )
        XCTAssertEqual(
            sourcePolicy.allowedAuthorizationPurpose,
            OperatorAuthorizationPurpose.sourceManifest
        )
        XCTAssertEqual(
            sourcePolicy.allowedRunClaimSubject,
            OperatorRunClaimSubject.absorbedMLALoadedResultPair
        )
        XCTAssertEqual(
            sourcePolicy.allowedSourceRoles,
            [RunSourceRole.baseline, RunSourceRole.candidate]
        )
        XCTAssertEqual(sourcePolicy.revokedSourceOperatorKeyIDs, [String]())
        XCTAssertEqual(
            sourcePolicy.rootSignatureSHA256,
            sha256Hex(
                Data(
                    base64Encoded: fixture.sourcePolicyRootSignatureBase64
                )!
            )
        )
        XCTAssertEqual(
            sourcePolicy.policyID.rawValue,
            expectedAdmittedSourcePolicyID(sourcePolicy)
        )
        XCTAssertNotEqual(sourcePolicy.policyID.rawValue, sourcePolicy.policySHA256)
        XCTAssertNotEqual(
            sourcePolicy.policyID.rawValue,
            sourcePolicy.activeSourceOperatorKeyID
        )

        let sharedRootFixture = try makeFixture(
            sourcePolicyRoot: .runPolicyRoot
        )
        let sharedRootPolicy = try admitSourcePolicy(
            from: sharedRootFixture
        )
        XCTAssertEqual(
            sharedRootPolicy.rootKeyID,
            sharedRootFixture.runKeyPolicy.rootKeyID
        )
    }

    func testRejectsNonCanonicalPolicyAndExternalAnchorConfusion() throws {
        let fixture = try makeFixture()
        var reordered = String(
            decoding: fixture.sourcePolicyBytes,
            as: UTF8.self
        )
        .split(separator: "\n", omittingEmptySubsequences: false)
        reordered.swapAt(3, 4)

        let otherID = String(repeating: "01", count: 32)
        let laterID = String(repeating: "02", count: 32)

        let cases: [(Data, SourceManifestSignerPolicyError)] = [
            (
                replacing(
                    "fast-mlx-proof-control-source-manifest-signer-policy-v1",
                    with: "foreign-policy-v1",
                    in: fixture.sourcePolicyBytes
                ),
                SourceManifestSignerPolicyError.nonCanonicalPolicy
            ),
            (
                replacing(
                    "subject=source-manifest-signing-key-policy",
                    with: "subject=operator-run-signing-key-policy",
                    in: fixture.sourcePolicyBytes
                ),
                SourceManifestSignerPolicyError.nonCanonicalPolicy
            ),
            (
                replacing(
                    "root_key_id=\(fixture.sourcePolicyRoot.keyID)",
                    with: "root_key_id=\(fixture.sourcePolicyRoot.keyID.uppercased())",
                    in: fixture.sourcePolicyBytes
                ),
                SourceManifestSignerPolicyError.nonCanonicalPolicy
            ),
            (
                replacing(
                    "active_source_operator_key_id=\(fixture.sourceManifestSigner.keyID)",
                    with: "active_source_operator_key_id=\(fixture.sourceManifestSigner.keyID.uppercased())",
                    in: fixture.sourcePolicyBytes
                ),
                SourceManifestSignerPolicyError.nonCanonicalPolicy
            ),
            (
                replacing(
                    "active_source_operator_key_id=\(fixture.sourceManifestSigner.keyID)",
                    with: "active_source_operator_key_id=\(Self.hex10)",
                    in: fixture.sourcePolicyBytes
                ),
                SourceManifestSignerPolicyError
                    .activeSourceOperatorKeyIDMismatch
            ),
            (
                replacing(
                    "active_source_operator_public_key_base64=\(fixture.sourceManifestSigner.publicKeyBase64)",
                    with: "active_source_operator_public_key_base64=not-base64",
                    in: fixture.sourcePolicyBytes
                ),
                SourceManifestSignerPolicyError
                    .invalidActiveSourceOperatorPublicKeyEncoding
            ),
            (
                replacing(
                    "policy_generation=11",
                    with: "policy_generation=011",
                    in: fixture.sourcePolicyBytes
                ),
                SourceManifestSignerPolicyError.nonCanonicalPolicy
            ),
            (
                replacing(
                    "policy_generation=11",
                    with: "policy_generation=0",
                    in: fixture.sourcePolicyBytes
                ),
                SourceManifestSignerPolicyError.nonCanonicalPolicy
            ),
            (
                replacing(
                    "valid_from_unix_seconds=1900000000",
                    with: "valid_from_unix_seconds=2100000001",
                    in: fixture.sourcePolicyBytes
                ),
                SourceManifestSignerPolicyError.nonCanonicalPolicy
            ),
            (
                replacing(
                    "active_source_operator_scope=source-manifest",
                    with: "active_source_operator_scope=run-claim",
                    in: fixture.sourcePolicyBytes
                ),
                SourceManifestSignerPolicyError.nonCanonicalPolicy
            ),
            (
                replacing(
                    "allowed_authorization_purpose=source-manifest",
                    with: "allowed_authorization_purpose=worker-bytes",
                    in: fixture.sourcePolicyBytes
                ),
                SourceManifestSignerPolicyError.nonCanonicalPolicy
            ),
            (
                replacing(
                    "allowed_run_claim_subject=absorbed-mla-loaded-result-pair",
                    with: "allowed_run_claim_subject=other",
                    in: fixture.sourcePolicyBytes
                ),
                SourceManifestSignerPolicyError.nonCanonicalPolicy
            ),
            (
                replacing(
                    "allowed_source_roles=baseline,candidate",
                    with: "allowed_source_roles=candidate,baseline",
                    in: fixture.sourcePolicyBytes
                ),
                SourceManifestSignerPolicyError.nonCanonicalPolicy
            ),
            (
                replacing(
                    "revoked_source_operator_key_ids=none",
                    with: "revoked_source_operator_key_ids=\(laterID),\(otherID)",
                    in: fixture.sourcePolicyBytes
                ),
                SourceManifestSignerPolicyError.nonCanonicalPolicy
            ),
            (
                fixture.sourcePolicyBytes + Data("extra=true\n".utf8),
                SourceManifestSignerPolicyError.nonCanonicalPolicy
            ),
            (
                Data(fixture.sourcePolicyBytes.dropLast()),
                SourceManifestSignerPolicyError.nonCanonicalPolicy
            ),
            (
                Data(
                    reordered.map(String.init).joined(separator: "\n").utf8
                ),
                SourceManifestSignerPolicyError.nonCanonicalPolicy
            ),
            (
                Data(
                    String(decoding: fixture.sourcePolicyBytes, as: UTF8.self)
                        .replacingOccurrences(of: "\n", with: "\r\n")
                        .utf8
                ),
                SourceManifestSignerPolicyError.nonCanonicalPolicy
            ),
            (
                Data([0xff]),
                SourceManifestSignerPolicyError.nonCanonicalPolicy
            ),
        ]

        for (bytes, expectedError) in cases {
            let file = try capturePayload(bytes, name: "\(UUID().uuidString).policy")
            XCTAssertThrowsError(
                try admitSourcePolicy(
                    file: file,
                    rootSignatureBase64:
                        fixture.sourcePolicyRoot.signingMaterial
                        .signature(for: bytes)
                        .base64EncodedString(),
                    root: fixture.sourcePolicyRoot,
                    expectedPolicySHA256: file.sha256,
                    runKeyPolicy: fixture.runKeyPolicy
                )
            ) { error in
                XCTAssertEqual(error as? SourceManifestSignerPolicyError, expectedError)
            }
        }

        let anchorCases: [
            (SourceManifestSignerPolicyTrustAnchor, SourceManifestSignerPolicyError)
        ] = [
            (
                trustAnchor(
                    root: fixture.sourcePolicyRoot,
                    expectedPolicySHA256: Self.hex10
                ),
                SourceManifestSignerPolicyError.policyDigestMismatch
            ),
            (
                trustAnchor(
                    root: fixture.sourcePolicyRoot,
                    expectedPolicySHA256: fixture.sourcePolicyFile.sha256,
                    minimumPolicyGeneration: 12
                ),
                SourceManifestSignerPolicyError.policyGenerationRollback(
                    minimum: 12,
                    actual: 11
                )
            ),
            (
                trustAnchor(
                    root: fixture.sourcePolicyRoot,
                    expectedPolicySHA256: fixture.sourcePolicyFile.sha256,
                    verificationUnixSeconds: 1_899_999_999
                ),
                SourceManifestSignerPolicyError.policyNotYetValid
            ),
            (
                trustAnchor(
                    root: fixture.sourcePolicyRoot,
                    expectedPolicySHA256: fixture.sourcePolicyFile.sha256,
                    verificationUnixSeconds: 2_100_000_001
                ),
                SourceManifestSignerPolicyError.policyExpired
            ),
            (
                SourceManifestSignerPolicyTrustAnchor(
                    rootPublicKeyBase64:
                        fixture.sourcePolicyRoot.publicKeyBase64,
                    rootKeyID: fixture.sourcePolicyRoot.keyID.uppercased(),
                    expectedCurrentPolicySHA256:
                        fixture.sourcePolicyFile.sha256,
                    minimumPolicyGeneration: 11,
                    verificationUnixSeconds: 2_000_000_000
                ),
                SourceManifestSignerPolicyError.invalidTrustAnchor(
                    .rootKeyID
                )
            ),
            (
                SourceManifestSignerPolicyTrustAnchor(
                    rootPublicKeyBase64:
                        fixture.sourcePolicyRoot.publicKeyBase64,
                    rootKeyID: fixture.sourcePolicyRoot.keyID,
                    expectedCurrentPolicySHA256:
                        fixture.sourcePolicyFile.sha256.uppercased(),
                    minimumPolicyGeneration: 11,
                    verificationUnixSeconds: 2_000_000_000
                ),
                SourceManifestSignerPolicyError.invalidTrustAnchor(
                    .expectedCurrentPolicySHA256
                )
            ),
        ]

        for (anchor, expectedError) in anchorCases {
            XCTAssertThrowsError(
                try SourceManifestSignerPolicyVerifier.admit(
                    policyFile: fixture.sourcePolicyFile,
                    rootSignatureBase64:
                        fixture.sourcePolicyRootSignatureBase64,
                    trustAnchor: anchor,
                    runKeyPolicy: fixture.runKeyPolicy
                )
            ) { error in
                XCTAssertEqual(error as? SourceManifestSignerPolicyError, expectedError)
            }
        }

        for boundary in [UInt64(1_900_000_000), 2_100_000_000] {
            XCTAssertNoThrow(
                try SourceManifestSignerPolicyVerifier.admit(
                    policyFile: fixture.sourcePolicyFile,
                    rootSignatureBase64:
                        fixture.sourcePolicyRootSignatureBase64,
                    trustAnchor: trustAnchor(
                        root: fixture.sourcePolicyRoot,
                        expectedPolicySHA256:
                            fixture.sourcePolicyFile.sha256,
                        verificationUnixSeconds: boundary
                    ),
                    runKeyPolicy: fixture.runKeyPolicy
                )
            )
        }
    }

    func testRejectsMalformedForeignRawDigestRevokedAndSeparatedKeys()
        throws
    {
        let fixture = try makeFixture()
        let foreign = try SourcePolicySigningKey(.foreignRoot)
        let rawDigestSignature = try fixture.sourcePolicyRoot.signingMaterial
            .signature(for: Data(hexToBytes(fixture.sourcePolicyFile.sha256)))
            .base64EncodedString()
        let revokedPolicyBytes = replacing(
            "revoked_source_operator_key_ids=none",
            with:
                "revoked_source_operator_key_ids=" +
                fixture.sourceManifestSigner.keyID,
            in: fixture.sourcePolicyBytes
        )
        let revokedFile = try capturePayload(
            revokedPolicyBytes,
            name: "revoked-source.policy"
        )
        let rootRevokedPolicyBytes = replacing(
            "revoked_source_operator_key_ids=none",
            with:
                "revoked_source_operator_key_ids=" +
                fixture.sourcePolicyRoot.keyID,
            in: fixture.sourcePolicyBytes
        )
        let rootRevokedFile = try capturePayload(
            rootRevokedPolicyBytes,
            name: "root-revoked-source.policy"
        )
        let rootIsActiveBytes = replacing(
            "active_source_operator_public_key_base64=" +
                fixture.sourceManifestSigner.publicKeyBase64,
            with:
                "active_source_operator_public_key_base64=" +
                fixture.sourcePolicyRoot.publicKeyBase64,
            in: replacing(
                "active_source_operator_key_id=" +
                    fixture.sourceManifestSigner.keyID,
                with:
                    "active_source_operator_key_id=" +
                    fixture.sourcePolicyRoot.keyID,
                in: fixture.sourcePolicyBytes
            )
        )
        let rootIsActiveFile = try capturePayload(
            rootIsActiveBytes,
            name: "root-is-active-source.policy"
        )
        let sourceActiveIsRunActiveBytes = try sourcePolicyBytes(
            root: fixture.sourcePolicyRoot,
            active: fixture.runClaimSigner,
            revokedIDs: []
        )
        let sourceActiveIsRunActiveFile = try capturePayload(
            sourceActiveIsRunActiveBytes,
            name: "source-active-is-run-active.policy"
        )
        let sourceRootIsRunActiveBytes = try sourcePolicyBytes(
            root: fixture.runClaimSigner,
            active: fixture.sourceManifestSigner,
            revokedIDs: []
        )
        let sourceRootIsRunActiveFile = try capturePayload(
            sourceRootIsRunActiveBytes,
            name: "source-root-is-run-active.policy"
        )
        let sourceActiveIsRunRootBytes = try sourcePolicyBytes(
            root: fixture.sourcePolicyRoot,
            active: fixture.runPolicyRoot,
            revokedIDs: []
        )
        let sourceActiveIsRunRootFile = try capturePayload(
            sourceActiveIsRunRootBytes,
            name: "source-active-is-run-root.policy"
        )

        let cases: [
            (
                AdmittedFile,
                SourcePolicySigningKey,
                String,
                SourceManifestSignerPolicyError
            )
        ] = [
            (
                fixture.sourcePolicyFile,
                fixture.sourcePolicyRoot,
                fixture.sourcePolicyRootSignatureBase64 + "\n",
                SourceManifestSignerPolicyError.invalidRootSignatureEncoding
            ),
            (
                fixture.sourcePolicyFile,
                fixture.sourcePolicyRoot,
                rawDigestSignature,
                SourceManifestSignerPolicyError.rootSignatureRejected
            ),
            (
                fixture.sourcePolicyFile,
                fixture.sourcePolicyRoot,
                try foreign.signingMaterial
                    .signature(for: fixture.sourcePolicyBytes)
                    .base64EncodedString(),
                SourceManifestSignerPolicyError.rootSignatureRejected
            ),
            (
                revokedFile,
                fixture.sourcePolicyRoot,
                try fixture.sourcePolicyRoot.signingMaterial
                    .signature(for: revokedPolicyBytes)
                    .base64EncodedString(),
                SourceManifestSignerPolicyError
                    .activeSourceOperatorKeyRevoked
            ),
            (
                rootRevokedFile,
                fixture.sourcePolicyRoot,
                try fixture.sourcePolicyRoot.signingMaterial
                    .signature(for: rootRevokedPolicyBytes)
                    .base64EncodedString(),
                SourceManifestSignerPolicyError
                    .rootKeyRevocationUnsupported
            ),
            (
                rootIsActiveFile,
                fixture.sourcePolicyRoot,
                try fixture.sourcePolicyRoot.signingMaterial
                    .signature(for: rootIsActiveBytes)
                    .base64EncodedString(),
                SourceManifestSignerPolicyError
                    .rootAndActiveSourceKeysMustDiffer
            ),
            (
                sourceActiveIsRunActiveFile,
                fixture.sourcePolicyRoot,
                try fixture.sourcePolicyRoot.signingMaterial
                    .signature(for: sourceActiveIsRunActiveBytes)
                    .base64EncodedString(),
                SourceManifestSignerPolicyError
                    .sourceAndRunActiveKeysMustDiffer
            ),
            (
                sourceRootIsRunActiveFile,
                fixture.runClaimSigner,
                try fixture.runClaimSigner.signingMaterial
                    .signature(for: sourceRootIsRunActiveBytes)
                    .base64EncodedString(),
                SourceManifestSignerPolicyError
                    .sourceRootAndRunActiveKeyMustDiffer
            ),
            (
                sourceActiveIsRunRootFile,
                fixture.sourcePolicyRoot,
                try fixture.sourcePolicyRoot.signingMaterial
                    .signature(for: sourceActiveIsRunRootBytes)
                    .base64EncodedString(),
                SourceManifestSignerPolicyError
                    .sourceActiveAndRunPolicyRootMustDiffer
            ),
        ]

        for (file, root, signature, expectedError) in cases {
            XCTAssertThrowsError(
                try admitSourcePolicy(
                    file: file,
                    rootSignatureBase64: signature,
                    root: root,
                    expectedPolicySHA256: file.sha256,
                    runKeyPolicy: fixture.runKeyPolicy
                )
            ) { error in
                XCTAssertEqual(error as? SourceManifestSignerPolicyError, expectedError)
            }
        }

        let malformedRootAnchor = SourceManifestSignerPolicyTrustAnchor(
            rootPublicKeyBase64:
                fixture.sourcePolicyRoot.publicKeyBase64 + "\n",
            rootKeyID: fixture.sourcePolicyRoot.keyID,
            expectedCurrentPolicySHA256: fixture.sourcePolicyFile.sha256,
            minimumPolicyGeneration: 11,
            verificationUnixSeconds: 2_000_000_000
        )
        XCTAssertThrowsError(
            try SourceManifestSignerPolicyVerifier.admit(
                policyFile: fixture.sourcePolicyFile,
                rootSignatureBase64: fixture.sourcePolicyRootSignatureBase64,
                trustAnchor: malformedRootAnchor,
                runKeyPolicy: fixture.runKeyPolicy
            )
        ) { error in
            XCTAssertEqual(
                error as? SourceManifestSignerPolicyError,
                SourceManifestSignerPolicyError.invalidRootPublicKeyEncoding
            )
        }
    }

    func testPolicyMatchesBothClaimSourceAuthorizationsAndTypedAggregateID()
        throws
    {
        let fixture = try makeFixture()
        let sourcePolicy = try admitSourcePolicy(from: fixture)

        let matched = try SourceInputsPolicyResolver.resolve(
            signedClaim: fixture.signedClaim,
            runKeyPolicy: fixture.runKeyPolicy,
            sourcePolicy: sourcePolicy,
            baseline: fixture.baselineClaimMatch,
            candidate: fixture.candidateClaimMatch
        )

        XCTAssertEqual(matched.signedClaimID, fixture.signedClaim.claimID)
        XCTAssertEqual(matched.runKeyPolicy, fixture.runKeyPolicy)
        XCTAssertEqual(matched.sourcePolicy, sourcePolicy)
        XCTAssertEqual(matched.baseline.sourceManifest, fixture.baselineManifest)
        XCTAssertEqual(matched.candidate.sourceManifest, fixture.candidateManifest)
        XCTAssertEqual(matched.baseline.role, RunSourceRole.baseline)
        XCTAssertEqual(matched.candidate.role, RunSourceRole.candidate)
        XCTAssertEqual(
            matched.baseline.policyMatchID.rawValue,
            expectedPolicyMatchID(
                sourcePolicy: sourcePolicy,
                signedClaim: fixture.signedClaim,
                claimMatch: fixture.baselineClaimMatch
            )
        )
        XCTAssertEqual(
            matched.candidate.policyMatchID.rawValue,
            expectedPolicyMatchID(
                sourcePolicy: sourcePolicy,
                signedClaim: fixture.signedClaim,
                claimMatch: fixture.candidateClaimMatch
            )
        )
        XCTAssertEqual(
            matched.sourceInputsPolicyMatchID.rawValue,
            expectedSourceInputsPolicyMatchID(
                sourcePolicy: sourcePolicy,
                signedClaim: fixture.signedClaim,
                runKeyPolicy: fixture.runKeyPolicy,
                baseline: matched.baseline,
                candidate: matched.candidate
            )
        )
        XCTAssertEqual(matched.sourcePolicySHA256, sourcePolicy.policySHA256)
        XCTAssertNotEqual(
            matched.sourceInputsPolicyMatchID.rawValue,
            sourcePolicy.policyID.rawValue
        )
        XCTAssertFalse(matched.canImportGitObjects)
        XCTAssertFalse(matched.canBuild)
        XCTAssertFalse(matched.canSpawn)
        XCTAssertFalse(matched.canLoadModel)
        XCTAssertFalse(matched.canReserveOutput)
        XCTAssertFalse(matched.canPublish)
    }

    func testResolutionRejectsWorkerAndWrongSourcePolicyReference()
        throws
    {
        let fixture = try makeFixture()
        let sourcePolicy = try admitSourcePolicy(from: fixture)
        let wrongPolicyFixture = try makeFixture(sourcePolicySHA256: Self.hex10)

        XCTAssertThrowsError(
            try SourceInputsPolicyResolver.resolve(
                signedClaim: wrongPolicyFixture.signedClaim,
                runKeyPolicy: wrongPolicyFixture.runKeyPolicy,
                sourcePolicy: sourcePolicy,
                baseline: wrongPolicyFixture.baselineClaimMatch,
                candidate: wrongPolicyFixture.candidateClaimMatch
            )
        ) { error in
            XCTAssertEqual(
                error as? SourceManifestSignerPolicyError,
                SourceManifestSignerPolicyError.signedClaimSourcePolicyMismatch
            )
        }

        let wrongRunKeyPolicy = try makeRunKeyPolicy(
            root: SourcePolicySigningKey(.foreignRoot),
            active: SourcePolicySigningKey(.worker)
        )
        XCTAssertThrowsError(
            try SourceInputsPolicyResolver.resolve(
                signedClaim: fixture.signedClaim,
                runKeyPolicy: wrongRunKeyPolicy,
                sourcePolicy: sourcePolicy,
                baseline: fixture.baselineClaimMatch,
                candidate: fixture.candidateClaimMatch
            )
        ) { error in
            XCTAssertEqual(
                error as? SourceManifestSignerPolicyError,
                SourceManifestSignerPolicyError
                    .signedClaimRunKeyPolicyMismatch
            )
        }

        XCTAssertThrowsError(
            try SourceManifestSignerPolicyVerifier.matchAuthorization(
                fixture.worker,
                sourcePolicy: sourcePolicy,
                role: RunAuthorizedInputRole.worker
            )
        ) { error in
            XCTAssertEqual(
                error as? SourceManifestSignerPolicyError,
                SourceManifestSignerPolicyError.unexpectedAuthorizationPurpose(
                    expected: OperatorAuthorizationPurpose.sourceManifest,
                    actual: OperatorAuthorizationPurpose.workerBytes
                )
            )
        }

        XCTAssertThrowsError(
            try SourceManifestSignerPolicyVerifier.matchAuthorization(
                fixture.baselineManifest.authorizedFile,
                sourcePolicy: sourcePolicy,
                role: RunAuthorizedInputRole.worker
            )
        ) { error in
            XCTAssertEqual(
                error as? SourceManifestSignerPolicyError,
                SourceManifestSignerPolicyError
                    .unsupportedInputRole(.worker)
            )
        }
    }

    func testResolutionRejectsSourceManifestAuthorizationsFromInactiveKeys()
        throws
    {
        let wrongBaseline = try makeFixture(
            baselineSourceSigner: .foreignRoot
        )
        let baselinePolicy = try admitSourcePolicy(from: wrongBaseline)
        XCTAssertThrowsError(
            try SourceInputsPolicyResolver.resolve(
                signedClaim: wrongBaseline.signedClaim,
                runKeyPolicy: wrongBaseline.runKeyPolicy,
                sourcePolicy: baselinePolicy,
                baseline: wrongBaseline.baselineClaimMatch,
                candidate: wrongBaseline.candidateClaimMatch
            )
        ) { error in
            XCTAssertEqual(
                error as? SourceManifestSignerPolicyError,
                SourceManifestSignerPolicyError
                    .sourceAuthorizationKeyMismatch(role: .baseline)
            )
        }

        let wrongCandidate = try makeFixture(
            candidateSourceSigner: .foreignRoot
        )
        let candidatePolicy = try admitSourcePolicy(from: wrongCandidate)
        XCTAssertThrowsError(
            try SourceInputsPolicyResolver.resolve(
                signedClaim: wrongCandidate.signedClaim,
                runKeyPolicy: wrongCandidate.runKeyPolicy,
                sourcePolicy: candidatePolicy,
                baseline: wrongCandidate.baselineClaimMatch,
                candidate: wrongCandidate.candidateClaimMatch
            )
        ) { error in
            XCTAssertEqual(
                error as? SourceManifestSignerPolicyError,
                SourceManifestSignerPolicyError
                    .sourceAuthorizationKeyMismatch(role: .candidate)
            )
        }
    }

    func testResolutionRejectsCrossClaimCrossRoleAndPartialAggregate()
        throws
    {
        let fixture = try makeFixture()
        let otherClaimFixture = try makeFixture(
            resultPairID: Self.hex23
        )
        let sourcePolicy = try admitSourcePolicy(from: fixture)
        XCTAssertNotEqual(
            fixture.signedClaim.claimID,
            otherClaimFixture.signedClaim.claimID
        )

        XCTAssertThrowsError(
            try SourceInputsPolicyResolver.resolve(
                signedClaim: fixture.signedClaim,
                runKeyPolicy: fixture.runKeyPolicy,
                sourcePolicy: sourcePolicy,
                baseline: otherClaimFixture.baselineClaimMatch,
                candidate: fixture.candidateClaimMatch
            )
        ) { error in
            XCTAssertEqual(
                error as? SourceManifestSignerPolicyError,
                SourceManifestSignerPolicyError
                    .claimMatchedSourceManifestClaimMismatch(
                        role: RunSourceRole.baseline
                    )
            )
        }

        XCTAssertThrowsError(
            try SourceInputsPolicyResolver.resolve(
                signedClaim: fixture.signedClaim,
                runKeyPolicy: fixture.runKeyPolicy,
                sourcePolicy: sourcePolicy,
                baseline: fixture.candidateClaimMatch,
                candidate: fixture.baselineClaimMatch
            )
        ) { error in
            XCTAssertEqual(
                error as? SourceManifestSignerPolicyError,
                SourceManifestSignerPolicyError.sourceRoleMismatch(
                    expected: RunSourceRole.baseline,
                    actual: RunSourceRole.candidate
                )
            )
        }

        XCTAssertThrowsError(
            try SourceInputsPolicyResolver.resolve(
                signedClaim: fixture.signedClaim,
                runKeyPolicy: fixture.runKeyPolicy,
                sourcePolicy: sourcePolicy,
                baseline: fixture.baselineClaimMatch,
                candidate: fixture.baselineClaimMatch
            )
        ) { error in
            XCTAssertEqual(
                error as? SourceManifestSignerPolicyError,
                SourceManifestSignerPolicyError.sourceRoleMismatch(
                    expected: RunSourceRole.candidate,
                    actual: RunSourceRole.baseline
                )
            )
        }

        XCTAssertThrowsError(
            try SourceInputsPolicyResolver.resolve(
                signedClaim: fixture.signedClaim,
                runKeyPolicy: fixture.runKeyPolicy,
                sourcePolicy: sourcePolicy,
                baseline: fixture.baselineClaimMatch,
                candidate: nil as ClaimMatchedSourceManifest?
            )
        ) { error in
            XCTAssertEqual(
                error as? SourceManifestSignerPolicyError,
                SourceManifestSignerPolicyError.missingRequiredSourceRole(
                    RunSourceRole.candidate
                )
            )
        }
    }

    private func makeFixture(
        sourcePolicySHA256: String? = nil,
        baselineSourceSigner: SourcePolicySigningKey.Kind = .sourceManifest,
        candidateSourceSigner: SourcePolicySigningKey.Kind = .sourceManifest,
        sourcePolicyRoot: SourcePolicySigningKey.Kind = .sourcePolicyRoot,
        resultPairID: String? = nil
    ) throws -> SourceSignerPolicyFixture {
        let runPolicyRoot = try SourcePolicySigningKey(.runPolicyRoot)
        let runClaimSigner = try SourcePolicySigningKey(.runClaim)
        let runKeyPolicy = try makeRunKeyPolicy(
            root: runPolicyRoot,
            active: runClaimSigner
        )

        let sourcePolicyRoot = try SourcePolicySigningKey(sourcePolicyRoot)
        let sourceManifestSigner = try SourcePolicySigningKey(.sourceManifest)
        let sourcePolicyBytes = try sourcePolicyBytes(
            root: sourcePolicyRoot,
            active: sourceManifestSigner,
            revokedIDs: []
        )
        let sourcePolicyFile = try capturePayload(
            sourcePolicyBytes,
            name: "source-\(UUID().uuidString).policy"
        )
        let sourcePolicyRootSignatureBase64 = try sourcePolicyRoot
            .signingMaterial
            .signature(for: sourcePolicyBytes)
            .base64EncodedString()

        let baselineSigner = try SourcePolicySigningKey(baselineSourceSigner)
        let candidateSigner = try SourcePolicySigningKey(candidateSourceSigner)
        let baselineManifest = try makeSourceManifest(
            role: .baseline,
            gitCommitSHA1: Self.baselineCommit,
            gitTreeSHA1: Self.baselineTree,
            signer: baselineSigner
        )
        let candidateManifest = try makeSourceManifest(
            role: .candidate,
            gitCommitSHA1: Self.candidateCommit,
            gitTreeSHA1: Self.candidateTree,
            signer: candidateSigner
        )
        let worker = try authorizePayload(
            Data("worker\n".utf8),
            name: "worker-\(UUID().uuidString).swift",
            purpose: .workerBytes,
            signer: .worker
        )
        let inputs = try RunAuthorizedInputs(
            worker: worker,
            baselineSourceManifest: baselineManifest.authorizedFile,
            candidateSourceManifest: candidateManifest.authorizedFile
        )
        let runner = try capturePayload(
            Data("runner\n".utf8),
            name: "runner-\(UUID().uuidString)"
        )
        let expectations = OperatorRunClaimAdmissionExpectations(
            keyPolicy: runKeyPolicy,
            hostAdmissionID: Self.hex21,
            runner: runner,
            resultPairID: resultPairID ?? Self.hex22,
            inputs: inputs
        )
        let fields = OperatorRunClaimFields(
            subject: .absorbedMLALoadedResultPair,
            operatorKeyID: runKeyPolicy.activeOperatorKeyID,
            operatorKeyPolicySHA256: runKeyPolicy.policySHA256,
            hostAdmissionID: expectations.hostAdmissionID,
            runner: OperatorRunClaimByteIdentity(
                sha256: runner.sha256,
                byteCount: UInt64(runner.bytes.count)
            ),
            worker: reference(worker),
            policies: OperatorRunClaimPolicyReferences(
                sourceSHA256: sourcePolicySHA256 ?? sourcePolicyFile.sha256,
                dependencySHA256: Self.hex10,
                buildSHA256: Self.hex11,
                runtimeSHA256: Self.hex12,
                preflightSHA256: Self.hex13,
                publicationSHA256: Self.hex14
            ),
            toolManifest: OperatorRunClaimByteIdentity(
                sha256: Self.hex15,
                byteCount: 19
            ),
            baseline: sourceReference(
                manifest: baselineManifest,
                buildReceiptID: Self.hex16,
                binarySHA256: Self.hex17,
                binaryBytes: 101
            ),
            candidate: sourceReference(
                manifest: candidateManifest,
                buildReceiptID: Self.hex18,
                binarySHA256: Self.hex19,
                binaryBytes: 102
            ),
            model: OperatorRunClaimAuthorizedPayloadReference(
                authorizationID: Self.hex20,
                payload: OperatorRunClaimByteIdentity(
                    sha256: Self.hex23,
                    byteCount: 55
                )
            ),
            tokenizer: OperatorRunClaimAuthorizedPayloadReference(
                authorizationID: Self.hex24,
                payload: OperatorRunClaimByteIdentity(
                    sha256: Self.hex25,
                    byteCount: 66
                )
            ),
            workload: OperatorRunClaimAuthorizedPayloadReference(
                authorizationID: Self.hex26,
                payload: OperatorRunClaimByteIdentity(
                    sha256: Self.hex10,
                    byteCount: 77
                )
            ),
            resultPairID: expectations.resultPairID
        )
        let claimBytes = try OperatorSignedRunClaimVerifier.claimBytes(
            fields: fields
        )
        let runClaimSignature = try runClaimSigner.signingMaterial
            .signature(for: claimBytes)
            .base64EncodedString()
        let signedClaim = try OperatorSignedRunClaimVerifier.verify(
            claimBytes: claimBytes,
            signatureBase64: runClaimSignature,
            expectations: expectations
        )
        let baselineClaimMatch = try SourceManifestAdmission.match(
            baselineManifest,
            to: signedClaim
        )
        let candidateClaimMatch = try SourceManifestAdmission.match(
            candidateManifest,
            to: signedClaim
        )

        return SourceSignerPolicyFixture(
            runPolicyRoot: runPolicyRoot,
            runClaimSigner: runClaimSigner,
            runKeyPolicy: runKeyPolicy,
            sourcePolicyRoot: sourcePolicyRoot,
            sourceManifestSigner: sourceManifestSigner,
            sourcePolicyFields: try sourcePolicyFields(
                root: sourcePolicyRoot,
                active: sourceManifestSigner,
                revokedIDs: []
            ),
            sourcePolicyBytes: sourcePolicyBytes,
            sourcePolicyFile: sourcePolicyFile,
            sourcePolicyRootSignatureBase64: sourcePolicyRootSignatureBase64,
            worker: worker,
            baselineManifest: baselineManifest,
            candidateManifest: candidateManifest,
            baselineClaimMatch: baselineClaimMatch,
            candidateClaimMatch: candidateClaimMatch,
            signedClaim: signedClaim
        )
    }

    private func makeRunKeyPolicy(
        root: SourcePolicySigningKey,
        active: SourcePolicySigningKey
    ) throws -> AdmittedOperatorKeyPolicy {
        let fields = OperatorKeyPolicyFields(
            rootKeyID: root.keyID,
            policyGeneration: 7,
            validFromUnixSeconds: 1_900_000_000,
            validUntilUnixSeconds: 2_100_000_000,
            activeOperatorKeyID: active.keyID,
            activeOperatorPublicKeyBase64: active.publicKeyBase64,
            activeOperatorScope: .runClaim,
            allowedClaimSubject: .absorbedMLALoadedResultPair,
            revokedOperatorKeyIDs: []
        )
        let bytes = try OperatorKeyPolicyVerifier.policyBytes(fields: fields)
        let file = try capturePayload(
            bytes,
            name: "run-\(UUID().uuidString).policy"
        )
        let signature = try root.signingMaterial.signature(for: bytes)
            .base64EncodedString()
        return try OperatorKeyPolicyVerifier.admit(
            policyFile: file,
            rootSignatureBase64: signature,
            trustAnchor: OperatorKeyPolicyTrustAnchor(
                rootPublicKeyBase64: root.publicKeyBase64,
                rootKeyID: root.keyID,
                expectedCurrentPolicySHA256: file.sha256,
                minimumPolicyGeneration: 7,
                verificationUnixSeconds: 2_000_000_000
            )
        )
    }

    private func sourcePolicyBytes(
        root: SourcePolicySigningKey,
        active: SourcePolicySigningKey,
        revokedIDs: [String]
    ) throws -> Data {
        try SourceManifestSignerPolicyVerifier.policyBytes(
            fields: sourcePolicyFields(
                root: root,
                active: active,
                revokedIDs: revokedIDs
            )
        )
    }

    private func sourcePolicyFields(
        root: SourcePolicySigningKey,
        active: SourcePolicySigningKey,
        revokedIDs: [String]
    ) throws -> SourceManifestSignerPolicyFields {
        SourceManifestSignerPolicyFields(
            rootKeyID: root.keyID,
            policyGeneration: 11,
            validFromUnixSeconds: 1_900_000_000,
            validUntilUnixSeconds: 2_100_000_000,
            activeSourceOperatorKeyID: active.keyID,
            activeSourceOperatorPublicKeyBase64: active.publicKeyBase64,
            activeSourceOperatorScope:
                SourceManifestSignerPolicyScope.sourceManifest,
            allowedAuthorizationPurpose:
                OperatorAuthorizationPurpose.sourceManifest,
            allowedRunClaimSubject:
                OperatorRunClaimSubject.absorbedMLALoadedResultPair,
            allowedSourceRoles: [RunSourceRole.baseline, RunSourceRole.candidate],
            revokedSourceOperatorKeyIDs: revokedIDs
        )
    }

    private func admitSourcePolicy(
        from fixture: SourceSignerPolicyFixture
    ) throws -> AdmittedSourceManifestSignerPolicy {
        try admitSourcePolicy(
            file: fixture.sourcePolicyFile,
            rootSignatureBase64: fixture.sourcePolicyRootSignatureBase64,
            root: fixture.sourcePolicyRoot,
            runKeyPolicy: fixture.runKeyPolicy
        )
    }

    private func admitSourcePolicy(
        file: AdmittedFile,
        rootSignatureBase64: String,
        root: SourcePolicySigningKey,
        expectedPolicySHA256: String? = nil,
        runKeyPolicy: AdmittedOperatorKeyPolicy
    ) throws -> AdmittedSourceManifestSignerPolicy {
        try SourceManifestSignerPolicyVerifier.admit(
            policyFile: file,
            rootSignatureBase64: rootSignatureBase64,
            trustAnchor: trustAnchor(
                root: root,
                expectedPolicySHA256: expectedPolicySHA256 ?? file.sha256
            ),
            runKeyPolicy: runKeyPolicy
        )
    }

    private func trustAnchor(
        root: SourcePolicySigningKey,
        expectedPolicySHA256: String,
        minimumPolicyGeneration: UInt64 = 11,
        verificationUnixSeconds: UInt64 = 2_000_000_000
    ) -> SourceManifestSignerPolicyTrustAnchor {
        SourceManifestSignerPolicyTrustAnchor(
            rootPublicKeyBase64: root.publicKeyBase64,
            rootKeyID: root.keyID,
            expectedCurrentPolicySHA256: expectedPolicySHA256,
            minimumPolicyGeneration: minimumPolicyGeneration,
            verificationUnixSeconds: verificationUnixSeconds
        )
    }

    private func makeSourceManifest(
        role: RunSourceRole,
        gitCommitSHA1: String,
        gitTreeSHA1: String,
        signer: SourcePolicySigningKey
    ) throws -> AdmittedSourceManifest {
        let bytes = try SourceManifestAdmission.manifestBytes(
            role: role,
            gitCommitSHA1: gitCommitSHA1,
            gitTreeSHA1: gitTreeSHA1,
            entries: [
                SourceManifestEntry(
                    mode: .regular,
                    gitBlobSHA1: Self.blobA,
                    byteCount: 0,
                    sha256: Self.shaA,
                    path: "README.md"
                ),
                SourceManifestEntry(
                    mode: .executable,
                    gitBlobSHA1: Self.blobB,
                    byteCount: 42,
                    sha256: Self.shaB,
                    path: "Sources/App/main.swift"
                ),
            ]
        )
        let authorized = try authorizePayload(
            bytes,
            name: "\(role.rawValue)-\(UUID().uuidString).manifest",
            purpose: .sourceManifest,
            signer: signer.kind
        )
        return try SourceManifestAdmission.admit(
            authorizedFile: authorized,
            expectedRole: role
        )
    }

    private func authorizePayload(
        _ bytes: Data,
        name: String,
        purpose: OperatorAuthorizationPurpose,
        signer: SourcePolicySigningKey.Kind
    ) throws -> OperatorAuthorizedFile {
        let signingKey = try SourcePolicySigningKey(signer)
        let admitted = try capturePayload(bytes, name: name)
        let claimBytes = try OperatorAuthorization.claimBytes(
            purpose: purpose,
            payloadSHA256: admitted.sha256,
            payloadByteCount: UInt64(admitted.bytes.count)
        )
        let signature = try signingKey.signingMaterial.signature(
            for: claimBytes
        )
        .base64EncodedString()
        return try OperatorAuthorization.verify(
            admittedFile: admitted,
            expectedPurpose: purpose,
            claimBytes: claimBytes,
            signatureBase64: signature,
            publicKeyBase64: signingKey.publicKeyBase64,
            allowedKeyID: signingKey.keyID
        )
    }

    private func capturePayload(_ bytes: Data, name: String) throws -> AdmittedFile {
        let url = caseRoot.appendingPathComponent(name)
        try bytes.write(to: url)
        return try AdmittedFile.capture(
            absolutePath: url.path,
            maximumBytes: 64 * 1024
        )
    }

    private func reference(
        _ authorized: OperatorAuthorizedFile
    ) -> OperatorRunClaimAuthorizedPayloadReference {
        OperatorRunClaimAuthorizedPayloadReference(
            authorizationID: authorized.authorizationID.rawValue,
            payload: OperatorRunClaimByteIdentity(
                sha256: authorized.file.sha256,
                byteCount: UInt64(authorized.file.bytes.count)
            )
        )
    }

    private func sourceReference(
        manifest: AdmittedSourceManifest,
        buildReceiptID: String,
        binarySHA256: String,
        binaryBytes: UInt64
    ) -> OperatorRunClaimSourceReference {
        OperatorRunClaimSourceReference(
            role: manifest.role == .baseline ? .baseline : .candidate,
            sourceManifest: reference(manifest.authorizedFile),
            gitCommitSHA1: manifest.gitCommitSHA1,
            gitTreeSHA1: manifest.gitTreeSHA1,
            route: manifest.route,
            slot: manifest.slot,
            buildReceiptID: buildReceiptID,
            binary: OperatorRunClaimByteIdentity(
                sha256: binarySHA256,
                byteCount: binaryBytes
            )
        )
    }

    private func expectedAdmittedSourcePolicyID(
        _ policy: AdmittedSourceManifestSignerPolicy
    ) -> String {
        typedID([
            "fast-mlx-proof-control-admitted-source-manifest-signer-policy-id-v1",
            "root_key_id=\(policy.rootKeyID)",
            "policy_sha256=\(policy.policySHA256)",
            "signature_sha256=\(policy.rootSignatureSHA256)",
        ])
    }

    private func expectedPolicyMatchID(
        sourcePolicy: AdmittedSourceManifestSignerPolicy,
        signedClaim: OperatorSignedRunClaim,
        claimMatch: ClaimMatchedSourceManifest
    ) -> String {
        let manifest = claimMatch.sourceManifest
        let authorization = manifest.authorizedFile
        return typedID([
            "fast-mlx-proof-control-policy-matched-source-manifest-authorization-id-v1",
            "source_policy_id=\(sourcePolicy.policyID.rawValue)",
            "run_claim_id=\(signedClaim.claimID.rawValue)",
            "claim_matched_source_manifest_id=\(claimMatch.matchID.rawValue)",
            "source_authorization_id=\(authorization.authorizationID.rawValue)",
            "purpose=source-manifest",
            "role=\(manifest.role.rawValue)",
            "operator_key_id=\(authorization.operatorKeyID)",
            "payload_sha256=\(authorization.file.sha256)",
            "payload_bytes=\(UInt64(authorization.file.bytes.count))",
            "claim_sha256=\(authorization.claimSHA256)",
            "signature_sha256=\(authorization.signatureSHA256)",
        ])
    }

    private func expectedSourceInputsPolicyMatchID(
        sourcePolicy: AdmittedSourceManifestSignerPolicy,
        signedClaim: OperatorSignedRunClaim,
        runKeyPolicy: AdmittedOperatorKeyPolicy,
        baseline: PolicyMatchedSourceManifestAuthorization,
        candidate: PolicyMatchedSourceManifestAuthorization
    ) -> String {
        typedID([
            "fast-mlx-proof-control-source-inputs-policy-matched-id-v1",
            "run_claim_id=\(signedClaim.claimID.rawValue)",
            "operator_key_policy_id=\(runKeyPolicy.admissionID.rawValue)",
            "source_policy_id=\(sourcePolicy.policyID.rawValue)",
            "baseline_policy_match_id=\(baseline.policyMatchID.rawValue)",
            "candidate_policy_match_id=\(candidate.policyMatchID.rawValue)",
            "source_policy_sha256=\(sourcePolicy.policySHA256)",
        ])
    }

    private func replacing(
        _ source: String,
        with replacement: String,
        in bytes: Data
    ) -> Data {
        let text = String(decoding: bytes, as: UTF8.self)
        return Data(text.replacingOccurrences(of: source, with: replacement).utf8)
    }

    private func hexToBytes(_ hex: String) -> [UInt8] {
        stride(from: 0, to: hex.count, by: 2).map { offset in
            let start = hex.index(hex.startIndex, offsetBy: offset)
            let end = hex.index(start, offsetBy: 2)
            return UInt8(hex[start..<end], radix: 16)!
        }
    }

    private func typedID(_ lines: [String]) -> String {
        sha256Hex(Data((lines.joined(separator: "\n") + "\n").utf8))
    }

    private func sha256Hex(_ bytes: Data) -> String {
        SHA256.hash(data: bytes)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}

private struct SourceSignerPolicyFixture {
    let runPolicyRoot: SourcePolicySigningKey
    let runClaimSigner: SourcePolicySigningKey
    let runKeyPolicy: AdmittedOperatorKeyPolicy
    let sourcePolicyRoot: SourcePolicySigningKey
    let sourceManifestSigner: SourcePolicySigningKey
    let sourcePolicyFields: SourceManifestSignerPolicyFields
    let sourcePolicyBytes: Data
    let sourcePolicyFile: AdmittedFile
    let sourcePolicyRootSignatureBase64: String
    let worker: OperatorAuthorizedFile
    let baselineManifest: AdmittedSourceManifest
    let candidateManifest: AdmittedSourceManifest
    let baselineClaimMatch: ClaimMatchedSourceManifest
    let candidateClaimMatch: ClaimMatchedSourceManifest
    let signedClaim: OperatorSignedRunClaim
}

private struct SourcePolicySigningKey {
    enum Kind {
        case runPolicyRoot
        case runClaim
        case sourcePolicyRoot
        case sourceManifest
        case foreignRoot
        case worker
    }

    let kind: Kind
    // gitleaks:allow
    let signingMaterial: Curve25519.Signing.PrivateKey
    let publicKeyBase64: String
    let keyID: String

    init(_ kind: Kind) throws {
        self.kind = kind
        signingMaterial = try Curve25519.Signing.PrivateKey(
            rawRepresentation: Data(kind.seed)
        )
        let publicKeyBytes = signingMaterial.publicKey.rawRepresentation
        publicKeyBase64 = publicKeyBytes.base64EncodedString()
        keyID = SHA256.hash(data: publicKeyBytes)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}

private extension SourcePolicySigningKey.Kind {
    var seed: [UInt8] {
        let start: UInt8
        switch self {
        case .runPolicyRoot:
            start = 0x10
        case .runClaim:
            start = 0x30
        case .sourcePolicyRoot:
            start = 0x70
        case .sourceManifest:
            start = 0x90
        case .foreignRoot:
            start = 0xb0
        case .worker:
            start = 0xd0
        }
        return (0..<32).map { start &+ UInt8($0) }
    }
}
