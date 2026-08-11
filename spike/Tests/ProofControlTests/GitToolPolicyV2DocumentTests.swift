import CryptoKit
import Foundation
import XCTest

@testable import ProofControl

final class GitToolPolicyV2DocumentTests: XCTestCase {
    func testCanonicalPolicyAnchorAndFreshClaimReferenceAreExactAndInert()
        throws
    {
        XCTAssertEqual(GitToolPolicyV2Verifier.maximumPolicyBytes, 2_199)
        XCTAssertEqual(GitToolPolicyV2Verifier.maximumLineBytes, 115)
        XCTAssertEqual(Self.maximumSchemaBytes.count, 2_199)
        let maximumLines = Self.maximumSchemaBytes.split(
            separator: 0x0a,
            omittingEmptySubsequences: false
        )
        XCTAssertEqual(maximumLines.count, 54)
        XCTAssertTrue(maximumLines.last?.isEmpty == true)
        XCTAssertEqual(maximumLines.dropLast().map(\.count).max(), 115)

        let generated = try GitToolPolicyV2Verifier.policyBytes(
            fields: Self.fields()
        )
        XCTAssertEqual(generated, Self.expectedPolicyBytes)
        XCTAssertEqual(generated.count, 1_912)
        XCTAssertEqual(
            Self.sha256Hex(generated),
            "31480164530be831ac078b696c38706b07042f9defd4896da6b17cf6bc34748a"
        )

        var callerBytes = generated
        let policyFile = Self.admittedFile(callerBytes)
        let document = try GitToolPolicyV2Verifier.anchor(
            policyFile: policyFile,
            trustAnchor: Self.trustAnchor(matching: policyFile)
        )
        callerBytes.append(0x78)

        XCTAssertEqual(document.policyFile, policyFile)
        XCTAssertEqual(document.policyFile.bytes, Self.expectedPolicyBytes)
        XCTAssertEqual(document.policySHA256, policyFile.sha256)
        XCTAssertEqual(document.policyBytes, 1_912)
        XCTAssertEqual(document.fields, Self.fields())
        XCTAssertEqual(
            document.gitRuntimeClosureManifestSHA256,
            Self.gitRuntimeClosureManifestSHA256
        )
        XCTAssertEqual(
            document.gitFileImageRuntimeClosureContentEvidenceID,
            Self.gitFileImageRuntimeClosureContentEvidenceID
        )
        Self.assertNoAuthority(document)

        let claim = try Self.signedClaim(
            toolManifestSHA256: policyFile.sha256,
            toolManifestBytes: UInt64(policyFile.bytes.count),
            runtimePolicySHA256: Self.runtimePolicySHA256
        )
        let reference = try GitToolPolicyV2Verifier.reference(
            signedClaim: claim,
            policyDocument: document
        )

        XCTAssertEqual(reference.signedClaim, claim)
        XCTAssertEqual(reference.policyDocument, document)
        XCTAssertEqual(
            claim.fields.toolManifest,
            OperatorRunClaimByteIdentity(
                sha256: policyFile.sha256,
                byteCount: UInt64(policyFile.bytes.count)
            )
        )
        XCTAssertEqual(
            claim.fields.policies.runtimeSHA256,
            Self.runtimePolicySHA256
        )
        Self.assertNoAuthority(reference)
        XCTAssertEqual(
            Set(Mirror(reflecting: document).children.compactMap(\.label)),
            [
                "policyFile", "policySHA256", "policyBytes",
                "policyGeneration", "validFromUnixSeconds",
                "validUntilUnixSeconds", "executableSHA256",
                "gitRuntimeClosureManifestSHA256",
                "gitFileImageRuntimeClosureContentEvidenceID",
                "runtimePolicySHA256", "fields", "canExecute",
                "canSpawn", "canAccessNetwork", "canConsumePack",
                "canMutateFileSystem", "canImportGitObjects", "canBuild",
                "canLoadModel", "canReserveOutput", "canPublish",
            ]
        )
        XCTAssertEqual(
            Set(Mirror(reflecting: reference).children.compactMap(\.label)),
            [
                "signedClaim", "policyDocument", "canExecute", "canSpawn",
                "canAccessNetwork", "canConsumePack", "canMutateFileSystem",
                "canImportGitObjects", "canBuild", "canLoadModel",
                "canReserveOutput", "canPublish",
            ]
        )
        XCTAssertEqual(Self.effects, .zero)
    }

    func testRejectsMalformedOrderDuplicateUnknownCaseWhitespaceUTF8AndRange()
        throws
    {
        var reordered = String(
            decoding: Self.expectedPolicyBytes,
            as: UTF8.self
        ).split(separator: "\n", omittingEmptySubsequences: false)
        reordered.swapAt(2, 3)

        let malformed: [Data] = [
            Self.replacing(
                "fast-mlx-proof-control-git-tool-policy-v2",
                with: "fast-mlx-proof-control-git-tool-policy-v1"
            ),
            Self.replacing(
                "subject=absorbed-mla-source-import-git",
                with: "subject=other"
            ),
            Self.replacing("policy_generation=7", with: "policy_generation=07"),
            Self.replacing("policy_generation=7", with: "policy_generation=0"),
            Self.replacing(
                "policy_generation=7",
                with: "policy_generation=18446744073709551616"
            ),
            Self.replacing(
                "valid_from_unix_seconds=1900000000",
                with: "valid_from_unix_seconds=2100000001"
            ),
            Self.replacing(
                "claim_subject=absorbed-mla-loaded-result-pair",
                with: "claim_subject=other"
            ),
            Self.replacing(
                "executable_sha256=\(Self.executableSHA256)",
                with: "executable_sha256=\(Self.executableSHA256.uppercased())"
            ),
            Self.replacing("executable_bytes=1048576", with: "executable_bytes=0"),
            Self.replacing(
                "executable_macho_uuid=\(Self.executableMachOUUID)",
                with: "executable_macho_uuid=\(Self.executableMachOUUID)0"
            ),
            Self.replacing(
                "git_runtime_closure_manifest_sha256=" +
                    Self.gitRuntimeClosureManifestSHA256,
                with: "git_runtime_closure_manifest_sha256=" +
                    Self.gitRuntimeClosureManifestSHA256.uppercased()
            ),
            Self.replacing(
                "git_runtime_closure_manifest_bytes=4096",
                with: "git_runtime_closure_manifest_bytes=0"
            ),
            Self.replacing(
                "git_file_image_runtime_closure_content_evidence_id=" +
                    Self.gitFileImageRuntimeClosureContentEvidenceID,
                with: "git_file_image_runtime_closure_content_evidence_id=" +
                    String(repeating: "AB", count: 32)
            ),
            Self.replacing(
                "runtime_policy_sha256=\(Self.runtimePolicySHA256)",
                with: "runtime_policy_sha256=\(Self.runtimePolicySHA256) "
            ),
            Self.replacing("umask=0077", with: "umask=077"),
            Self.replacing("cloexec_default=true", with: "cloexec_default=false"),
            Self.replacing("network_policy=deny-all-v1", with: "network_policy=allow-v1"),
            Self.replacing("pack_version=2", with: "pack_version=3"),
            Self.replacing("object_format=sha1", with: "object_format=sha256"),
            Self.replacing("delta_depth=0", with: "delta_depth=1"),
            Self.expectedPolicyBytes + Data("unknown=true\n".utf8),
            Data(Self.expectedPolicyBytes.dropLast()),
            Self.replacing("policy_generation=7\n", with: ""),
            Self.replacing(
                "policy_generation=7\n",
                with: "policy_generation=7\npolicy_generation=7\n"
            ),
            Data(reordered.map(String.init).joined(separator: "\n").utf8),
            Data(
                String(decoding: Self.expectedPolicyBytes, as: UTF8.self)
                    .replacingOccurrences(of: "\n", with: "\r\n").utf8
            ),
            Data([0xff]),
            Data((String(repeating: "x", count: 116) + "\n").utf8),
        ]

        for (index, bytes) in malformed.enumerated() {
            let file = Self.admittedFile(bytes)
            XCTAssertThrowsError(
                try GitToolPolicyV2Verifier.anchor(
                    policyFile: file,
                    trustAnchor: Self.trustAnchor(matching: file)
                ),
                "malformed case \(index)"
            ) { error in
                XCTAssertEqual(
                    error as? GitToolPolicyV2Error,
                    .nonCanonicalPolicy
                )
            }
        }

        let oversized = Data(repeating: 0x78, count: 2_200)
        let oversizedFile = Self.admittedFile(oversized)
        XCTAssertThrowsError(
            try GitToolPolicyV2Verifier.anchor(
                policyFile: oversizedFile,
                trustAnchor: Self.trustAnchor(matching: oversizedFile)
            )
        ) { error in
            XCTAssertEqual(error as? GitToolPolicyV2Error, .policySize)
        }
    }

    func testAnchorRejectsInReviewedFirstFailureOrderAndIncludesBoundaries()
        throws
    {
        let policyFile = Self.admittedFile(Self.expectedPolicyBytes)
        let invalidAnchor = GitToolPolicyV2TrustAnchor(
            expectedCurrentPolicySHA256:
                Self.expectedPolicySHA256.uppercased(),
            expectedCurrentPolicyBytes: 0,
            minimumPolicyGeneration: 0,
            verificationUnixSeconds: 0
        )
        let oversizedFile = Self.admittedFile(
            Data(repeating: 0x78, count: 2_200)
        )
        XCTAssertThrowsError(
            try GitToolPolicyV2Verifier.anchor(
                policyFile: oversizedFile,
                trustAnchor: invalidAnchor
            )
        ) { error in
            XCTAssertEqual(
                error as? GitToolPolicyV2Error,
                .invalidTrustAnchor(.expectedCurrentPolicySHA256)
            )
        }

        let contexts: [(GitToolPolicyV2TrustAnchor, GitToolPolicyV2Error)] = [
            (
                Self.trustAnchor(expectedSHA256: Self.otherSHA256),
                .policyDigestMismatch
            ),
            (
                Self.trustAnchor(expectedBytes: 1_913),
                .policyByteCountMismatch(expected: 1_913, actual: 1_912)
            ),
            (
                Self.trustAnchor(minimumGeneration: 8),
                .policyGenerationRollback(minimum: 8, actual: 7)
            ),
            (
                Self.trustAnchor(verificationUnixSeconds: 1_899_999_999),
                .policyNotYetValid
            ),
            (
                Self.trustAnchor(verificationUnixSeconds: 2_100_000_001),
                .policyExpired
            ),
        ]
        for (anchor, expected) in contexts {
            XCTAssertThrowsError(
                try GitToolPolicyV2Verifier.anchor(
                    policyFile: policyFile,
                    trustAnchor: anchor
                )
            ) { error in
                XCTAssertEqual(error as? GitToolPolicyV2Error, expected)
            }
        }

        for boundary in [UInt64(1_900_000_000), 2_100_000_000] {
            XCTAssertNoThrow(
                try GitToolPolicyV2Verifier.anchor(
                    policyFile: policyFile,
                    trustAnchor: Self.trustAnchor(
                        verificationUnixSeconds: boundary
                    )
                )
            )
        }

        let invalidAnchors: [
            (GitToolPolicyV2TrustAnchor, GitToolPolicyV2Error)
        ] = [
            (
                Self.trustAnchor(
                    expectedSHA256: Self.expectedPolicySHA256.uppercased()
                ),
                .invalidTrustAnchor(.expectedCurrentPolicySHA256)
            ),
            (
                Self.trustAnchor(expectedBytes: 0),
                .invalidTrustAnchor(.expectedCurrentPolicyBytes)
            ),
            (
                Self.trustAnchor(minimumGeneration: 0),
                .invalidTrustAnchor(.minimumPolicyGeneration)
            ),
        ]
        for (anchor, expected) in invalidAnchors {
            XCTAssertThrowsError(
                try GitToolPolicyV2Verifier.anchor(
                    policyFile: policyFile,
                    trustAnchor: anchor
                )
            ) { error in
                XCTAssertEqual(error as? GitToolPolicyV2Error, expected)
            }
        }
    }

    func testCompileTimeResourceCeilingsRejectEveryWideningAndAllowNarrowing()
        throws
    {
        let widened: [
            (
                GitToolPolicyV2ResourceField,
                key: String,
                maximum: UInt64,
                limits: GitToolPolicyResourceLimits
            )
        ] = [
            (.maxPackBytes, "max_pack_bytes", 134_217_728, Self.limits(maxPackBytes: 134_217_729)),
            (.maxPackObjects, "max_pack_objects", 20_000, Self.limits(maxPackObjects: 20_001)),
            (.maxCommitBytes, "max_commit_bytes", 1_048_576, Self.limits(maxCommitBytes: 1_048_577)),
            (.maxSingleInflatedObjectBytes, "max_single_inflated_object_bytes", 16_777_216, Self.limits(maxSingleInflatedObjectBytes: 16_777_217)),
            (.maxTotalInflatedBytes, "max_total_inflated_bytes", 268_435_456, Self.limits(maxTotalInflatedBytes: 268_435_457)),
            (.maxCompressionRatio, "max_compression_ratio", 64, Self.limits(maxCompressionRatio: 65)),
            (.maxTreeDepth, "max_tree_depth", 32, Self.limits(maxTreeDepth: 33)),
            (.maxTreeCount, "max_tree_count", 10_000, Self.limits(maxTreeCount: 10_001)),
            (.maxObjectDatabaseBytes, "max_odb_bytes", 167_772_160, Self.limits(maxObjectDatabaseBytes: 167_772_161)),
            (.maxStdoutBytes, "max_stdout_bytes", 67_108_864, Self.limits(maxStdoutBytes: 67_108_865)),
            (.maxStderrBytes, "max_stderr_bytes", 1_048_576, Self.limits(maxStderrBytes: 1_048_577)),
            (.wallTimeoutMilliseconds, "wall_timeout_milliseconds", 120_000, Self.limits(wallTimeoutMilliseconds: 120_001)),
            (.cpuTimeoutSeconds, "cpu_timeout_seconds", 120, Self.limits(cpuTimeoutSeconds: 121)),
            (.maxAddressSpaceBytes, "max_address_space_bytes", 1_073_741_824, Self.limits(maxAddressSpaceBytes: 1_073_741_825)),
            (.maxFileSizeBytes, "max_file_size_bytes", 167_772_160, Self.limits(maxFileSizeBytes: 167_772_161)),
            (.maxOpenFiles, "max_open_files", 64, Self.limits(maxOpenFiles: 65)),
        ]

        for item in widened {
            let expected = GitToolPolicyV2Error
                .resourceLimitExceedsCompileTimeCeiling(
                    field: item.0,
                    maximum: item.maximum,
                    actual: item.maximum + 1
                )
            XCTAssertThrowsError(
                try GitToolPolicyV2Verifier.policyBytes(
                    fields: Self.fields(limits: item.limits)
                )
            ) { error in
                XCTAssertEqual(error as? GitToolPolicyV2Error, expected)
            }

            let widenedBytes = Self.replacing(
                "\(item.key)=\(item.maximum)",
                with: "\(item.key)=\(item.maximum + 1)"
            )
            let widenedFile = Self.admittedFile(widenedBytes)
            XCTAssertThrowsError(
                try GitToolPolicyV2Verifier.anchor(
                    policyFile: widenedFile,
                    trustAnchor: Self.trustAnchor(matching: widenedFile)
                )
            ) { error in
                XCTAssertEqual(error as? GitToolPolicyV2Error, expected)
            }
        }

        let narrowedFields = Self.fields(
            limits: Self.limits(
                maxPackBytes: 67_108_864,
                maxPackObjects: 10_000,
                maxTreeDepth: 16
            )
        )
        let narrowedBytes = try GitToolPolicyV2Verifier.policyBytes(
            fields: narrowedFields
        )
        let narrowedFile = Self.admittedFile(narrowedBytes)
        let narrowed = try GitToolPolicyV2Verifier.anchor(
            policyFile: narrowedFile,
            trustAnchor: Self.trustAnchor(matching: narrowedFile)
        )
        XCTAssertEqual(narrowed.fields.limits, narrowedFields.limits)

        XCTAssertThrowsError(
            try GitToolPolicyV2Verifier.policyBytes(
                fields: Self.fields(
                    limits: Self.limits(maxPackObjects: 0)
                )
            )
        ) { error in
            XCTAssertEqual(error as? GitToolPolicyV2Error, .nonCanonicalPolicy)
        }
    }

    func testClaimReferenceRejectsDigestThenBytesThenRuntimeAndHasNoID()
        throws
    {
        let file = Self.admittedFile(Self.expectedPolicyBytes)
        let document = try GitToolPolicyV2Verifier.anchor(
            policyFile: file,
            trustAnchor: Self.trustAnchor(matching: file)
        )
        let digestMismatch = try Self.signedClaim(
            toolManifestSHA256: Self.otherSHA256,
            toolManifestBytes: document.policyBytes + 1,
            runtimePolicySHA256: Self.otherSHA256
        )
        let byteMismatch = try Self.signedClaim(
            toolManifestSHA256: document.policySHA256,
            toolManifestBytes: document.policyBytes + 1,
            runtimePolicySHA256: Self.otherSHA256
        )
        let runtimeMismatch = try Self.signedClaim(
            toolManifestSHA256: document.policySHA256,
            toolManifestBytes: document.policyBytes,
            runtimePolicySHA256: Self.otherSHA256
        )
        let valid = try Self.signedClaim(
            toolManifestSHA256: document.policySHA256,
            toolManifestBytes: document.policyBytes,
            runtimePolicySHA256: document.runtimePolicySHA256
        )

        let cases: [
            (OperatorSignedRunClaim, GitToolPolicyV2ClaimReferenceError)
        ] = [
            (digestMismatch, .toolManifestDigestMismatch),
            (byteMismatch, .toolManifestByteCountMismatch),
            (runtimeMismatch, .runtimePolicyDigestMismatch),
        ]
        for (claim, expected) in cases {
            XCTAssertThrowsError(
                try GitToolPolicyV2Verifier.reference(
                    signedClaim: claim,
                    policyDocument: document
                )
            ) { error in
                XCTAssertEqual(
                    error as? GitToolPolicyV2ClaimReferenceError,
                    expected
                )
            }
        }

        let reference = try GitToolPolicyV2Verifier.reference(
            signedClaim: valid,
            policyDocument: document
        )
        XCTAssertEqual(reference.signedClaim.claimID, valid.claimID)
        XCTAssertNotEqual(valid.claimID, digestMismatch.claimID)
        XCTAssertNotEqual(valid.claimID, byteMismatch.claimID)
        XCTAssertNotEqual(valid.claimID, runtimeMismatch.claimID)
        XCTAssertFalse(
            Mirror(reflecting: reference).children.contains {
                $0.label?.localizedCaseInsensitiveContains("id") == true
            }
        )
        Self.assertNoAuthority(reference)
        XCTAssertEqual(Self.effects, .zero)
    }
}

private extension GitToolPolicyV2DocumentTests {
    struct Effects: Equatable {
        var gitExecutions = 0
        var networkAccesses = 0
        var packConsumptions = 0
        var fileSystemMutations = 0
        var sourceImports = 0
        var builds = 0
        var modelLoads = 0
        var outputReservations = 0
        var publications = 0

        static let zero = Self()
    }

    struct TestKey {
        let privateKey: Curve25519.Signing.PrivateKey  // gitleaks:allow
        let publicKeyBase64: String
        let keyID: String
    }

    static let effects = Effects.zero
    static let executableSHA256 = String(repeating: "ab", count: 32)
    static let executableMachOUUID = String(repeating: "cd", count: 16)
    static let executableCodeDirectorySHA256 =
        String(repeating: "de", count: 32)
    static let gitRuntimeClosureManifestSHA256 =
        String(repeating: "bc", count: 32)
    static let gitFileImageRuntimeClosureContentEvidenceID =
        String(repeating: "12", count: 32)
    static let runtimePolicySHA256 = String(repeating: "ef", count: 32)
    static let otherSHA256 = String(repeating: "66", count: 32)
    static let expectedPolicySHA256 =
        "31480164530be831ac078b696c38706b07042f9defd4896da6b17cf6bc34748a"

    static func fields(
        limits: GitToolPolicyResourceLimits = limits()
    ) -> GitToolPolicyV2Fields {
        GitToolPolicyV2Fields(
            policyGeneration: 7,
            validFromUnixSeconds: 1_900_000_000,
            validUntilUnixSeconds: 2_100_000_000,
            executableSHA256: executableSHA256,
            executableBytes: 1_048_576,
            executableMachOUUID: executableMachOUUID,
            executableCodeDirectorySHA256:
                executableCodeDirectorySHA256,
            gitRuntimeClosureManifestSHA256:
                gitRuntimeClosureManifestSHA256,
            gitRuntimeClosureManifestBytes: 4_096,
            gitFileImageRuntimeClosureContentEvidenceID:
                gitFileImageRuntimeClosureContentEvidenceID,
            runtimePolicySHA256: runtimePolicySHA256,
            limits: limits
        )
    }

    static func limits(
        maxPackBytes: UInt64 = 134_217_728,
        maxPackObjects: UInt64 = 20_000,
        maxCommitBytes: UInt64 = 1_048_576,
        maxSingleInflatedObjectBytes: UInt64 = 16_777_216,
        maxTotalInflatedBytes: UInt64 = 268_435_456,
        maxCompressionRatio: UInt64 = 64,
        maxTreeDepth: UInt64 = 32,
        maxTreeCount: UInt64 = 10_000,
        maxObjectDatabaseBytes: UInt64 = 167_772_160,
        maxStdoutBytes: UInt64 = 67_108_864,
        maxStderrBytes: UInt64 = 1_048_576,
        wallTimeoutMilliseconds: UInt64 = 120_000,
        cpuTimeoutSeconds: UInt64 = 120,
        maxAddressSpaceBytes: UInt64 = 1_073_741_824,
        maxFileSizeBytes: UInt64 = 167_772_160,
        maxOpenFiles: UInt64 = 64
    ) -> GitToolPolicyResourceLimits {
        GitToolPolicyResourceLimits(
            maxPackBytes: maxPackBytes,
            maxPackObjects: maxPackObjects,
            maxCommitBytes: maxCommitBytes,
            maxSingleInflatedObjectBytes:
                maxSingleInflatedObjectBytes,
            maxTotalInflatedBytes: maxTotalInflatedBytes,
            maxCompressionRatio: maxCompressionRatio,
            maxTreeDepth: maxTreeDepth,
            maxTreeCount: maxTreeCount,
            maxObjectDatabaseBytes: maxObjectDatabaseBytes,
            maxStdoutBytes: maxStdoutBytes,
            maxStderrBytes: maxStderrBytes,
            wallTimeoutMilliseconds: wallTimeoutMilliseconds,
            cpuTimeoutSeconds: cpuTimeoutSeconds,
            maxAddressSpaceBytes: maxAddressSpaceBytes,
            maxFileSizeBytes: maxFileSizeBytes,
            maxOpenFiles: maxOpenFiles
        )
    }

    static let expectedPolicyBytes = Data(
        """
        fast-mlx-proof-control-git-tool-policy-v2
        subject=absorbed-mla-source-import-git
        policy_generation=7
        valid_from_unix_seconds=1900000000
        valid_until_unix_seconds=2100000000
        claim_subject=absorbed-mla-loaded-result-pair
        executable_sha256=\(executableSHA256)
        executable_bytes=1048576
        executable_architecture=arm64
        executable_macho_uuid=\(executableMachOUUID)
        executable_code_directory_sha256=\(executableCodeDirectorySHA256)
        git_runtime_closure_manifest_sha256=\(gitRuntimeClosureManifestSHA256)
        git_runtime_closure_manifest_bytes=4096
        git_file_image_runtime_closure_content_evidence_id=\(gitFileImageRuntimeClosureContentEvidenceID)
        runtime_policy_sha256=\(runtimePolicySHA256)
        argv_profile=git-source-pack-ingress-v1
        environment_profile=git-source-pack-environment-v1
        fd_profile=self-guard-fchdir-then-stdio-v1
        sandbox_profile=git-source-pack-sandbox-v1
        cwd_profile=descriptor-anchored-git-directory-v1
        umask=0077
        cloexec_default=true
        network_policy=deny-all-v1
        descendant_process_policy=deny-all-v1
        config_policy=none-v1
        hook_policy=deny-v1
        attributes_policy=raw-only-v1
        filter_policy=deny-v1
        textconv_policy=deny-v1
        replace_refs_policy=deny-v1
        alternates_policy=deny-v1
        promisor_lazy_fetch_policy=deny-v1
        transport_policy=deny-v1
        pack_version=2
        object_format=sha1
        delta_depth=0
        max_invocations_per_role=4
        max_pack_bytes=134217728
        max_pack_objects=20000
        max_commit_bytes=1048576
        max_single_inflated_object_bytes=16777216
        max_total_inflated_bytes=268435456
        max_compression_ratio=64
        max_tree_depth=32
        max_tree_count=10000
        max_odb_bytes=167772160
        max_stdout_bytes=67108864
        max_stderr_bytes=1048576
        wall_timeout_milliseconds=120000
        cpu_timeout_seconds=120
        max_address_space_bytes=1073741824
        max_file_size_bytes=167772160
        max_open_files=64

        """.utf8
    )

    static let maximumSchemaBytes: Data = {
        var text = String(decoding: expectedPolicyBytes, as: UTF8.self)
        let replacements = [
            ("policy_generation=7", "policy_generation=18446744073709551615"),
            ("valid_from_unix_seconds=1900000000", "valid_from_unix_seconds=18446744073709551615"),
            ("valid_until_unix_seconds=2100000000", "valid_until_unix_seconds=18446744073709551615"),
            ("executable_bytes=1048576", "executable_bytes=18446744073709551615"),
            ("git_runtime_closure_manifest_bytes=4096", "git_runtime_closure_manifest_bytes=18446744073709551615"),
            ("max_pack_bytes=134217728", "max_pack_bytes=18446744073709551615"),
            ("max_pack_objects=20000", "max_pack_objects=18446744073709551615"),
            ("max_commit_bytes=1048576", "max_commit_bytes=18446744073709551615"),
            ("max_single_inflated_object_bytes=16777216", "max_single_inflated_object_bytes=18446744073709551615"),
            ("max_total_inflated_bytes=268435456", "max_total_inflated_bytes=18446744073709551615"),
            ("max_compression_ratio=64", "max_compression_ratio=18446744073709551615"),
            ("max_tree_depth=32", "max_tree_depth=18446744073709551615"),
            ("max_tree_count=10000", "max_tree_count=18446744073709551615"),
            ("max_odb_bytes=167772160", "max_odb_bytes=18446744073709551615"),
            ("max_stdout_bytes=67108864", "max_stdout_bytes=18446744073709551615"),
            ("max_stderr_bytes=1048576", "max_stderr_bytes=18446744073709551615"),
            ("wall_timeout_milliseconds=120000", "wall_timeout_milliseconds=18446744073709551615"),
            ("cpu_timeout_seconds=120", "cpu_timeout_seconds=18446744073709551615"),
            ("max_address_space_bytes=1073741824", "max_address_space_bytes=18446744073709551615"),
            ("max_file_size_bytes=167772160", "max_file_size_bytes=18446744073709551615"),
            ("max_open_files=64", "max_open_files=18446744073709551615"),
        ]
        for (source, replacement) in replacements {
            text = text.replacingOccurrences(of: source, with: replacement)
        }
        return Data(text.utf8)
    }()

    static func trustAnchor(
        expectedSHA256: String = expectedPolicySHA256,
        expectedBytes: UInt64 = 1_912,
        minimumGeneration: UInt64 = 7,
        verificationUnixSeconds: UInt64 = 2_000_000_000
    ) -> GitToolPolicyV2TrustAnchor {
        GitToolPolicyV2TrustAnchor(
            expectedCurrentPolicySHA256: expectedSHA256,
            expectedCurrentPolicyBytes: expectedBytes,
            minimumPolicyGeneration: minimumGeneration,
            verificationUnixSeconds: verificationUnixSeconds
        )
    }

    static func trustAnchor(
        matching file: AdmittedFile,
        minimumGeneration: UInt64 = 7,
        verificationUnixSeconds: UInt64 = 2_000_000_000
    ) -> GitToolPolicyV2TrustAnchor {
        GitToolPolicyV2TrustAnchor(
            expectedCurrentPolicySHA256: file.sha256,
            expectedCurrentPolicyBytes: UInt64(file.bytes.count),
            minimumPolicyGeneration: minimumGeneration,
            verificationUnixSeconds: verificationUnixSeconds
        )
    }

    static func replacing(_ source: String, with replacement: String) -> Data {
        Data(
            String(decoding: expectedPolicyBytes, as: UTF8.self)
                .replacingOccurrences(of: source, with: replacement).utf8
        )
    }

    static func admittedFile(_ bytes: Data) -> AdmittedFile {
        AdmittedFile(
            bytes: bytes,
            sha256: sha256Hex(bytes),
            identity: AdmittedFileIdentity(
                device: 0,
                inode: 0,
                size: UInt64(bytes.count),
                linkCount: 1,
                mode: 0,
                modificationSeconds: 0,
                modificationNanoseconds: 0,
                changeSeconds: 0,
                changeNanoseconds: 0
            )
        )
    }

    static func signedClaim(
        toolManifestSHA256: String,
        toolManifestBytes: UInt64,
        runtimePolicySHA256: String
    ) throws -> OperatorSignedRunClaim {
        let rootKey = try testKey(seed: 0x10)
        let runKey = try testKey(seed: 0x30)
        let authorizationKey = try testKey(seed: 0x50)
        let worker = try authorizedFile(
            Data("worker-p1-v2\n".utf8),
            purpose: .workerBytes,
            key: authorizationKey
        )
        let baseline = try authorizedFile(
            Data("baseline-p1-v2\n".utf8),
            purpose: .sourceManifest,
            key: authorizationKey
        )
        let candidate = try authorizedFile(
            Data("candidate-p1-v2\n".utf8),
            purpose: .sourceManifest,
            key: authorizationKey
        )
        let inputs = try RunAuthorizedInputs(
            worker: worker,
            baselineSourceManifest: baseline,
            candidateSourceManifest: candidate
        )

        let keyFields = OperatorKeyPolicyFields(
            rootKeyID: rootKey.keyID,
            policyGeneration: 7,
            validFromUnixSeconds: 1_900_000_000,
            validUntilUnixSeconds: 2_100_000_000,
            activeOperatorKeyID: runKey.keyID,
            activeOperatorPublicKeyBase64: runKey.publicKeyBase64,
            activeOperatorScope: .runClaim,
            allowedClaimSubject: .absorbedMLALoadedResultPair,
            revokedOperatorKeyIDs: []
        )
        let keyPolicyBytes = try OperatorKeyPolicyVerifier.policyBytes(
            fields: keyFields
        )
        let keyPolicyFile = admittedFile(keyPolicyBytes)
        let keyPolicySignature = try rootKey.privateKey.signature(
            for: keyPolicyBytes
        ).base64EncodedString()
        let keyPolicy = try OperatorKeyPolicyVerifier.admit(
            policyFile: keyPolicyFile,
            rootSignatureBase64: keyPolicySignature,
            trustAnchor: OperatorKeyPolicyTrustAnchor(
                rootPublicKeyBase64: rootKey.publicKeyBase64,
                rootKeyID: rootKey.keyID,
                expectedCurrentPolicySHA256: keyPolicyFile.sha256,
                minimumPolicyGeneration: 7,
                verificationUnixSeconds: 2_000_000_000
            )
        )

        let runner = admittedFile(Data("runner-p1-v2\n".utf8))
        let expectations = OperatorRunClaimAdmissionExpectations(
            keyPolicy: keyPolicy,
            hostAdmissionID: hexByte(0x04),
            runner: runner,
            resultPairID: hexByte(0x05),
            inputs: inputs
        )
        let fields = OperatorRunClaimFields(
            subject: .absorbedMLALoadedResultPair,
            operatorKeyID: keyPolicy.activeOperatorKeyID,
            operatorKeyPolicySHA256: keyPolicy.policySHA256,
            hostAdmissionID: expectations.hostAdmissionID,
            runner: OperatorRunClaimByteIdentity(
                sha256: runner.sha256,
                byteCount: UInt64(runner.bytes.count)
            ),
            worker: authorizedReference(worker),
            policies: OperatorRunClaimPolicyReferences(
                sourceSHA256: hexByte(0x10),
                dependencySHA256: hexByte(0x11),
                buildSHA256: hexByte(0x12),
                runtimeSHA256: runtimePolicySHA256,
                preflightSHA256: hexByte(0x14),
                publicationSHA256: hexByte(0x15)
            ),
            toolManifest: OperatorRunClaimByteIdentity(
                sha256: toolManifestSHA256,
                byteCount: toolManifestBytes
            ),
            baseline: OperatorRunClaimSourceReference(
                role: .baseline,
                sourceManifest: authorizedReference(baseline),
                gitCommitSHA1: String(repeating: "a", count: 40),
                gitTreeSHA1: String(repeating: "b", count: 40),
                route: .decompressedDeepSeekV3,
                slot: .baseline,
                buildReceiptID: hexByte(0x17),
                binary: OperatorRunClaimByteIdentity(
                    sha256: hexByte(0x18),
                    byteCount: 1_001
                )
            ),
            candidate: OperatorRunClaimSourceReference(
                role: .candidate,
                sourceManifest: authorizedReference(candidate),
                gitCommitSHA1: String(repeating: "c", count: 40),
                gitTreeSHA1: String(repeating: "d", count: 40),
                route: .absorbedMLADeepSeekV3Explicit,
                slot: .candidate,
                buildReceiptID: hexByte(0x19),
                binary: OperatorRunClaimByteIdentity(
                    sha256: hexByte(0x20),
                    byteCount: 1_002
                )
            ),
            model: OperatorRunClaimAuthorizedPayloadReference(
                authorizationID: hexByte(0x21),
                payload: OperatorRunClaimByteIdentity(
                    sha256: hexByte(0x22),
                    byteCount: 55
                )
            ),
            tokenizer: OperatorRunClaimAuthorizedPayloadReference(
                authorizationID: hexByte(0x23),
                payload: OperatorRunClaimByteIdentity(
                    sha256: hexByte(0x24),
                    byteCount: 66
                )
            ),
            workload: OperatorRunClaimAuthorizedPayloadReference(
                authorizationID: hexByte(0x25),
                payload: OperatorRunClaimByteIdentity(
                    sha256: hexByte(0x26),
                    byteCount: 77
                )
            ),
            resultPairID: expectations.resultPairID
        )
        let claimBytes = try OperatorSignedRunClaimVerifier.claimBytes(
            fields: fields
        )
        let signature = try runKey.privateKey.signature(
            for: claimBytes
        ).base64EncodedString()
        return try OperatorSignedRunClaimVerifier.verify(
            claimBytes: claimBytes,
            signatureBase64: signature,
            expectations: expectations
        )
    }

    static func authorizedFile(
        _ bytes: Data,
        purpose: OperatorAuthorizationPurpose,
        key: TestKey
    ) throws -> OperatorAuthorizedFile {
        let file = admittedFile(bytes)
        let claim = try OperatorAuthorization.claimBytes(
            purpose: purpose,
            payloadSHA256: file.sha256,
            payloadByteCount: UInt64(file.bytes.count)
        )
        let signature = try key.privateKey.signature(
            for: claim
        ).base64EncodedString()
        return try OperatorAuthorization.verify(
            admittedFile: file,
            expectedPurpose: purpose,
            claimBytes: claim,
            signatureBase64: signature,
            publicKeyBase64: key.publicKeyBase64,
            allowedKeyID: key.keyID
        )
    }

    static func authorizedReference(
        _ file: OperatorAuthorizedFile
    ) -> OperatorRunClaimAuthorizedPayloadReference {
        OperatorRunClaimAuthorizedPayloadReference(
            authorizationID: file.authorizationID.rawValue,
            payload: OperatorRunClaimByteIdentity(
                sha256: file.file.sha256,
                byteCount: UInt64(file.file.bytes.count)
            )
        )
    }

    static func testKey(seed: UInt8) throws -> TestKey {
        let key = try Curve25519.Signing.PrivateKey(
            rawRepresentation: Data((0..<32).map { seed &+ UInt8($0) })
        )
        let publicBytes = key.publicKey.rawRepresentation
        return TestKey(
            privateKey: key,
            publicKeyBase64: publicBytes.base64EncodedString(),
            keyID: sha256Hex(publicBytes)
        )
    }

    static func hexByte(_ value: UInt8) -> String {
        String(repeating: String(format: "%02x", value), count: 32)
    }

    static func sha256Hex(_ bytes: Data) -> String {
        SHA256.hash(data: bytes)
            .map { String(format: "%02x", $0) }
            .joined()
    }

    static func assertNoAuthority(
        _ value: AnchoredGitToolPolicyV2Document,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertFalse(value.canExecute, file: file, line: line)
        XCTAssertFalse(value.canSpawn, file: file, line: line)
        XCTAssertFalse(value.canAccessNetwork, file: file, line: line)
        XCTAssertFalse(value.canConsumePack, file: file, line: line)
        XCTAssertFalse(value.canMutateFileSystem, file: file, line: line)
        XCTAssertFalse(value.canImportGitObjects, file: file, line: line)
        XCTAssertFalse(value.canBuild, file: file, line: line)
        XCTAssertFalse(value.canLoadModel, file: file, line: line)
        XCTAssertFalse(value.canReserveOutput, file: file, line: line)
        XCTAssertFalse(value.canPublish, file: file, line: line)
    }

    static func assertNoAuthority(
        _ value: SignedClaimGitToolPolicyV2Reference,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertFalse(value.canExecute, file: file, line: line)
        XCTAssertFalse(value.canSpawn, file: file, line: line)
        XCTAssertFalse(value.canAccessNetwork, file: file, line: line)
        XCTAssertFalse(value.canConsumePack, file: file, line: line)
        XCTAssertFalse(value.canMutateFileSystem, file: file, line: line)
        XCTAssertFalse(value.canImportGitObjects, file: file, line: line)
        XCTAssertFalse(value.canBuild, file: file, line: line)
        XCTAssertFalse(value.canLoadModel, file: file, line: line)
        XCTAssertFalse(value.canReserveOutput, file: file, line: line)
        XCTAssertFalse(value.canPublish, file: file, line: line)
    }
}
