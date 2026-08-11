import CryptoKit
import Foundation
import XCTest

@testable import ProofControl

final class GitRuntimePolicyDenialV2DocumentTests: XCTestCase {
    func testReviewedP2TypesExist() {
        _ = GitRuntimePolicyDenialV2Fields.self
        _ = GitRuntimePolicyDenialV2TrustAnchor.self
        _ = GitRuntimePolicyDenialV2TrustAnchorField.self
        _ = GitRuntimePolicyDenialV2ResourceField.self
        _ = GitRuntimePolicyV2ReferenceField.self
        _ = GitRuntimePolicyDenialV2Error.self
        _ = AnchoredGitRuntimePolicyDenialV2Document.self
        _ = GitRuntimePolicyDenialV2Verifier.self
    }

    func testCanonicalV2DenialPolicyAnchorsExactInertDocument() throws {
        XCTAssertEqual(GitRuntimePolicyDenialV2Verifier.maximumPolicyBytes, 2_643)
        XCTAssertEqual(GitRuntimePolicyDenialV2Verifier.maximumLineBytes, 122)
        XCTAssertEqual(Self.maximumSchemaBytes.count, 2_643)
        let maximumLines = Self.maximumSchemaBytes.split(
            separator: 0x0a,
            omittingEmptySubsequences: false
        )
        XCTAssertEqual(maximumLines.count, 58)
        XCTAssertTrue(maximumLines.last?.isEmpty == true)
        XCTAssertEqual(maximumLines.dropLast().map(\.count).max(), 122)

        let generated = try GitRuntimePolicyDenialV2Verifier.policyBytes(
            fields: Self.fields()
        )
        XCTAssertEqual(generated, Self.expectedPolicyBytes)
        XCTAssertEqual(generated.count, 2_453)
        XCTAssertEqual(
            Self.sha256Hex(generated),
            "2b47f20a02abe75f287148ebf487aeea315dd4f7a2918cacb0a8446334ebc7a9"
        )

        var callerBytes = generated
        let policyFile = Self.admittedFile(callerBytes)
        let document = try GitRuntimePolicyDenialV2Verifier.anchor(
            policyFile: policyFile,
            trustAnchor: Self.trustAnchor(matching: policyFile)
        )
        callerBytes.append(0x78)

        XCTAssertEqual(document.policyFile, policyFile)
        XCTAssertEqual(document.policyFile.bytes, Self.expectedPolicyBytes)
        XCTAssertEqual(document.policySHA256, policyFile.sha256)
        XCTAssertEqual(document.policyBytes, 2_453)
        XCTAssertEqual(document.fields, Self.fields())
        XCTAssertEqual(document.policyGeneration, 7)
        XCTAssertEqual(document.validFromUnixSeconds, 1_900_000_000)
        XCTAssertEqual(document.validUntilUnixSeconds, 2_100_000_000)
        XCTAssertEqual(document.selfGuardSHA256, Self.selfGuardSHA256)
        XCTAssertEqual(
            document.selfGuardFileImageRuntimeClosureContentEvidenceID,
            Self.selfGuardFileImageRuntimeClosureContentEvidenceID
        )
        XCTAssertEqual(
            document.gitFileImageRuntimeClosureContentEvidenceID,
            Self.gitFileImageRuntimeClosureContentEvidenceID
        )
        Self.assertNoAuthority(document)
        XCTAssertEqual(
            Set(Mirror(reflecting: document).children.compactMap(\.label)),
            [
                "policyFile", "policySHA256", "policyBytes",
                "policyGeneration", "validFromUnixSeconds",
                "validUntilUnixSeconds", "selfGuardSHA256",
                "selfGuardFileImageRuntimeClosureContentEvidenceID",
                "gitFileImageRuntimeClosureContentEvidenceID", "fields",
                "canExecute", "canSpawn", "canAccessNetwork",
                "canConsumePack", "canMutateFileSystem",
                "canImportGitObjects", "canBuild", "canLoadModel",
                "canReserveOutput", "canPublish",
            ]
        )
        XCTAssertFalse(
            Mirror(reflecting: document).children.contains {
                $0.label?.contains("ID") == true &&
                    $0.label !=
                    "selfGuardFileImageRuntimeClosureContentEvidenceID" &&
                    $0.label !=
                    "gitFileImageRuntimeClosureContentEvidenceID"
            }
        )
        XCTAssertFalse(
            String(decoding: document.policyFile.bytes, as: UTF8.self)
                .contains(document.policySHA256)
        )
        XCTAssertEqual(Self.effects, .zero)
    }

    func testRejectsMalformedOrderDuplicateUnknownCaseWhitespaceUTF8LineAndRange()
        throws
    {
        var reordered = String(decoding: Self.expectedPolicyBytes, as: UTF8.self)
            .split(separator: "\n", omittingEmptySubsequences: false)
        reordered.swapAt(2, 3)

        let malformed: [Data] = [
            Self.replacing(
                "fast-mlx-proof-control-git-runtime-policy-v2",
                with: "fast-mlx-proof-control-git-runtime-policy-v1"
            ),
            Self.replacing(
                "subject=absorbed-mla-source-import-git-runtime",
                with: "subject=other"
            ),
            Self.replacing("policy_generation=7", with: "policy_generation=07"),
            Self.replacing("policy_generation=7", with: "policy_generation=0"),
            Self.replacing(
                "policy_generation=7",
                with: "policy_generation=18446744073709551616"
            ),
            Self.replacing(
                "valid_until_unix_seconds=2100000000",
                with: "valid_until_unix_seconds=1899999999"
            ),
            Self.replacing(
                "platform_architecture=arm64",
                with: "platform_architecture=ARM64"
            ),
            Self.replacing(
                "platform_hardware_model=Mac15,14",
                with: " platform_hardware_model=Mac15,14"
            ),
            Self.replacing(
                "platform_hardware_model=Mac15,14",
                with: "platform_hardware_model=Mac15,14 "
            ),
            Self.replacing(
                "self_guard_sha256=\(Self.selfGuardSHA256)",
                with: "self_guard_sha256=\(Self.selfGuardSHA256.uppercased())"
            ),
            Self.replacing(
                "self_guard_bytes=1048576",
                with: "self_guard_bytes=0"
            ),
            Self.replacing(
                "self_guard_bytes=1048576",
                with: "self_guard_bytes=18446744073709551616"
            ),
            Self.replacing(
                "self_guard_macho_uuid=\(Self.selfGuardMachOUUID)",
                with: "self_guard_macho_uuid=\(Self.selfGuardMachOUUID.dropLast())"
            ),
            Self.replacing(
                "self_guard_code_directory_sha256=" +
                    Self.selfGuardCodeDirectorySHA256,
                with: "self_guard_code_directory_sha256=not-hex"
            ),
            Self.replacing(
                "self_guard_file_image_runtime_closure_content_evidence_id=" +
                    Self.selfGuardFileImageRuntimeClosureContentEvidenceID,
                with:
                    "self_guard_file_image_runtime_closure_content_evidence_id=" +
                    String(repeating: "AB", count: 32)
            ),
            Self.replacing(
                "git_file_image_runtime_closure_content_evidence_id=" +
                    Self.gitFileImageRuntimeClosureContentEvidenceID,
                with: "git_file_image_runtime_closure_content_evidence_id=not-hex"
            ),
            Self.replacing("umask=0077", with: "umask=077"),
            Self.replacing(
                "sandbox_policy_sha256=\(Self.zeroSHA256)",
                with: "sandbox_policy_sha256=\(Self.otherSHA256)"
            ),
            Self.replacing("sandbox_policy_bytes=0", with: "sandbox_policy_bytes=1"),
            Self.replacing("rlimit_core_bytes=0", with: "rlimit_core_bytes=1"),
            Self.replacing("runtime_decision=no-go", with: "runtime_decision=go"),
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
            Data([0xef, 0xbb, 0xbf]) + Self.expectedPolicyBytes,
            Data([0xff]),
            Data((String(repeating: "x", count: 123) + "\n").utf8),
        ]

        for (index, bytes) in malformed.enumerated() {
            let file = Self.admittedFile(bytes)
            XCTAssertThrowsError(
                try GitRuntimePolicyDenialV2Verifier.anchor(
                    policyFile: file,
                    trustAnchor: Self.trustAnchor(matching: file)
                ),
                "malformed case \(index)"
            ) { error in
                XCTAssertEqual(
                    error as? GitRuntimePolicyDenialV2Error,
                    .nonCanonicalPolicy
                )
            }
        }

        let oversized = Data(repeating: 0x78, count: 2_644)
        let oversizedFile = Self.admittedFile(oversized)
        XCTAssertThrowsError(
            try GitRuntimePolicyDenialV2Verifier.anchor(
                policyFile: oversizedFile,
                trustAnchor: Self.trustAnchor(matching: oversizedFile)
            )
        ) { error in
            XCTAssertEqual(error as? GitRuntimePolicyDenialV2Error, .policySize)
        }
    }

    func testRejectsEveryFixedProfilePolicyAndDecisionSubstitution() throws {
        let fixedLines = [
            "fast-mlx-proof-control-git-runtime-policy-v2",
            "subject=absorbed-mla-source-import-git-runtime",
            "platform_architecture=arm64",
            "platform_hardware_model=Mac15,14",
            "platform_os_version=26.5.2",
            "platform_os_build=25F84",
            "platform_xcode_version=26.6",
            "platform_xcode_build=17F113",
            "platform_sdk_version=26.5",
            "spawn_profile=posix-spawn-self-guard-v1",
            "spawn_flags_profile=cloexec-default-new-pgroup-clean-signals-v1",
            "tool_fd_profile=self-guard-fchdir-then-stdio-v1",
            "tool_cwd_profile=descriptor-anchored-git-directory-v1",
            "tool_sandbox_profile=git-source-pack-sandbox-v1",
            "spawn_setup_fd_profile=stdio-plus-directory-fd3-v1",
            "git_fd_profile=stdio-only-v1",
            "self_guard_profile=fchdir-revalidate-close-fd3-v1",
            "umask=0077",
            "environment_profile=git-source-pack-environment-v1",
            "argv_profile=git-source-pack-ingress-v1",
            "required_network_policy=deny-all-v1",
            "required_filesystem_policy=role-and-phase-exact-v1",
            "required_process_exec_policy=exact-admitted-git-only-v1",
            "required_descendant_process_policy=deny-all-v1",
            "sandbox_mechanism=none-no-supported-public-in-process-profile-v1",
            "sandbox_policy_sha256=\(Self.zeroSHA256)",
            "sandbox_policy_bytes=0",
            "network_enforcement=unavailable",
            "filesystem_enforcement=unavailable",
            "process_exec_enforcement=unavailable",
            "descendant_process_enforcement=unavailable",
            "resource_enforcement=partial-public-controls-not-complete-v1",
            "resource_profile=self-guard-rlimit-plus-parent-watchdog-v1",
            "rlimit_core_bytes=0",
            "watchdog_profile=monotonic-killpg-waitpid-v1",
            "compatibility_profile=macos-26-5-2-xcode-26-6-public-api-v1",
            "compatibility_outcome=no-go",
            "runtime_decision=no-go",
        ]

        for line in fixedLines {
            Self.assertAnchorRejects(
                Self.replacing(line, with: "\(line)-changed"),
                expected: .nonCanonicalPolicy
            )
        }
    }

    func testRejectsEveryUnprovedSandboxMechanismSubstitution() throws {
        let mechanisms = [
            "sandbox-exec",
            "sandbox_init",
            "seatbelt-profile",
            "app-sandbox",
            "endpoint-security",
            "network-extension",
            "chroot",
            "virtualization-framework",
        ]

        for mechanism in mechanisms {
            Self.assertAnchorRejects(
                Self.replacing(
                    "sandbox_mechanism=" +
                        "none-no-supported-public-in-process-profile-v1",
                    with: "sandbox_mechanism=\(mechanism)"
                ),
                expected: .nonCanonicalPolicy
            )
        }
    }

    func testAnchorRejectsInReviewedFirstFailureOrderAndIncludesBoundaries()
        throws
    {
        let policyFile = Self.admittedFile(Self.expectedPolicyBytes)
        let invalidAnchor = GitRuntimePolicyDenialV2TrustAnchor(
            expectedCurrentPolicySHA256:
                Self.expectedPolicySHA256.uppercased(),
            expectedCurrentPolicyBytes: 0,
            minimumPolicyGeneration: 0,
            verificationUnixSeconds: 0
        )
        let oversizedFile = Self.admittedFile(Data(repeating: 0x78, count: 2_644))
        XCTAssertThrowsError(
            try GitRuntimePolicyDenialV2Verifier.anchor(
                policyFile: oversizedFile,
                trustAnchor: invalidAnchor
            )
        ) { error in
            XCTAssertEqual(
                error as? GitRuntimePolicyDenialV2Error,
                .invalidTrustAnchor(.expectedCurrentPolicySHA256)
            )
        }

        let contexts: [(GitRuntimePolicyDenialV2TrustAnchor, GitRuntimePolicyDenialV2Error)] = [
            (
                Self.trustAnchor(expectedSHA256: Self.otherSHA256),
                .policyDigestMismatch
            ),
            (
                Self.trustAnchor(expectedBytes: 2_454),
                .policyByteCountMismatch(expected: 2_454, actual: 2_453)
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
                try GitRuntimePolicyDenialV2Verifier.anchor(
                    policyFile: policyFile,
                    trustAnchor: anchor
                )
            ) { error in
                XCTAssertEqual(error as? GitRuntimePolicyDenialV2Error, expected)
            }
        }

        for boundary in [UInt64(1_900_000_000), 2_100_000_000] {
            XCTAssertNoThrow(
                try GitRuntimePolicyDenialV2Verifier.anchor(
                    policyFile: policyFile,
                    trustAnchor: Self.trustAnchor(
                        verificationUnixSeconds: boundary
                    )
                )
            )
        }

        let invalidAnchors: [
            (GitRuntimePolicyDenialV2TrustAnchor, GitRuntimePolicyDenialV2Error)
        ] = [
            (
                Self.trustAnchor(expectedSHA256: Self.expectedPolicySHA256.uppercased()),
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
                try GitRuntimePolicyDenialV2Verifier.anchor(
                    policyFile: policyFile,
                    trustAnchor: anchor
                )
            ) { error in
                XCTAssertEqual(error as? GitRuntimePolicyDenialV2Error, expected)
            }
        }
    }

    func testRejectsResourceCeilingsWatchdogRelationshipsAndOverflow()
        throws
    {
        let widened: [
            (
                GitRuntimePolicyDenialV2ResourceField,
                key: String,
                maximum: UInt64,
                fields: GitRuntimePolicyDenialV2Fields
            )
        ] = [
            (.cpuTimeoutSeconds, "cpu_timeout_seconds", 120, Self.fields(limits: Self.limits(cpuTimeoutSeconds: 121))),
            (.maxAddressSpaceBytes, "max_address_space_bytes", 1_073_741_824, Self.fields(limits: Self.limits(maxAddressSpaceBytes: 1_073_741_825))),
            (.maxFileSizeBytes, "max_file_size_bytes", 167_772_160, Self.fields(limits: Self.limits(maxFileSizeBytes: 167_772_161))),
            (.maxOpenFiles, "max_open_files", 64, Self.fields(limits: Self.limits(maxOpenFiles: 65))),
            (.maxObjectDatabaseBytes, "max_odb_bytes", 167_772_160, Self.fields(limits: Self.limits(maxObjectDatabaseBytes: 167_772_161))),
            (.maxStdoutBytes, "max_stdout_bytes", 67_108_864, Self.fields(limits: Self.limits(maxStdoutBytes: 67_108_865))),
            (.maxStderrBytes, "max_stderr_bytes", 1_048_576, Self.fields(limits: Self.limits(maxStderrBytes: 1_048_577))),
            (.wallTimeoutMilliseconds, "wall_timeout_milliseconds", 120_000, Self.fields(limits: Self.limits(wallTimeoutMilliseconds: 120_001))),
            (.terminationGraceMilliseconds, "termination_grace_milliseconds", 2_000, Self.fields(terminationGraceMilliseconds: 2_001)),
            (.reapTimeoutMilliseconds, "reap_timeout_milliseconds", 5_000, Self.fields(reapTimeoutMilliseconds: 5_001)),
        ]

        for item in widened {
            let expected = GitRuntimePolicyDenialV2Error
                .resourceLimitExceedsCompileTimeCeiling(
                    field: item.0,
                    maximum: item.maximum,
                    actual: item.maximum + 1
                )
            XCTAssertThrowsError(
                try GitRuntimePolicyDenialV2Verifier.policyBytes(
                    fields: item.fields
                )
            ) { error in
                XCTAssertEqual(error as? GitRuntimePolicyDenialV2Error, expected)
            }

            let widenedBytes = Self.replacing(
                "\(item.key)=\(item.maximum)",
                with: "\(item.key)=\(item.maximum + 1)"
            )
            let widenedFile = Self.admittedFile(widenedBytes)
            XCTAssertThrowsError(
                try GitRuntimePolicyDenialV2Verifier.anchor(
                    policyFile: widenedFile,
                    trustAnchor: Self.trustAnchor(matching: widenedFile)
                )
            ) { error in
                XCTAssertEqual(error as? GitRuntimePolicyDenialV2Error, expected)
            }
        }

        let invalidFields: [
            (GitRuntimePolicyDenialV2Fields, GitRuntimePolicyDenialV2Error)
        ] = [
            (Self.fields(limits: Self.limits(cpuTimeoutSeconds: 0)), .nonCanonicalPolicy),
            (Self.fields(limits: Self.limits(maxAddressSpaceBytes: 0)), .nonCanonicalPolicy),
            (Self.fields(limits: Self.limits(maxFileSizeBytes: 0)), .nonCanonicalPolicy),
            (Self.fields(limits: Self.limits(maxOpenFiles: 0)), .nonCanonicalPolicy),
            (Self.fields(limits: Self.limits(maxObjectDatabaseBytes: 0)), .nonCanonicalPolicy),
            (Self.fields(limits: Self.limits(maxStdoutBytes: 0)), .nonCanonicalPolicy),
            (Self.fields(limits: Self.limits(maxStderrBytes: 0)), .nonCanonicalPolicy),
            (Self.fields(limits: Self.limits(wallTimeoutMilliseconds: 0)), .nonCanonicalPolicy),
            (Self.fields(terminationGraceMilliseconds: 0), .nonCanonicalPolicy),
            (Self.fields(reapTimeoutMilliseconds: 0), .nonCanonicalPolicy),
            (
                Self.fields(
                    terminationGraceMilliseconds: 2_000,
                    reapTimeoutMilliseconds: 2_000
                ),
                .invalidWatchdogInterval
            ),
            (
                Self.fields(
                    terminationGraceMilliseconds: 2_000,
                    reapTimeoutMilliseconds: 1_999
                ),
                .invalidWatchdogInterval
            ),
            (
                Self.fields(
                    limits: Self.limits(wallTimeoutMilliseconds: 4_000),
                    reapTimeoutMilliseconds: 5_000
                ),
                .invalidWatchdogInterval
            ),
            (
                Self.fields(
                    limits: Self.limits(
                        wallTimeoutMilliseconds: UInt64.max
                    ),
                    terminationGraceMilliseconds: UInt64.max,
                    reapTimeoutMilliseconds: UInt64.max
                ),
                .cleanupDeadlineOverflow
            ),
        ]

        for (fields, expected) in invalidFields {
            XCTAssertThrowsError(
                try GitRuntimePolicyDenialV2Verifier.policyBytes(fields: fields)
            ) { error in
                XCTAssertEqual(error as? GitRuntimePolicyDenialV2Error, expected)
            }
        }

        XCTAssertNoThrow(
            try GitRuntimePolicyDenialV2Verifier.policyBytes(
                fields: Self.fields(
                    limits: Self.limits(wallTimeoutMilliseconds: 120_000),
                    terminationGraceMilliseconds: 2_000,
                    reapTimeoutMilliseconds: 5_000
                )
            )
        )
        XCTAssertEqual(Self.effects, .zero)
    }

    func testRequireUnavailableJoinsFreshP1ReferenceDenialDigestResourcesAndTerminal()
        throws
    {
        let document = try Self.anchoredDocument()
        let reference = try Self.toolReference(
            runtimePolicySHA256: document.policySHA256
        )

        XCTAssertEqual(
            reference.signedClaim.fields.policies.runtimeSHA256,
            document.policySHA256
        )
        XCTAssertEqual(
            reference.policyDocument.runtimePolicySHA256,
            document.policySHA256
        )
        XCTAssertEqual(
            reference.policyDocument.policySHA256,
            reference.signedClaim.fields.toolManifest.sha256
        )
        XCTAssertEqual(
            reference.policyDocument.policyBytes,
            reference.signedClaim.fields.toolManifest.byteCount
        )
        XCTAssertThrowsError(
            try GitRuntimePolicyDenialV2Verifier.requireUnavailable(
                policyDocument: document,
                toolPolicyReference: reference
            )
        ) { error in
            XCTAssertEqual(
                error as? GitRuntimePolicyDenialV2Error,
                .supportedSandboxMechanismUnavailable
            )
        }
        Self.assertNoAuthority(document)
        Self.assertNoAuthority(reference)
        XCTAssertEqual(Self.effects, .zero)
    }

    func testRequireUnavailableRejectsRuntimeDigestBeforeResourceMismatches()
        throws
    {
        let document = try Self.anchoredDocument()
        let reference = try Self.toolReference(
            runtimePolicySHA256: Self.otherSHA256,
            limits: Self.limits(maxOpenFiles: 63)
        )

        XCTAssertThrowsError(
            try GitRuntimePolicyDenialV2Verifier.requireUnavailable(
                policyDocument: document,
                toolPolicyReference: reference
            )
        ) { error in
            XCTAssertEqual(
                error as? GitRuntimePolicyDenialV2Error,
                .runtimePolicyDigestMismatch
            )
        }
        XCTAssertEqual(Self.effects, .zero)
    }

    func testRequireUnavailableRejectsEveryDuplicatedToolResourceMismatch()
        throws
    {
        let document = try Self.anchoredDocument()
        let cases: [
            (GitRuntimePolicyDenialV2ResourceField, GitToolPolicyResourceLimits)
        ] = [
            (.cpuTimeoutSeconds, Self.limits(cpuTimeoutSeconds: 119)),
            (.maxAddressSpaceBytes, Self.limits(maxAddressSpaceBytes: 1_073_741_823)),
            (.maxFileSizeBytes, Self.limits(maxFileSizeBytes: 167_772_159)),
            (.maxOpenFiles, Self.limits(maxOpenFiles: 63)),
            (.maxObjectDatabaseBytes, Self.limits(maxObjectDatabaseBytes: 167_772_159)),
            (.maxStdoutBytes, Self.limits(maxStdoutBytes: 67_108_863)),
            (.maxStderrBytes, Self.limits(maxStderrBytes: 1_048_575)),
            (.wallTimeoutMilliseconds, Self.limits(wallTimeoutMilliseconds: 119_999)),
        ]

        for (field, limits) in cases {
            let reference = try Self.toolReference(
                runtimePolicySHA256: document.policySHA256,
                limits: limits
            )
            XCTAssertThrowsError(
                try GitRuntimePolicyDenialV2Verifier.requireUnavailable(
                    policyDocument: document,
                    toolPolicyReference: reference
                )
            ) { error in
                XCTAssertEqual(
                    error as? GitRuntimePolicyDenialV2Error,
                    .toolPolicyResourceMismatch(field)
                )
            }
            XCTAssertEqual(Self.effects, .zero)
        }
    }

    func testCommittedP1ReferenceRejectsDigestBytesAndRuntimeDiscontinuity()
        throws
    {
        let document = try Self.toolPolicyDocument(
            runtimePolicySHA256: Self.expectedPolicySHA256
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
        XCTAssertEqual(Self.effects, .zero)
    }

    func testV1RuntimeDenialIsDistinctAtRuntime() throws {
        let v1File = Self.admittedFile(Self.v1PolicyBytes)
        XCTAssertThrowsError(
            try GitRuntimePolicyDenialV2Verifier.anchor(
                policyFile: v1File,
                trustAnchor: Self.trustAnchor(matching: v1File)
            )
        ) { error in
            XCTAssertEqual(
                error as? GitRuntimePolicyDenialV2Error,
                .nonCanonicalPolicy
            )
        }
    }
}

private extension GitRuntimePolicyDenialV2DocumentTests {
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
    static let zeroSHA256 = String(repeating: "0", count: 64)
    static let selfGuardSHA256 = String(repeating: "a", count: 64)
    static let selfGuardMachOUUID = String(repeating: "b", count: 32)
    static let selfGuardCodeDirectorySHA256 =
        String(repeating: "c", count: 64)
    static let selfGuardFileImageRuntimeClosureContentEvidenceID =
        String(repeating: "d", count: 64)
    static let gitFileImageRuntimeClosureContentEvidenceID =
        String(repeating: "e", count: 64)
    static let executableSHA256 = String(repeating: "ab", count: 32)
    static let executableMachOUUID = String(repeating: "cd", count: 16)
    static let executableCodeDirectorySHA256 =
        String(repeating: "de", count: 32)
    static let gitRuntimeClosureManifestSHA256 =
        String(repeating: "bc", count: 32)
    static let otherSHA256 = String(repeating: "66", count: 32)
    static let expectedPolicySHA256 =
        "2b47f20a02abe75f287148ebf487aeea315dd4f7a2918cacb0a8446334ebc7a9"

    static func fields(
        limits: GitToolPolicyResourceLimits = limits(),
        terminationGraceMilliseconds: UInt64 = 2_000,
        reapTimeoutMilliseconds: UInt64 = 5_000
    ) -> GitRuntimePolicyDenialV2Fields {
        GitRuntimePolicyDenialV2Fields(
            policyGeneration: 7,
            validFromUnixSeconds: 1_900_000_000,
            validUntilUnixSeconds: 2_100_000_000,
            selfGuardSHA256: selfGuardSHA256,
            selfGuardBytes: 1_048_576,
            selfGuardMachOUUID: selfGuardMachOUUID,
            selfGuardCodeDirectorySHA256:
                selfGuardCodeDirectorySHA256,
            selfGuardFileImageRuntimeClosureContentEvidenceID:
                selfGuardFileImageRuntimeClosureContentEvidenceID,
            gitFileImageRuntimeClosureContentEvidenceID:
                gitFileImageRuntimeClosureContentEvidenceID,
            limits: limits,
            terminationGraceMilliseconds:
                terminationGraceMilliseconds,
            reapTimeoutMilliseconds: reapTimeoutMilliseconds
        )
    }

    static func limits(
        maxObjectDatabaseBytes: UInt64 = 167_772_160,
        maxStdoutBytes: UInt64 = 67_108_864,
        maxStderrBytes: UInt64 = 1_048_576,
        wallTimeoutMilliseconds: UInt64 = 120_000,
        cpuTimeoutSeconds: UInt64 = 120,
        maxAddressSpaceBytes: UInt64 = 1_073_741_824,
        maxFileSizeBytes: UInt64 = 167_772_160,
        maxOpenFiles: UInt64 = 64
    ) -> GitToolPolicyResourceLimits {
        let base = GitToolPolicyVerifier.phase1ResourceCeilings
        return GitToolPolicyResourceLimits(
            maxPackBytes: base.maxPackBytes,
            maxPackObjects: base.maxPackObjects,
            maxCommitBytes: base.maxCommitBytes,
            maxSingleInflatedObjectBytes:
                base.maxSingleInflatedObjectBytes,
            maxTotalInflatedBytes: base.maxTotalInflatedBytes,
            maxCompressionRatio: base.maxCompressionRatio,
            maxTreeDepth: base.maxTreeDepth,
            maxTreeCount: base.maxTreeCount,
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
        fast-mlx-proof-control-git-runtime-policy-v2
        subject=absorbed-mla-source-import-git-runtime
        policy_generation=7
        valid_from_unix_seconds=1900000000
        valid_until_unix_seconds=2100000000
        platform_architecture=arm64
        platform_hardware_model=Mac15,14
        platform_os_version=26.5.2
        platform_os_build=25F84
        platform_xcode_version=26.6
        platform_xcode_build=17F113
        platform_sdk_version=26.5
        self_guard_sha256=\(selfGuardSHA256)
        self_guard_bytes=1048576
        self_guard_macho_uuid=\(selfGuardMachOUUID)
        self_guard_code_directory_sha256=\(selfGuardCodeDirectorySHA256)
        self_guard_file_image_runtime_closure_content_evidence_id=\(selfGuardFileImageRuntimeClosureContentEvidenceID)
        git_file_image_runtime_closure_content_evidence_id=\(gitFileImageRuntimeClosureContentEvidenceID)
        spawn_profile=posix-spawn-self-guard-v1
        spawn_flags_profile=cloexec-default-new-pgroup-clean-signals-v1
        tool_fd_profile=self-guard-fchdir-then-stdio-v1
        tool_cwd_profile=descriptor-anchored-git-directory-v1
        tool_sandbox_profile=git-source-pack-sandbox-v1
        spawn_setup_fd_profile=stdio-plus-directory-fd3-v1
        git_fd_profile=stdio-only-v1
        self_guard_profile=fchdir-revalidate-close-fd3-v1
        umask=0077
        environment_profile=git-source-pack-environment-v1
        argv_profile=git-source-pack-ingress-v1
        required_network_policy=deny-all-v1
        required_filesystem_policy=role-and-phase-exact-v1
        required_process_exec_policy=exact-admitted-git-only-v1
        required_descendant_process_policy=deny-all-v1
        sandbox_mechanism=none-no-supported-public-in-process-profile-v1
        sandbox_policy_sha256=\(zeroSHA256)
        sandbox_policy_bytes=0
        network_enforcement=unavailable
        filesystem_enforcement=unavailable
        process_exec_enforcement=unavailable
        descendant_process_enforcement=unavailable
        resource_enforcement=partial-public-controls-not-complete-v1
        resource_profile=self-guard-rlimit-plus-parent-watchdog-v1
        rlimit_core_bytes=0
        cpu_timeout_seconds=120
        max_address_space_bytes=1073741824
        max_file_size_bytes=167772160
        max_open_files=64
        max_odb_bytes=167772160
        max_stdout_bytes=67108864
        max_stderr_bytes=1048576
        wall_timeout_milliseconds=120000
        termination_grace_milliseconds=2000
        reap_timeout_milliseconds=5000
        watchdog_profile=monotonic-killpg-waitpid-v1
        compatibility_profile=macos-26-5-2-xcode-26-6-public-api-v1
        compatibility_outcome=no-go
        runtime_decision=no-go

        """.utf8
    )

    static let maximumSchemaBytes: Data = {
        var text = String(decoding: expectedPolicyBytes, as: UTF8.self)
        let replacements = [
            ("policy_generation=7", "policy_generation=18446744073709551615"),
            ("valid_from_unix_seconds=1900000000", "valid_from_unix_seconds=18446744073709551615"),
            ("valid_until_unix_seconds=2100000000", "valid_until_unix_seconds=18446744073709551615"),
            ("self_guard_bytes=1048576", "self_guard_bytes=18446744073709551615"),
            ("cpu_timeout_seconds=120", "cpu_timeout_seconds=18446744073709551615"),
            ("max_address_space_bytes=1073741824", "max_address_space_bytes=18446744073709551615"),
            ("max_file_size_bytes=167772160", "max_file_size_bytes=18446744073709551615"),
            ("max_open_files=64", "max_open_files=18446744073709551615"),
            ("max_odb_bytes=167772160", "max_odb_bytes=18446744073709551615"),
            ("max_stdout_bytes=67108864", "max_stdout_bytes=18446744073709551615"),
            ("max_stderr_bytes=1048576", "max_stderr_bytes=18446744073709551615"),
            ("wall_timeout_milliseconds=120000", "wall_timeout_milliseconds=18446744073709551615"),
            ("termination_grace_milliseconds=2000", "termination_grace_milliseconds=18446744073709551615"),
            ("reap_timeout_milliseconds=5000", "reap_timeout_milliseconds=18446744073709551615"),
        ]
        for (source, replacement) in replacements {
            text = text.replacingOccurrences(of: source, with: replacement)
        }
        return Data(text.utf8)
    }()

    static let v1PolicyBytes = Data(
        String(decoding: expectedPolicyBytes, as: UTF8.self)
            .replacingOccurrences(
                of: "fast-mlx-proof-control-git-runtime-policy-v2",
                with: "fast-mlx-proof-control-git-runtime-policy-v1"
            )
            .replacingOccurrences(
                of: "self_guard_file_image_runtime_closure_content_evidence_id",
                with: "self_guard_runtime_closure_id"
            )
            .replacingOccurrences(
                of: "git_file_image_runtime_closure_content_evidence_id",
                with: "git_runtime_closure_id"
            )
            .utf8
    )

    static func anchoredDocument()
        throws -> AnchoredGitRuntimePolicyDenialV2Document
    {
        let file = admittedFile(expectedPolicyBytes)
        return try GitRuntimePolicyDenialV2Verifier.anchor(
            policyFile: file,
            trustAnchor: trustAnchor(matching: file)
        )
    }

    static func trustAnchor(
        expectedSHA256: String = expectedPolicySHA256,
        expectedBytes: UInt64 = 2_453,
        minimumGeneration: UInt64 = 7,
        verificationUnixSeconds: UInt64 = 2_000_000_000
    ) -> GitRuntimePolicyDenialV2TrustAnchor {
        GitRuntimePolicyDenialV2TrustAnchor(
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
    ) -> GitRuntimePolicyDenialV2TrustAnchor {
        GitRuntimePolicyDenialV2TrustAnchor(
            expectedCurrentPolicySHA256: file.sha256,
            expectedCurrentPolicyBytes: UInt64(file.bytes.count),
            minimumPolicyGeneration: minimumGeneration,
            verificationUnixSeconds: verificationUnixSeconds
        )
    }

    static func toolReference(
        runtimePolicySHA256: String,
        limits: GitToolPolicyResourceLimits = limits()
    ) throws -> SignedClaimGitToolPolicyV2Reference {
        let document = try toolPolicyDocument(
            runtimePolicySHA256: runtimePolicySHA256,
            limits: limits
        )
        let claim = try signedClaim(
            toolManifestSHA256: document.policySHA256,
            toolManifestBytes: document.policyBytes,
            runtimePolicySHA256: runtimePolicySHA256
        )
        return try GitToolPolicyV2Verifier.reference(
            signedClaim: claim,
            policyDocument: document
        )
    }

    static func toolPolicyDocument(
        runtimePolicySHA256: String,
        limits: GitToolPolicyResourceLimits = limits()
    ) throws -> AnchoredGitToolPolicyV2Document {
        let fields = GitToolPolicyV2Fields(
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
        let bytes = try GitToolPolicyV2Verifier.policyBytes(fields: fields)
        let file = admittedFile(bytes)
        return try GitToolPolicyV2Verifier.anchor(
            policyFile: file,
            trustAnchor: GitToolPolicyV2TrustAnchor(
                expectedCurrentPolicySHA256: file.sha256,
                expectedCurrentPolicyBytes: UInt64(file.bytes.count),
                minimumPolicyGeneration: 7,
                verificationUnixSeconds: 2_000_000_000
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
            Data("worker-p2-v2\n".utf8),
            purpose: .workerBytes,
            key: authorizationKey
        )
        let baseline = try authorizedFile(
            Data("baseline-p2-v2\n".utf8),
            purpose: .sourceManifest,
            key: authorizationKey
        )
        let candidate = try authorizedFile(
            Data("candidate-p2-v2\n".utf8),
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

        let runner = admittedFile(Data("runner-p2-v2\n".utf8))
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

    static func replacing(_ source: String, with replacement: String) -> Data {
        Data(
            String(decoding: expectedPolicyBytes, as: UTF8.self)
                .replacingOccurrences(of: source, with: replacement).utf8
        )
    }

    static func assertAnchorRejects(
        _ bytes: Data,
        expected: GitRuntimePolicyDenialV2Error,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let policyFile = admittedFile(bytes)
        XCTAssertThrowsError(
            try GitRuntimePolicyDenialV2Verifier.anchor(
                policyFile: policyFile,
                trustAnchor: trustAnchor(matching: policyFile)
            ),
            file: file,
            line: line
        ) { error in
            XCTAssertEqual(
                error as? GitRuntimePolicyDenialV2Error,
                expected,
                file: file,
                line: line
            )
        }
        XCTAssertEqual(effects, .zero, file: file, line: line)
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
        _ value: AnchoredGitRuntimePolicyDenialV2Document,
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
