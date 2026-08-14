import Darwin
import Foundation
import XCTest
@testable import ProofControl

final class GitRuntimePolicyDenialDocumentTests: XCTestCase {
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
            "fast-mlx-git-runtime-denial-\(UUID().uuidString)"
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

    func testCanonicalDenialPolicyAnchorsOnlyInertExactDocument() throws {
        let effects = FakeOperationalEffects()
        let expectedBytes = Self.expectedPolicyBytes
        XCTAssertEqual(
            try GitRuntimePolicyDenialVerifier.policyBytes(
                fields: Self.fields()
            ),
            expectedBytes
        )
        XCTAssertEqual(
            String(decoding: expectedBytes, as: UTF8.self)
                .split(
                    separator: "\n",
                    omittingEmptySubsequences: false
                )
                .count,
            58
        )

        let policyFile = try capturePolicy(expectedBytes)
        XCTAssertEqual(policyFile.sha256, Self.expectedPolicySHA256)
        XCTAssertEqual(
            UInt64(policyFile.bytes.count),
            Self.expectedPolicyByteCount
        )
        let document = try GitRuntimePolicyDenialVerifier.anchor(
            policyFile: policyFile,
            trustAnchor: trustAnchorMatching(policyFile)
        )

        XCTAssertEqual(document.policyFile, policyFile)
        XCTAssertEqual(document.policySHA256, policyFile.sha256)
        XCTAssertEqual(document.policyBytes, UInt64(expectedBytes.count))
        XCTAssertEqual(document.fields, Self.fields())
        XCTAssertFalse(document.canImportGitObjects)
        XCTAssertFalse(document.canMutateFileSystem)
        XCTAssertFalse(document.canAccessNetwork)
        XCTAssertFalse(document.canConsumePack)
        XCTAssertFalse(document.canSpawn)
        XCTAssertFalse(document.canBuild)
        XCTAssertFalse(document.canLoadModel)
        XCTAssertFalse(document.canReserveOutput)
        XCTAssertFalse(document.canPublish)
        XCTAssertEqual(effects, .zero)
    }

    func testAnchoredDenialRetainsCapturedBytesAfterBackingRewrite()
        throws
    {
        let url = caseRoot.appendingPathComponent(
            "\(UUID().uuidString).git-runtime-policy"
        )
        try Self.expectedPolicyBytes.write(to: url)
        let policyFile = try AdmittedFile.capture(
            absolutePath: url.path,
            maximumBytes: 16 * 1024
        )
        let document = try GitRuntimePolicyDenialVerifier.anchor(
            policyFile: policyFile,
            trustAnchor: trustAnchorMatching(policyFile)
        )

        try Data("rewritten\n".utf8).write(
            to: url
        )

        XCTAssertEqual(document.policyFile.bytes, Self.expectedPolicyBytes)
        XCTAssertEqual(document.policySHA256, policyFile.sha256)
        XCTAssertEqual(
            document.policyBytes,
            UInt64(Self.expectedPolicyBytes.count)
        )
        XCTAssertFalse(document.canSpawn)
        XCTAssertFalse(document.canMutateFileSystem)
    }

    func testAnchorRejectsEveryExternalTrustMismatchBeforeEffects() throws {
        let policyFile = try capturePolicy(Self.expectedPolicyBytes)
        let effects = FakeOperationalEffects()
        let cases: [
            (
                GitRuntimePolicyDenialTrustAnchor,
                GitRuntimePolicyDenialError
            )
        ] = [
            (
                trustAnchorMatching(
                    policyFile,
                    expectedSHA256: Self.uppercaseSHA256
                ),
                .invalidTrustAnchor(.expectedCurrentPolicySHA256)
            ),
            (
                trustAnchorMatching(policyFile, expectedBytes: 0),
                .invalidTrustAnchor(.expectedCurrentPolicyBytes)
            ),
            (
                trustAnchorMatching(policyFile, minimumGeneration: 0),
                .invalidTrustAnchor(.minimumPolicyGeneration)
            ),
            (
                trustAnchorMatching(
                    policyFile,
                    expectedSHA256: Self.otherSHA256
                ),
                .policyDigestMismatch
            ),
            (
                trustAnchorMatching(
                    policyFile,
                    expectedBytes: UInt64(policyFile.bytes.count) + 1
                ),
                .policyByteCountMismatch(
                    expected: UInt64(policyFile.bytes.count) + 1,
                    actual: UInt64(policyFile.bytes.count)
                )
            ),
            (
                trustAnchorMatching(policyFile, minimumGeneration: 8),
                .policyGenerationRollback(minimum: 8, actual: 7)
            ),
            (
                trustAnchorMatching(
                    policyFile,
                    verificationUnixSeconds: 1_899_999_999
                ),
                .policyNotYetValid
            ),
            (
                trustAnchorMatching(
                    policyFile,
                    verificationUnixSeconds: 2_100_000_001
                ),
                .policyExpired
            ),
        ]

        for (anchor, expectedError) in cases {
            XCTAssertThrowsError(
                try GitRuntimePolicyDenialVerifier.anchor(
                    policyFile: policyFile,
                    trustAnchor: anchor
                )
            ) { error in
                XCTAssertEqual(
                    error as? GitRuntimePolicyDenialError,
                    expectedError
                )
            }
            XCTAssertEqual(effects, .zero)
        }

        XCTAssertNoThrow(
            try GitRuntimePolicyDenialVerifier.anchor(
                policyFile: policyFile,
                trustAnchor: trustAnchorMatching(
                    policyFile,
                    verificationUnixSeconds: 1_900_000_000
                )
            )
        )
        XCTAssertNoThrow(
            try GitRuntimePolicyDenialVerifier.anchor(
                policyFile: policyFile,
                trustAnchor: trustAnchorMatching(
                    policyFile,
                    verificationUnixSeconds: 2_100_000_000
                )
            )
        )
        XCTAssertEqual(effects, .zero)
    }

    func testRejectsMalformedOrderDuplicateUnknownEncodingAndScalars()
        throws
    {
        var reordered = String(
            decoding: Self.expectedPolicyBytes,
            as: UTF8.self
        )
        .split(separator: "\n", omittingEmptySubsequences: false)
        reordered.swapAt(2, 3)

        let cases: [Data] = [
            Data(Self.expectedPolicyBytes.dropLast()),
            Data(
                String(
                    decoding: Self.expectedPolicyBytes,
                    as: UTF8.self
                )
                .replacingOccurrences(of: "\n", with: "\r\n")
                .utf8
            ),
            Data([0xef, 0xbb, 0xbf]) + Self.expectedPolicyBytes,
            Data([0xff]),
            Self.expectedPolicyBytes + Data("unknown=true\n".utf8),
            replacing(
                "subject=absorbed-mla-source-import-git-runtime\n",
                with: ""
            ),
            Data(
                (reordered.map(String.init).joined(separator: "\n"))
                    .utf8
            ),
            replacing(
                "subject=absorbed-mla-source-import-git-runtime",
                with:
                    "subject=absorbed-mla-source-import-git-runtime\n" +
                    "subject=absorbed-mla-source-import-git-runtime"
            ),
            replacing(
                "subject=absorbed-mla-source-import-git-runtime",
                with: "unknown_subject=absorbed-mla-source-import-git-runtime"
            ),
            replacing(
                "platform_architecture=arm64",
                with: "platform_architecture=ARM64"
            ),
            replacing(
                "platform_hardware_model=Mac15,14",
                with: " platform_hardware_model=Mac15,14"
            ),
            replacing(
                "platform_hardware_model=Mac15,14",
                with: "platform_hardware_model=Mac15,14 "
            ),
            replacing(
                "policy_generation=7",
                with: "policy_generation=07"
            ),
            replacing(
                "policy_generation=7",
                with: "policy_generation=0"
            ),
            replacing(
                "policy_generation=7",
                with: "policy_generation=18446744073709551616"
            ),
            replacing(
                "valid_until_unix_seconds=2100000000",
                with: "valid_until_unix_seconds=1899999999"
            ),
            replacing(
                "self_guard_sha256=\(Self.selfGuardSHA256)",
                with:
                    "self_guard_sha256=" +
                    Self.selfGuardSHA256.uppercased()
            ),
            replacing(
                "self_guard_bytes=1048576",
                with: "self_guard_bytes=0"
            ),
            replacing(
                "self_guard_macho_uuid=\(Self.selfGuardMachOUUID)",
                with:
                    "self_guard_macho_uuid=" +
                    Self.selfGuardMachOUUID.dropLast()
            ),
            replacing(
                "git_runtime_closure_id=\(Self.gitRuntimeClosureID)",
                with: "git_runtime_closure_id=not-hex"
            ),
            replacing(
                "rlimit_core_bytes=0",
                with: "rlimit_core_bytes=1"
            ),
            replacing(
                "sandbox_policy_sha256=\(Self.zeroSHA256)",
                with: "sandbox_policy_sha256=\(Self.otherSHA256)"
            ),
            replacing(
                "sandbox_policy_bytes=0",
                with: "sandbox_policy_bytes=1"
            ),
            replacing(
                "runtime_decision=no-go",
                with: "runtime_decision=go"
            ),
        ]

        for bytes in cases {
            assertAnchorRejects(bytes, expected: .nonCanonicalPolicy)
        }
    }

    func testRejectsEveryFixedPlatformProfileAndDecisionSubstitution()
        throws
    {
        let fixedLines = [
            "fast-mlx-proof-control-git-runtime-policy-v1",
            "subject=absorbed-mla-source-import-git-runtime",
            "platform_architecture=arm64",
            "platform_hardware_model=Mac15,14",
            "platform_os_version=26.5.2",
            "platform_os_build=25F84",
            "platform_xcode_version=26.6",
            "platform_xcode_build=17F113",
            "platform_sdk_version=26.5",
            "spawn_profile=posix-spawn-self-guard-v1",
            "spawn_flags_profile=" +
                "cloexec-default-new-pgroup-clean-signals-v1",
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
            "sandbox_mechanism=" +
                "none-no-supported-public-in-process-profile-v1",
            "sandbox_policy_sha256=\(Self.zeroSHA256)",
            "sandbox_policy_bytes=0",
            "network_enforcement=unavailable",
            "filesystem_enforcement=unavailable",
            "process_exec_enforcement=unavailable",
            "descendant_process_enforcement=unavailable",
            "resource_enforcement=" +
                "partial-public-controls-not-complete-v1",
            "resource_profile=" +
                "self-guard-rlimit-plus-parent-watchdog-v1",
            "rlimit_core_bytes=0",
            "watchdog_profile=monotonic-killpg-waitpid-v1",
            "compatibility_profile=" +
                "macos-26-5-2-xcode-26-6-public-api-v1",
            "compatibility_outcome=no-go",
            "runtime_decision=no-go",
        ]

        for line in fixedLines {
            assertAnchorRejects(
                replacing(line, with: "\(line)-changed"),
                expected: .nonCanonicalPolicy
            )
        }
    }

    func testRejectsEveryUnprovedSandboxMechanismSubstitution() throws {
        let substitutions = [
            "sandbox-exec",
            "sandbox_init",
            "seatbelt-profile",
            "app-sandbox",
            "endpoint-security",
            "network-extension",
            "chroot",
            "virtualization-framework",
        ]

        for mechanism in substitutions {
            assertAnchorRejects(
                replacing(
                    "sandbox_mechanism=" +
                    "none-no-supported-public-in-process-profile-v1",
                    with: "sandbox_mechanism=\(mechanism)"
                ),
                expected: .nonCanonicalPolicy
            )
        }
    }

    func testRejectsResourceCeilingsWatchdogRelationshipsAndOverflow()
        throws
    {
        let effects = FakeOperationalEffects()
        let cases: [
            (
                GitRuntimePolicyDenialFields,
                GitRuntimePolicyDenialError
            )
        ] = [
            (
                Self.fields(
                    limits: Self.limits(cpuTimeoutSeconds: 121)
                ),
                .resourceLimitExceedsCompileTimeCeiling(
                    field: .cpuTimeoutSeconds,
                    maximum: 120,
                    actual: 121
                )
            ),
            (
                Self.fields(
                    limits: Self.limits(
                        maxAddressSpaceBytes: 1_073_741_825
                    )
                ),
                .resourceLimitExceedsCompileTimeCeiling(
                    field: .maxAddressSpaceBytes,
                    maximum: 1_073_741_824,
                    actual: 1_073_741_825
                )
            ),
            (
                Self.fields(
                    limits: Self.limits(maxFileSizeBytes: 167_772_161)
                ),
                .resourceLimitExceedsCompileTimeCeiling(
                    field: .maxFileSizeBytes,
                    maximum: 167_772_160,
                    actual: 167_772_161
                )
            ),
            (
                Self.fields(limits: Self.limits(maxOpenFiles: 65)),
                .resourceLimitExceedsCompileTimeCeiling(
                    field: .maxOpenFiles,
                    maximum: 64,
                    actual: 65
                )
            ),
            (
                Self.fields(
                    limits: Self.limits(
                        maxObjectDatabaseBytes: 167_772_161
                    )
                ),
                .resourceLimitExceedsCompileTimeCeiling(
                    field: .maxObjectDatabaseBytes,
                    maximum: 167_772_160,
                    actual: 167_772_161
                )
            ),
            (
                Self.fields(
                    limits: Self.limits(maxStdoutBytes: 67_108_865)
                ),
                .resourceLimitExceedsCompileTimeCeiling(
                    field: .maxStdoutBytes,
                    maximum: 67_108_864,
                    actual: 67_108_865
                )
            ),
            (
                Self.fields(
                    limits: Self.limits(maxStderrBytes: 1_048_577)
                ),
                .resourceLimitExceedsCompileTimeCeiling(
                    field: .maxStderrBytes,
                    maximum: 1_048_576,
                    actual: 1_048_577
                )
            ),
            (
                Self.fields(
                    limits: Self.limits(
                        wallTimeoutMilliseconds: 120_001
                    )
                ),
                .resourceLimitExceedsCompileTimeCeiling(
                    field: .wallTimeoutMilliseconds,
                    maximum: 120_000,
                    actual: 120_001
                )
            ),
            (
                Self.fields(
                    limits: Self.limits(cpuTimeoutSeconds: 0)
                ),
                .nonCanonicalPolicy
            ),
            (
                Self.fields(
                    limits: Self.limits(maxAddressSpaceBytes: 0)
                ),
                .nonCanonicalPolicy
            ),
            (
                Self.fields(
                    limits: Self.limits(maxFileSizeBytes: 0)
                ),
                .nonCanonicalPolicy
            ),
            (
                Self.fields(limits: Self.limits(maxOpenFiles: 0)),
                .nonCanonicalPolicy
            ),
            (
                Self.fields(
                    limits: Self.limits(maxObjectDatabaseBytes: 0)
                ),
                .nonCanonicalPolicy
            ),
            (
                Self.fields(
                    limits: Self.limits(maxStdoutBytes: 0)
                ),
                .nonCanonicalPolicy
            ),
            (
                Self.fields(
                    limits: Self.limits(maxStderrBytes: 0)
                ),
                .nonCanonicalPolicy
            ),
            (
                Self.fields(
                    limits: Self.limits(wallTimeoutMilliseconds: 0)
                ),
                .nonCanonicalPolicy
            ),
            (
                Self.fields(terminationGraceMilliseconds: 0),
                .nonCanonicalPolicy
            ),
            (
                Self.fields(reapTimeoutMilliseconds: 0),
                .nonCanonicalPolicy
            ),
            (
                Self.fields(terminationGraceMilliseconds: 2_001),
                .resourceLimitExceedsCompileTimeCeiling(
                    field: .terminationGraceMilliseconds,
                    maximum: 2_000,
                    actual: 2_001
                )
            ),
            (
                Self.fields(reapTimeoutMilliseconds: 5_001),
                .resourceLimitExceedsCompileTimeCeiling(
                    field: .reapTimeoutMilliseconds,
                    maximum: 5_000,
                    actual: 5_001
                )
            ),
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

        for (fields, expectedError) in cases {
            XCTAssertThrowsError(
                try GitRuntimePolicyDenialVerifier.policyBytes(
                    fields: fields
                )
            ) { error in
                XCTAssertEqual(
                    error as? GitRuntimePolicyDenialError,
                    expectedError
                )
            }
            XCTAssertEqual(effects, .zero)
        }

        XCTAssertNoThrow(
            try GitRuntimePolicyDenialVerifier.policyBytes(
                fields: Self.fields(
                    limits: Self.limits(
                        wallTimeoutMilliseconds: 120_000
                    ),
                    terminationGraceMilliseconds: 2_000,
                    reapTimeoutMilliseconds: 5_000
                )
            )
        )
        XCTAssertEqual(effects, .zero)
    }

    private func assertAnchorRejects(
        _ bytes: Data,
        expected: GitRuntimePolicyDenialError,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let effects = FakeOperationalEffects()
        XCTAssertThrowsError(
            try {
                let policyFile = try capturePolicy(bytes)
                _ = try GitRuntimePolicyDenialVerifier.anchor(
                    policyFile: policyFile,
                    trustAnchor: trustAnchorMatching(policyFile)
                )
            }(),
            file: file,
            line: line
        ) { error in
            XCTAssertEqual(
                error as? GitRuntimePolicyDenialError,
                expected,
                file: file,
                line: line
            )
        }
        XCTAssertEqual(effects, .zero, file: file, line: line)
    }

    private func capturePolicy(_ bytes: Data) throws -> AdmittedFile {
        let url = caseRoot.appendingPathComponent(
            "\(UUID().uuidString).git-runtime-policy"
        )
        try bytes.write(to: url)
        return try AdmittedFile.capture(
            absolutePath: url.path,
            maximumBytes: 16 * 1024
        )
    }

    private func trustAnchorMatching(
        _ file: AdmittedFile,
        expectedSHA256: String? = nil,
        expectedBytes: UInt64? = nil,
        minimumGeneration: UInt64 = 7,
        verificationUnixSeconds: UInt64 = 2_000_000_000
    ) -> GitRuntimePolicyDenialTrustAnchor {
        GitRuntimePolicyDenialTrustAnchor(
            expectedCurrentPolicySHA256:
                expectedSHA256 ?? file.sha256,
            expectedCurrentPolicyBytes:
                expectedBytes ?? UInt64(file.bytes.count),
            minimumPolicyGeneration: minimumGeneration,
            verificationUnixSeconds: verificationUnixSeconds
        )
    }

    private func replacing(_ source: String, with replacement: String) -> Data {
        let text = String(
            decoding: Self.expectedPolicyBytes,
            as: UTF8.self
        )
        return Data(
            text.replacingOccurrences(
                of: source,
                with: replacement
            )
            .utf8
        )
    }
}

private extension GitRuntimePolicyDenialDocumentTests {
    struct FakeOperationalEffects: Equatable {
        var spawnCount = 0
        var fileSystemMutationCount = 0
        var networkAccessCount = 0
        var packConsumptionCount = 0
        var sourceMutationCount = 0

        static let zero = Self()
    }

    static let zeroSHA256 = String(repeating: "0", count: 64)
    static let selfGuardSHA256 = String(repeating: "a", count: 64)
    static let selfGuardMachOUUID = String(repeating: "b", count: 32)
    static let selfGuardCodeDirectorySHA256 =
        String(repeating: "c", count: 64)
    static let selfGuardRuntimeClosureID =
        String(repeating: "d", count: 64)
    static let gitRuntimeClosureID =
        String(repeating: "e", count: 64)
    static let otherSHA256 = String(repeating: "6", count: 64)
    static let uppercaseSHA256 = String(repeating: "A", count: 64)
    static let expectedPolicySHA256 =
        "32c141ca6811a2b77d9d3510ce2c8c87558e3e647f4c849a2800f311795659d0"
    static let expectedPolicyByteCount: UInt64 = 2_397

    static func fields(
        limits: GitToolPolicyResourceLimits =
            GitToolPolicyVerifier.phase1ResourceCeilings,
        terminationGraceMilliseconds: UInt64 = 2_000,
        reapTimeoutMilliseconds: UInt64 = 5_000
    ) -> GitRuntimePolicyDenialFields {
        GitRuntimePolicyDenialFields(
            policyGeneration: 7,
            validFromUnixSeconds: 1_900_000_000,
            validUntilUnixSeconds: 2_100_000_000,
            selfGuardSHA256: selfGuardSHA256,
            selfGuardBytes: 1_048_576,
            selfGuardMachOUUID: selfGuardMachOUUID,
            selfGuardCodeDirectorySHA256:
                selfGuardCodeDirectorySHA256,
            selfGuardRuntimeClosureID:
                selfGuardRuntimeClosureID,
            gitRuntimeClosureID: gitRuntimeClosureID,
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
        fast-mlx-proof-control-git-runtime-policy-v1
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
        self_guard_runtime_closure_id=\(selfGuardRuntimeClosureID)
        git_runtime_closure_id=\(gitRuntimeClosureID)
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
}
