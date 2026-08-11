import CryptoKit
import Darwin
import Foundation
import XCTest
@testable import ProofControl

final class GitToolPolicyDocumentTests: XCTestCase {
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
            "fast-mlx-git-tool-policy-\(UUID().uuidString)"
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

    func testCanonicalPolicyBytesAnchorOnlyInertExactDocument() throws {
        let expectedBytes = Self.expectedPolicyBytes
        XCTAssertEqual(
            try GitToolPolicyVerifier.policyBytes(fields: Self.fields()),
            expectedBytes
        )

        let policyFile = try capturePolicy(expectedBytes)
        let document = try GitToolPolicyVerifier.anchor(
            policyFile: policyFile,
            trustAnchor: trustAnchor()
        )

        XCTAssertEqual(document.policyFile, policyFile)
        XCTAssertEqual(policyFile.sha256, Self.expectedPolicySHA256)
        XCTAssertEqual(
            UInt64(policyFile.bytes.count),
            Self.expectedPolicyByteCount
        )
        XCTAssertEqual(document.policySHA256, policyFile.sha256)
        XCTAssertEqual(
            document.policyBytes,
            UInt64(expectedBytes.count)
        )
        XCTAssertEqual(document.fields, Self.fields())
        XCTAssertEqual(
            document.runtimePolicySHA256,
            Self.runtimePolicySHA256
        )
        XCTAssertEqual(
            document.executableSHA256,
            Self.executableSHA256
        )
        XCTAssertEqual(
            document.runtimeClosureManifestSHA256,
            Self.runtimeClosureManifestSHA256
        )
        XCTAssertFalse(document.canImportGitObjects)
        XCTAssertFalse(document.canMutateFileSystem)
        XCTAssertFalse(document.canSpawn)
        XCTAssertFalse(document.canBuild)
        XCTAssertFalse(document.canLoadModel)
        XCTAssertFalse(document.canReserveOutput)
        XCTAssertFalse(document.canPublish)
    }

    func testRejectsNonCanonicalStructureProfilesScalarsAndRuntimeDigest()
        throws
    {
        var reordered = String(
            decoding: Self.expectedPolicyBytes,
            as: UTF8.self
        )
        .split(separator: "\n", omittingEmptySubsequences: false)
        reordered.swapAt(2, 3)

        let cases: [Data] = [
            replacing(
                "fast-mlx-proof-control-git-tool-policy-v1",
                with: "foreign-tool-policy-v1"
            ),
            replacing(
                "subject=absorbed-mla-source-import-git",
                with: "subject=other"
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
                "valid_from_unix_seconds=1900000000",
                with: "valid_from_unix_seconds=01900000000"
            ),
            replacing(
                "valid_until_unix_seconds=2100000000",
                with: "valid_until_unix_seconds=1899999999"
            ),
            replacing(
                "claim_subject=absorbed-mla-loaded-result-pair",
                with: "claim_subject=other"
            ),
            replacing(
                "executable_sha256=\(Self.executableSHA256)",
                with:
                    "executable_sha256=" +
                    Self.executableSHA256.uppercased()
            ),
            replacing(
                "executable_bytes=1048576",
                with: "executable_bytes=0"
            ),
            replacing(
                "executable_architecture=arm64",
                with: "executable_architecture=x86_64"
            ),
            replacing(
                "executable_macho_uuid=\(Self.executableMachOUUID)",
                with:
                    "executable_macho_uuid=" +
                    Self.executableMachOUUID.uppercased()
            ),
            replacing(
                "executable_code_directory_sha256=" +
                    Self.executableCodeDirectorySHA256,
                with:
                    "executable_code_directory_sha256=" +
                    Self.executableCodeDirectorySHA256.uppercased()
            ),
            replacing(
                "runtime_closure_manifest_sha256=" +
                    Self.runtimeClosureManifestSHA256,
                with:
                    "runtime_closure_manifest_sha256=" +
                    Self.runtimeClosureManifestSHA256.uppercased()
            ),
            replacing(
                "runtime_policy_sha256=\(Self.runtimePolicySHA256)",
                with:
                    "runtime_policy_sha256=" +
                    Self.runtimePolicySHA256.uppercased()
            ),
            replacing(
                "runtime_closure_manifest_bytes=4096",
                with: "runtime_closure_manifest_bytes=0"
            ),
            replacing(
                "argv_profile=git-source-pack-ingress-v1",
                with: "argv_profile=other"
            ),
            replacing(
                "environment_profile=git-source-pack-environment-v1",
                with: "environment_profile=inherit-v1"
            ),
            replacing(
                "fd_profile=self-guard-fchdir-then-stdio-v1",
                with: "fd_profile=inherit-all-v1"
            ),
            replacing(
                "sandbox_profile=git-source-pack-sandbox-v1",
                with: "sandbox_profile=none-v1"
            ),
            replacing(
                "cwd_profile=descriptor-anchored-git-directory-v1",
                with: "cwd_profile=ambient-repository-v1"
            ),
            replacing("umask=0077", with: "umask=077"),
            replacing("cloexec_default=true", with: "cloexec_default=false"),
            replacing(
                "network_policy=deny-all-v1",
                with: "network_policy=allow-v1"
            ),
            replacing(
                "descendant_process_policy=deny-all-v1",
                with: "descendant_process_policy=allow-v1"
            ),
            replacing(
                "config_policy=none-v1",
                with: "config_policy=ambient-v1"
            ),
            replacing(
                "hook_policy=deny-v1",
                with: "hook_policy=allow-v1"
            ),
            replacing(
                "attributes_policy=raw-only-v1",
                with: "attributes_policy=worktree-v1"
            ),
            replacing(
                "filter_policy=deny-v1",
                with: "filter_policy=allow-v1"
            ),
            replacing(
                "textconv_policy=deny-v1",
                with: "textconv_policy=allow-v1"
            ),
            replacing(
                "replace_refs_policy=deny-v1",
                with: "replace_refs_policy=allow-v1"
            ),
            replacing(
                "alternates_policy=deny-v1",
                with: "alternates_policy=allow-v1"
            ),
            replacing(
                "promisor_lazy_fetch_policy=deny-v1",
                with: "promisor_lazy_fetch_policy=allow-v1"
            ),
            replacing(
                "transport_policy=deny-v1",
                with: "transport_policy=allow-v1"
            ),
            replacing(
                "pack_version=2",
                with: "pack_version=3"
            ),
            replacing(
                "object_format=sha1",
                with: "object_format=sha256"
            ),
            replacing(
                "delta_depth=0",
                with: "delta_depth=1"
            ),
            replacing(
                "max_invocations_per_role=4",
                with: "max_invocations_per_role=5"
            ),
            Self.expectedPolicyBytes + Data("unknown=true\n".utf8),
            Data(Self.expectedPolicyBytes.dropLast()),
            replacing(
                "policy_generation=7\n",
                with: ""
            ),
            replacing(
                "policy_generation=7\n",
                with: "policy_generation=7\npolicy_generation=7\n"
            ),
            Data(
                reordered.map(String.init).joined(separator: "\n").utf8
            ),
            Data(
                String(
                    decoding: Self.expectedPolicyBytes,
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
                try GitToolPolicyVerifier.anchor(
                    policyFile: file,
                    trustAnchor: trustAnchorMatching(file)
                )
            ) { error in
                XCTAssertEqual(
                    error as? GitToolPolicyError,
                    .nonCanonicalPolicy
                )
            }
        }
    }

    func testRejectsExternalAnchorDigestByteGenerationAndTimeMismatch()
        throws
    {
        let policyFile = try capturePolicy(Self.expectedPolicyBytes)
        let contexts: [
            (GitToolPolicyTrustAnchor, GitToolPolicyError)
        ] = [
            (
                trustAnchor(
                    expectedSHA256: Self.otherSHA256
                ),
                .policyDigestMismatch
            ),
            (
                trustAnchor(
                    expectedBytes: UInt64(policyFile.bytes.count + 1)
                ),
                .policyByteCountMismatch(
                    expected: UInt64(policyFile.bytes.count + 1),
                    actual: UInt64(policyFile.bytes.count)
                )
            ),
            (
                trustAnchor(
                    minimumGeneration: 8
                ),
                .policyGenerationRollback(minimum: 8, actual: 7)
            ),
            (
                trustAnchor(
                    verificationUnixSeconds: 1_899_999_999
                ),
                .policyNotYetValid
            ),
            (
                trustAnchor(
                    verificationUnixSeconds: 2_100_000_001
                ),
                .policyExpired
            ),
        ]

        for (anchor, expectedError) in contexts {
            XCTAssertThrowsError(
                try GitToolPolicyVerifier.anchor(
                    policyFile: policyFile,
                    trustAnchor: anchor
                )
            ) { error in
                XCTAssertEqual(
                    error as? GitToolPolicyError,
                    expectedError
                )
            }
        }

        for boundary in [UInt64(1_900_000_000), 2_100_000_000] {
            XCTAssertNoThrow(
                try GitToolPolicyVerifier.anchor(
                    policyFile: policyFile,
                    trustAnchor: trustAnchor(
                        verificationUnixSeconds: boundary
                    )
                )
            )
        }

        let invalidDigest = trustAnchor(
            expectedSHA256: policyFile.sha256.uppercased()
        )
        XCTAssertThrowsError(
            try GitToolPolicyVerifier.anchor(
                policyFile: policyFile,
                trustAnchor: invalidDigest
            )
        ) { error in
            XCTAssertEqual(
                error as? GitToolPolicyError,
                .invalidTrustAnchor(.expectedCurrentPolicySHA256)
            )
        }

        let invalidBytes = trustAnchor(
            expectedBytes: 0
        )
        XCTAssertThrowsError(
            try GitToolPolicyVerifier.anchor(
                policyFile: policyFile,
                trustAnchor: invalidBytes
            )
        ) { error in
            XCTAssertEqual(
                error as? GitToolPolicyError,
                .invalidTrustAnchor(.expectedCurrentPolicyBytes)
            )
        }

        let invalidGeneration = trustAnchor(minimumGeneration: 0)
        XCTAssertThrowsError(
            try GitToolPolicyVerifier.anchor(
                policyFile: policyFile,
                trustAnchor: invalidGeneration
            )
        ) { error in
            XCTAssertEqual(
                error as? GitToolPolicyError,
                .invalidTrustAnchor(.minimumPolicyGeneration)
            )
        }
    }

    func testCompileTimeCeilingsRejectWideningAndAllowNarrowing() throws {
        let widenedCases: [
            (
                field: GitToolPolicyResourceField,
                policyKey: String,
                maximum: UInt64,
                limits: GitToolPolicyResourceLimits
            )
        ] = [
            (.maxPackBytes, "max_pack_bytes", 134_217_728, Self.limits(
                maxPackBytes: 134_217_729
            )),
            (.maxPackObjects, "max_pack_objects", 20_000, Self.limits(
                maxPackObjects: 20_001
            )),
            (.maxCommitBytes, "max_commit_bytes", 1_048_576, Self.limits(
                maxCommitBytes: 1_048_577
            )),
            (
                .maxSingleInflatedObjectBytes,
                "max_single_inflated_object_bytes",
                16_777_216,
                Self.limits(maxSingleInflatedObjectBytes: 16_777_217)
            ),
            (
                .maxTotalInflatedBytes,
                "max_total_inflated_bytes",
                268_435_456,
                Self.limits(maxTotalInflatedBytes: 268_435_457)
            ),
            (.maxCompressionRatio, "max_compression_ratio", 64, Self.limits(
                maxCompressionRatio: 65
            )),
            (.maxTreeDepth, "max_tree_depth", 32, Self.limits(
                maxTreeDepth: 33
            )),
            (.maxTreeCount, "max_tree_count", 10_000, Self.limits(
                maxTreeCount: 10_001
            )),
            (
                .maxObjectDatabaseBytes,
                "max_odb_bytes",
                167_772_160,
                Self.limits(maxObjectDatabaseBytes: 167_772_161)
            ),
            (.maxStdoutBytes, "max_stdout_bytes", 67_108_864, Self.limits(
                maxStdoutBytes: 67_108_865
            )),
            (.maxStderrBytes, "max_stderr_bytes", 1_048_576, Self.limits(
                maxStderrBytes: 1_048_577
            )),
            (
                .wallTimeoutMilliseconds,
                "wall_timeout_milliseconds",
                120_000,
                Self.limits(wallTimeoutMilliseconds: 120_001)
            ),
            (.cpuTimeoutSeconds, "cpu_timeout_seconds", 120, Self.limits(
                cpuTimeoutSeconds: 121
            )),
            (
                .maxAddressSpaceBytes,
                "max_address_space_bytes",
                1_073_741_824,
                Self.limits(maxAddressSpaceBytes: 1_073_741_825)
            ),
            (.maxFileSizeBytes, "max_file_size_bytes", 167_772_160, Self.limits(
                maxFileSizeBytes: 167_772_161
            )),
            (.maxOpenFiles, "max_open_files", 64, Self.limits(
                maxOpenFiles: 65
            )),
        ]

        for context in widenedCases {
            let expectedError = GitToolPolicyError
                .resourceLimitExceedsCompileTimeCeiling(
                    field: context.field,
                    maximum: context.maximum,
                    actual: context.maximum + 1
                )
            XCTAssertThrowsError(
                try GitToolPolicyVerifier.policyBytes(
                    fields: Self.fields(limits: context.limits)
                )
            ) { error in
                XCTAssertEqual(error as? GitToolPolicyError, expectedError)
            }

            let widenedPolicyBytes = replacing(
                "\(context.policyKey)=\(context.maximum)",
                with: "\(context.policyKey)=\(context.maximum + 1)"
            )
            let widenedPolicyFile = try capturePolicy(widenedPolicyBytes)
            XCTAssertThrowsError(
                try GitToolPolicyVerifier.anchor(
                    policyFile: widenedPolicyFile,
                    trustAnchor: trustAnchorMatching(widenedPolicyFile)
                )
            ) { error in
                XCTAssertEqual(error as? GitToolPolicyError, expectedError)
            }
        }

        let narrowedFields = Self.fields(
            limits: Self.limits(
                maxPackBytes: 67_108_864,
                maxPackObjects: 10_000,
                maxTreeDepth: 16
            )
        )
        let narrowedBytes = try GitToolPolicyVerifier.policyBytes(
            fields: narrowedFields
        )
        let narrowedFile = try capturePolicy(narrowedBytes)
        let document = try GitToolPolicyVerifier.anchor(
            policyFile: narrowedFile,
            trustAnchor: trustAnchorMatching(narrowedFile)
        )
        XCTAssertEqual(document.fields.limits, narrowedFields.limits)

        let zero = Self.fields(
            limits: Self.limits(maxPackObjects: 0)
        )
        XCTAssertThrowsError(
            try GitToolPolicyVerifier.policyBytes(fields: zero)
        ) { error in
            XCTAssertEqual(
                error as? GitToolPolicyError,
                .nonCanonicalPolicy
            )
        }
    }

    func testAnchoredDocumentRetainsCapturedBytesAfterBackingRewrite()
        throws
    {
        let url = caseRoot.appendingPathComponent("git-tool.policy")
        try Self.expectedPolicyBytes.write(to: url)
        let policyFile = try AdmittedFile.capture(
            absolutePath: url.path,
            maximumBytes: 16 * 1024
        )
        try Data("replaced\n".utf8).write(to: url)

        let document = try GitToolPolicyVerifier.anchor(
            policyFile: policyFile,
            trustAnchor: trustAnchor()
        )
        XCTAssertEqual(document.policyFile.bytes, Self.expectedPolicyBytes)
        XCTAssertEqual(document.policySHA256, Self.sha256Hex(
            Self.expectedPolicyBytes
        ))
    }

    private func capturePolicy(_ bytes: Data) throws -> AdmittedFile {
        let url = caseRoot.appendingPathComponent(
            "\(UUID().uuidString).git-tool-policy"
        )
        try bytes.write(to: url)
        return try AdmittedFile.capture(
            absolutePath: url.path,
            maximumBytes: 16 * 1024
        )
    }

    private func trustAnchor(
        expectedSHA256: String =
            GitToolPolicyDocumentTests.expectedPolicySHA256,
        expectedBytes: UInt64 =
            GitToolPolicyDocumentTests.expectedPolicyByteCount,
        minimumGeneration: UInt64 = 7,
        verificationUnixSeconds: UInt64 = 2_000_000_000
    ) -> GitToolPolicyTrustAnchor {
        GitToolPolicyTrustAnchor(
            expectedCurrentPolicySHA256: expectedSHA256,
            expectedCurrentPolicyBytes: expectedBytes,
            minimumPolicyGeneration: minimumGeneration,
            verificationUnixSeconds: verificationUnixSeconds
        )
    }

    private func trustAnchorMatching(
        _ file: AdmittedFile,
        minimumGeneration: UInt64 = 7,
        verificationUnixSeconds: UInt64 = 2_000_000_000
    ) -> GitToolPolicyTrustAnchor {
        GitToolPolicyTrustAnchor(
            expectedCurrentPolicySHA256: file.sha256,
            expectedCurrentPolicyBytes: UInt64(file.bytes.count),
            minimumPolicyGeneration: minimumGeneration,
            verificationUnixSeconds: verificationUnixSeconds
        )
    }

    private func replacing(_ source: String, with replacement: String) -> Data {
        let text = String(decoding: Self.expectedPolicyBytes, as: UTF8.self)
        return Data(
            text.replacingOccurrences(of: source, with: replacement).utf8
        )
    }
}

private extension GitToolPolicyDocumentTests {
    static let executableSHA256 = String(repeating: "ab", count: 32)
    static let executableMachOUUID = String(repeating: "cd", count: 16)
    static let executableCodeDirectorySHA256 =
        String(repeating: "de", count: 32)
    static let runtimeClosureManifestSHA256 =
        String(repeating: "bc", count: 32)
    static let runtimePolicySHA256 = String(repeating: "ef", count: 32)
    static let otherSHA256 = String(repeating: "66", count: 32)
    static let expectedPolicySHA256 =
        "449f02ce94ac5c3a913adb9aaf38dc2461ff1632c60bfc386dd9ab1370126d45"
    static let expectedPolicyByteCount: UInt64 = 1_788

    static func fields(
        limits: GitToolPolicyResourceLimits = limits()
    ) -> GitToolPolicyFields {
        GitToolPolicyFields(
            policyGeneration: 7,
            validFromUnixSeconds: 1_900_000_000,
            validUntilUnixSeconds: 2_100_000_000,
            executableSHA256: executableSHA256,
            executableBytes: 1_048_576,
            executableMachOUUID: executableMachOUUID,
            executableCodeDirectorySHA256:
                executableCodeDirectorySHA256,
            runtimeClosureManifestSHA256:
                runtimeClosureManifestSHA256,
            runtimeClosureManifestBytes: 4_096,
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
        fast-mlx-proof-control-git-tool-policy-v1
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
        runtime_closure_manifest_sha256=\(runtimeClosureManifestSHA256)
        runtime_closure_manifest_bytes=4096
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

    static func sha256Hex(_ bytes: Data) -> String {
        SHA256.hash(data: bytes)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}
