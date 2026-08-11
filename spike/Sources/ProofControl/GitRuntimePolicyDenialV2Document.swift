import Foundation

struct GitRuntimePolicyDenialV2Fields: Equatable, Sendable {
    let policyGeneration: UInt64
    let validFromUnixSeconds: UInt64
    let validUntilUnixSeconds: UInt64
    let selfGuardSHA256: String
    let selfGuardBytes: UInt64
    let selfGuardMachOUUID: String
    let selfGuardCodeDirectorySHA256: String
    let selfGuardFileImageRuntimeClosureContentEvidenceID: String
    let gitFileImageRuntimeClosureContentEvidenceID: String
    let cpuTimeoutSeconds: UInt64
    let maxAddressSpaceBytes: UInt64
    let maxFileSizeBytes: UInt64
    let maxOpenFiles: UInt64
    let maxObjectDatabaseBytes: UInt64
    let maxStdoutBytes: UInt64
    let maxStderrBytes: UInt64
    let wallTimeoutMilliseconds: UInt64
    let terminationGraceMilliseconds: UInt64
    let reapTimeoutMilliseconds: UInt64

    init(
        policyGeneration: UInt64,
        validFromUnixSeconds: UInt64,
        validUntilUnixSeconds: UInt64,
        selfGuardSHA256: String,
        selfGuardBytes: UInt64,
        selfGuardMachOUUID: String,
        selfGuardCodeDirectorySHA256: String,
        selfGuardFileImageRuntimeClosureContentEvidenceID: String,
        gitFileImageRuntimeClosureContentEvidenceID: String,
        limits: GitToolPolicyResourceLimits,
        terminationGraceMilliseconds: UInt64,
        reapTimeoutMilliseconds: UInt64
    ) {
        self.policyGeneration = policyGeneration
        self.validFromUnixSeconds = validFromUnixSeconds
        self.validUntilUnixSeconds = validUntilUnixSeconds
        self.selfGuardSHA256 = selfGuardSHA256
        self.selfGuardBytes = selfGuardBytes
        self.selfGuardMachOUUID = selfGuardMachOUUID
        self.selfGuardCodeDirectorySHA256 =
            selfGuardCodeDirectorySHA256
        self.selfGuardFileImageRuntimeClosureContentEvidenceID =
            selfGuardFileImageRuntimeClosureContentEvidenceID
        self.gitFileImageRuntimeClosureContentEvidenceID =
            gitFileImageRuntimeClosureContentEvidenceID
        self.cpuTimeoutSeconds = limits.cpuTimeoutSeconds
        self.maxAddressSpaceBytes = limits.maxAddressSpaceBytes
        self.maxFileSizeBytes = limits.maxFileSizeBytes
        self.maxOpenFiles = limits.maxOpenFiles
        self.maxObjectDatabaseBytes = limits.maxObjectDatabaseBytes
        self.maxStdoutBytes = limits.maxStdoutBytes
        self.maxStderrBytes = limits.maxStderrBytes
        self.wallTimeoutMilliseconds = limits.wallTimeoutMilliseconds
        self.terminationGraceMilliseconds =
            terminationGraceMilliseconds
        self.reapTimeoutMilliseconds = reapTimeoutMilliseconds
    }
}

/// External comparison context only; this value grants no authority.
struct GitRuntimePolicyDenialV2TrustAnchor: Equatable, Sendable {
    let expectedCurrentPolicySHA256: String
    let expectedCurrentPolicyBytes: UInt64
    let minimumPolicyGeneration: UInt64
    let verificationUnixSeconds: UInt64
}

enum GitRuntimePolicyDenialV2TrustAnchorField:
    String,
    Equatable,
    Sendable
{
    case expectedCurrentPolicySHA256 =
        "expected-current-policy-sha256"
    case expectedCurrentPolicyBytes =
        "expected-current-policy-bytes"
    case minimumPolicyGeneration =
        "minimum-policy-generation"
}

enum GitRuntimePolicyDenialV2ResourceField:
    String,
    Equatable,
    Sendable
{
    case cpuTimeoutSeconds = "cpu-timeout-seconds"
    case maxAddressSpaceBytes = "max-address-space-bytes"
    case maxFileSizeBytes = "max-file-size-bytes"
    case maxOpenFiles = "max-open-files"
    case maxObjectDatabaseBytes = "max-object-database-bytes"
    case maxStdoutBytes = "max-stdout-bytes"
    case maxStderrBytes = "max-stderr-bytes"
    case wallTimeoutMilliseconds = "wall-timeout-milliseconds"
    case terminationGraceMilliseconds =
        "termination-grace-milliseconds"
    case reapTimeoutMilliseconds = "reap-timeout-milliseconds"
}

enum GitRuntimePolicyV2ReferenceField:
    String,
    Equatable,
    Sendable
{
    case toolManifestDigest = "tool-manifest-digest"
    case toolManifestByteCount = "tool-manifest-byte-count"
    case runtimePolicyDigest = "runtime-policy-digest"
}

enum GitRuntimePolicyDenialV2Error: Error, Equatable, Sendable {
    case policySize
    case nonCanonicalPolicy
    case invalidTrustAnchor(
        GitRuntimePolicyDenialV2TrustAnchorField
    )
    case policyDigestMismatch
    case policyByteCountMismatch(expected: UInt64, actual: UInt64)
    case policyGenerationRollback(minimum: UInt64, actual: UInt64)
    case policyNotYetValid
    case policyExpired
    case resourceLimitExceedsCompileTimeCeiling(
        field: GitRuntimePolicyDenialV2ResourceField,
        maximum: UInt64,
        actual: UInt64
    )
    case cleanupDeadlineOverflow
    case invalidWatchdogInterval
    case toolPolicyReferenceDiscontinuity(
        GitRuntimePolicyV2ReferenceField
    )
    case runtimePolicyDigestMismatch
    case toolPolicyResourceMismatch(
        GitRuntimePolicyDenialV2ResourceField
    )
    case supportedSandboxMechanismUnavailable
}

/// Exact current v2 denial bytes anchored for comparison only.
struct AnchoredGitRuntimePolicyDenialV2Document: Equatable, Sendable {
    let policyFile: AdmittedFile
    let policySHA256: String
    let policyBytes: UInt64
    let policyGeneration: UInt64
    let validFromUnixSeconds: UInt64
    let validUntilUnixSeconds: UInt64
    let selfGuardSHA256: String
    let selfGuardFileImageRuntimeClosureContentEvidenceID: String
    let gitFileImageRuntimeClosureContentEvidenceID: String
    let fields: GitRuntimePolicyDenialV2Fields

    let canExecute = false
    let canSpawn = false
    let canAccessNetwork = false
    let canConsumePack = false
    let canMutateFileSystem = false
    let canImportGitObjects = false
    let canBuild = false
    let canLoadModel = false
    let canReserveOutput = false
    let canPublish = false

    fileprivate init(
        policyFile: AdmittedFile,
        fields: GitRuntimePolicyDenialV2Fields
    ) {
        self.policyFile = policyFile
        self.policySHA256 = policyFile.sha256
        self.policyBytes = UInt64(policyFile.bytes.count)
        self.policyGeneration = fields.policyGeneration
        self.validFromUnixSeconds = fields.validFromUnixSeconds
        self.validUntilUnixSeconds = fields.validUntilUnixSeconds
        self.selfGuardSHA256 = fields.selfGuardSHA256
        self.selfGuardFileImageRuntimeClosureContentEvidenceID =
            fields.selfGuardFileImageRuntimeClosureContentEvidenceID
        self.gitFileImageRuntimeClosureContentEvidenceID =
            fields.gitFileImageRuntimeClosureContentEvidenceID
        self.fields = fields
    }
}

enum GitRuntimePolicyDenialV2Verifier {
    static let policyDomain =
        "fast-mlx-proof-control-git-runtime-policy-v2"
    static let policySubject =
        "absorbed-mla-source-import-git-runtime"
    static let maximumPolicyBytes = 2_643
    static let maximumLineBytes = 122
    static let terminationGraceMaximumMilliseconds: UInt64 = 2_000
    static let reapTimeoutMaximumMilliseconds: UInt64 = 5_000
    static let maximumCleanupDeadlineMilliseconds: UInt64 = 127_000

    static func policyBytes(
        fields: GitRuntimePolicyDenialV2Fields
    ) throws -> Data {
        try validate(fields)
        let lines = [
            policyDomain,
            "subject=\(policySubject)",
            "policy_generation=\(fields.policyGeneration)",
            "valid_from_unix_seconds=\(fields.validFromUnixSeconds)",
            "valid_until_unix_seconds=\(fields.validUntilUnixSeconds)",
            "platform_architecture=arm64",
            "platform_hardware_model=Mac15,14",
            "platform_os_version=26.5.2",
            "platform_os_build=25F84",
            "platform_xcode_version=26.6",
            "platform_xcode_build=17F113",
            "platform_sdk_version=26.5",
            "self_guard_sha256=\(fields.selfGuardSHA256)",
            "self_guard_bytes=\(fields.selfGuardBytes)",
            "self_guard_macho_uuid=\(fields.selfGuardMachOUUID)",
            "self_guard_code_directory_sha256=\(fields.selfGuardCodeDirectorySHA256)",
            "self_guard_file_image_runtime_closure_content_evidence_id=\(fields.selfGuardFileImageRuntimeClosureContentEvidenceID)",
            "git_file_image_runtime_closure_content_evidence_id=\(fields.gitFileImageRuntimeClosureContentEvidenceID)",
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
            "sandbox_policy_sha256=\(String(repeating: "0", count: 64))",
            "sandbox_policy_bytes=0",
            "network_enforcement=unavailable",
            "filesystem_enforcement=unavailable",
            "process_exec_enforcement=unavailable",
            "descendant_process_enforcement=unavailable",
            "resource_enforcement=partial-public-controls-not-complete-v1",
            "resource_profile=self-guard-rlimit-plus-parent-watchdog-v1",
            "rlimit_core_bytes=0",
            "cpu_timeout_seconds=\(fields.cpuTimeoutSeconds)",
            "max_address_space_bytes=\(fields.maxAddressSpaceBytes)",
            "max_file_size_bytes=\(fields.maxFileSizeBytes)",
            "max_open_files=\(fields.maxOpenFiles)",
            "max_odb_bytes=\(fields.maxObjectDatabaseBytes)",
            "max_stdout_bytes=\(fields.maxStdoutBytes)",
            "max_stderr_bytes=\(fields.maxStderrBytes)",
            "wall_timeout_milliseconds=\(fields.wallTimeoutMilliseconds)",
            "termination_grace_milliseconds=\(fields.terminationGraceMilliseconds)",
            "reap_timeout_milliseconds=\(fields.reapTimeoutMilliseconds)",
            "watchdog_profile=monotonic-killpg-waitpid-v1",
            "compatibility_profile=macos-26-5-2-xcode-26-6-public-api-v1",
            "compatibility_outcome=no-go",
            "runtime_decision=no-go",
        ]
        precondition(lines.count == 57)
        let bytes = Data((lines.joined(separator: "\n") + "\n").utf8)
        guard bytes.count <= maximumPolicyBytes else {
            throw GitRuntimePolicyDenialV2Error.policySize
        }
        return bytes
    }

    static func anchor(
        policyFile: AdmittedFile,
        trustAnchor: GitRuntimePolicyDenialV2TrustAnchor
    ) throws -> AnchoredGitRuntimePolicyDenialV2Document {
        try validate(trustAnchor)
        guard policyFile.bytes.count <= maximumPolicyBytes else {
            throw GitRuntimePolicyDenialV2Error.policySize
        }
        guard policyFile.sha256 ==
            trustAnchor.expectedCurrentPolicySHA256
        else {
            throw GitRuntimePolicyDenialV2Error.policyDigestMismatch
        }
        let actualBytes = UInt64(policyFile.bytes.count)
        guard actualBytes == trustAnchor.expectedCurrentPolicyBytes else {
            throw GitRuntimePolicyDenialV2Error.policyByteCountMismatch(
                expected: trustAnchor.expectedCurrentPolicyBytes,
                actual: actualBytes
            )
        }

        let fields = try parsePolicy(policyFile.bytes)
        guard fields.policyGeneration >=
            trustAnchor.minimumPolicyGeneration
        else {
            throw GitRuntimePolicyDenialV2Error
                .policyGenerationRollback(
                    minimum: trustAnchor.minimumPolicyGeneration,
                    actual: fields.policyGeneration
                )
        }
        guard trustAnchor.verificationUnixSeconds >=
            fields.validFromUnixSeconds
        else {
            throw GitRuntimePolicyDenialV2Error.policyNotYetValid
        }
        guard trustAnchor.verificationUnixSeconds <=
            fields.validUntilUnixSeconds
        else {
            throw GitRuntimePolicyDenialV2Error.policyExpired
        }
        return AnchoredGitRuntimePolicyDenialV2Document(
            policyFile: policyFile,
            fields: fields
        )
    }

    static func requireUnavailable(
        policyDocument: AnchoredGitRuntimePolicyDenialV2Document,
        toolPolicyReference: SignedClaimGitToolPolicyV2Reference
    ) throws -> Never {
        try validate(toolPolicyReference)
        guard
            policyDocument.policySHA256 ==
                toolPolicyReference.policyDocument.runtimePolicySHA256,
            policyDocument.policySHA256 ==
                toolPolicyReference.signedClaim.fields.policies
                .runtimeSHA256
        else {
            throw GitRuntimePolicyDenialV2Error
                .runtimePolicyDigestMismatch
        }
        try validate(
            policyDocument.fields,
            matches: toolPolicyReference.policyDocument.fields.limits
        )
        throw GitRuntimePolicyDenialV2Error
            .supportedSandboxMechanismUnavailable
    }
}

private extension GitRuntimePolicyDenialV2Verifier {
    struct CheckedByteLineCursor {
        let bytes: Data
        var offset: Data.Index
        var lineCount = 0

        init(bytes: Data) {
            self.bytes = bytes
            self.offset = bytes.startIndex
        }

        var isAtEnd: Bool {
            offset == bytes.endIndex
        }

        mutating func require(_ expected: String) throws {
            guard try nextLine() == expected else {
                throw GitRuntimePolicyDenialV2Error.nonCanonicalPolicy
            }
        }

        mutating func value(prefix: String) throws -> String {
            let line = try nextLine()
            guard line.hasPrefix(prefix) else {
                throw GitRuntimePolicyDenialV2Error.nonCanonicalPolicy
            }
            return String(line.dropFirst(prefix.utf8.count))
        }

        mutating func canonicalUInt64(
            prefix: String
        ) throws -> UInt64 {
            let text = try value(prefix: prefix)
            guard
                isCanonicalDecimal(text),
                let value = UInt64(text)
            else {
                throw GitRuntimePolicyDenialV2Error.nonCanonicalPolicy
            }
            return value
        }

        mutating func lowercaseHex(
            prefix: String,
            count: Int
        ) throws -> String {
            let text = try value(prefix: prefix)
            guard isLowercaseHex(text, count: count) else {
                throw GitRuntimePolicyDenialV2Error.nonCanonicalPolicy
            }
            return text
        }

        mutating func nextLine() throws -> String {
            guard offset < bytes.endIndex else {
                throw GitRuntimePolicyDenialV2Error.nonCanonicalPolicy
            }
            let start = offset
            var cursor = offset
            while cursor < bytes.endIndex {
                let byte = bytes[cursor]
                if byte == 0x0a {
                    let count = cursor - start
                    guard count > 0, count <= maximumLineBytes else {
                        throw GitRuntimePolicyDenialV2Error
                            .nonCanonicalPolicy
                    }
                    let lineBytes = bytes[start..<cursor]
                    guard lineBytes.allSatisfy({
                        (0x20...0x7e).contains($0)
                    }) else {
                        throw GitRuntimePolicyDenialV2Error
                            .nonCanonicalPolicy
                    }
                    offset = bytes.index(after: cursor)
                    lineCount += 1
                    return String(decoding: lineBytes, as: UTF8.self)
                }
                guard cursor - start < maximumLineBytes else {
                    throw GitRuntimePolicyDenialV2Error.nonCanonicalPolicy
                }
                cursor = bytes.index(after: cursor)
            }
            throw GitRuntimePolicyDenialV2Error.nonCanonicalPolicy
        }
    }

    static func parsePolicy(_ bytes: Data) throws
        -> GitRuntimePolicyDenialV2Fields
    {
        var parser = CheckedByteLineCursor(bytes: bytes)
        try parser.require(policyDomain)
        try parser.require("subject=\(policySubject)")
        let policyGeneration = try parser.canonicalUInt64(
            prefix: "policy_generation="
        )
        let validFromUnixSeconds = try parser.canonicalUInt64(
            prefix: "valid_from_unix_seconds="
        )
        let validUntilUnixSeconds = try parser.canonicalUInt64(
            prefix: "valid_until_unix_seconds="
        )
        try parser.require("platform_architecture=arm64")
        try parser.require("platform_hardware_model=Mac15,14")
        try parser.require("platform_os_version=26.5.2")
        try parser.require("platform_os_build=25F84")
        try parser.require("platform_xcode_version=26.6")
        try parser.require("platform_xcode_build=17F113")
        try parser.require("platform_sdk_version=26.5")
        let selfGuardSHA256 = try parser.lowercaseHex(
            prefix: "self_guard_sha256=",
            count: 64
        )
        let selfGuardBytes = try parser.canonicalUInt64(
            prefix: "self_guard_bytes="
        )
        let selfGuardMachOUUID = try parser.lowercaseHex(
            prefix: "self_guard_macho_uuid=",
            count: 32
        )
        let selfGuardCodeDirectorySHA256 = try parser.lowercaseHex(
            prefix: "self_guard_code_directory_sha256=",
            count: 64
        )
        let selfGuardFileImageRuntimeClosureContentEvidenceID =
            try parser.lowercaseHex(
                prefix:
                    "self_guard_file_image_runtime_closure_content_evidence_id=",
                count: 64
            )
        let gitFileImageRuntimeClosureContentEvidenceID =
            try parser.lowercaseHex(
                prefix:
                    "git_file_image_runtime_closure_content_evidence_id=",
                count: 64
            )
        try parser.require("spawn_profile=posix-spawn-self-guard-v1")
        try parser.require(
            "spawn_flags_profile=" +
                "cloexec-default-new-pgroup-clean-signals-v1"
        )
        try parser.require(
            "tool_fd_profile=self-guard-fchdir-then-stdio-v1"
        )
        try parser.require(
            "tool_cwd_profile=descriptor-anchored-git-directory-v1"
        )
        try parser.require(
            "tool_sandbox_profile=git-source-pack-sandbox-v1"
        )
        try parser.require(
            "spawn_setup_fd_profile=stdio-plus-directory-fd3-v1"
        )
        try parser.require("git_fd_profile=stdio-only-v1")
        try parser.require(
            "self_guard_profile=fchdir-revalidate-close-fd3-v1"
        )
        try parser.require("umask=0077")
        try parser.require(
            "environment_profile=git-source-pack-environment-v1"
        )
        try parser.require("argv_profile=git-source-pack-ingress-v1")
        try parser.require("required_network_policy=deny-all-v1")
        try parser.require(
            "required_filesystem_policy=role-and-phase-exact-v1"
        )
        try parser.require(
            "required_process_exec_policy=exact-admitted-git-only-v1"
        )
        try parser.require(
            "required_descendant_process_policy=deny-all-v1"
        )
        try parser.require(
            "sandbox_mechanism=" +
                "none-no-supported-public-in-process-profile-v1"
        )
        try parser.require(
            "sandbox_policy_sha256=" +
                String(repeating: "0", count: 64)
        )
        try parser.require("sandbox_policy_bytes=0")
        try parser.require("network_enforcement=unavailable")
        try parser.require("filesystem_enforcement=unavailable")
        try parser.require("process_exec_enforcement=unavailable")
        try parser.require(
            "descendant_process_enforcement=unavailable"
        )
        try parser.require(
            "resource_enforcement=" +
                "partial-public-controls-not-complete-v1"
        )
        try parser.require(
            "resource_profile=" +
                "self-guard-rlimit-plus-parent-watchdog-v1"
        )
        try parser.require("rlimit_core_bytes=0")
        let cpuTimeoutSeconds = try parser.canonicalUInt64(
            prefix: "cpu_timeout_seconds="
        )
        let maxAddressSpaceBytes = try parser.canonicalUInt64(
            prefix: "max_address_space_bytes="
        )
        let maxFileSizeBytes = try parser.canonicalUInt64(
            prefix: "max_file_size_bytes="
        )
        let maxOpenFiles = try parser.canonicalUInt64(
            prefix: "max_open_files="
        )
        let maxObjectDatabaseBytes = try parser.canonicalUInt64(
            prefix: "max_odb_bytes="
        )
        let maxStdoutBytes = try parser.canonicalUInt64(
            prefix: "max_stdout_bytes="
        )
        let maxStderrBytes = try parser.canonicalUInt64(
            prefix: "max_stderr_bytes="
        )
        let wallTimeoutMilliseconds = try parser.canonicalUInt64(
            prefix: "wall_timeout_milliseconds="
        )
        let terminationGraceMilliseconds = try parser.canonicalUInt64(
            prefix: "termination_grace_milliseconds="
        )
        let reapTimeoutMilliseconds = try parser.canonicalUInt64(
            prefix: "reap_timeout_milliseconds="
        )
        try parser.require(
            "watchdog_profile=monotonic-killpg-waitpid-v1"
        )
        try parser.require(
            "compatibility_profile=" +
                "macos-26-5-2-xcode-26-6-public-api-v1"
        )
        try parser.require("compatibility_outcome=no-go")
        try parser.require("runtime_decision=no-go")
        guard parser.isAtEnd, parser.lineCount == 57 else {
            throw GitRuntimePolicyDenialV2Error.nonCanonicalPolicy
        }

        let base = GitToolPolicyVerifier.phase1ResourceCeilings
        let limits = GitToolPolicyResourceLimits(
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
        let fields = GitRuntimePolicyDenialV2Fields(
            policyGeneration: policyGeneration,
            validFromUnixSeconds: validFromUnixSeconds,
            validUntilUnixSeconds: validUntilUnixSeconds,
            selfGuardSHA256: selfGuardSHA256,
            selfGuardBytes: selfGuardBytes,
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
        try validate(fields)
        guard try policyBytes(fields: fields) == bytes else {
            throw GitRuntimePolicyDenialV2Error.nonCanonicalPolicy
        }
        return fields
    }

    static func validate(
        _ fields: GitRuntimePolicyDenialV2Fields
    ) throws {
        guard
            fields.policyGeneration > 0,
            fields.validFromUnixSeconds <= fields.validUntilUnixSeconds,
            isLowercaseHex(fields.selfGuardSHA256, count: 64),
            fields.selfGuardBytes > 0,
            isLowercaseHex(fields.selfGuardMachOUUID, count: 32),
            isLowercaseHex(
                fields.selfGuardCodeDirectorySHA256,
                count: 64
            ),
            isLowercaseHex(
                fields.selfGuardFileImageRuntimeClosureContentEvidenceID,
                count: 64
            ),
            isLowercaseHex(
                fields.gitFileImageRuntimeClosureContentEvidenceID,
                count: 64
            ),
            fields.cpuTimeoutSeconds > 0,
            fields.maxAddressSpaceBytes > 0,
            fields.maxFileSizeBytes > 0,
            fields.maxOpenFiles > 0,
            fields.maxObjectDatabaseBytes > 0,
            fields.maxStdoutBytes > 0,
            fields.maxStderrBytes > 0,
            fields.wallTimeoutMilliseconds > 0,
            fields.terminationGraceMilliseconds > 0,
            fields.reapTimeoutMilliseconds > 0
        else {
            throw GitRuntimePolicyDenialV2Error.nonCanonicalPolicy
        }

        let (wallAndGrace, firstOverflow) =
            fields.wallTimeoutMilliseconds.addingReportingOverflow(
                fields.terminationGraceMilliseconds
            )
        let (cleanupDeadline, secondOverflow) =
            wallAndGrace.addingReportingOverflow(
                fields.reapTimeoutMilliseconds
            )
        guard !firstOverflow, !secondOverflow else {
            throw GitRuntimePolicyDenialV2Error.cleanupDeadlineOverflow
        }

        let ceilings = GitToolPolicyVerifier.phase1ResourceCeilings
        let values: [
            (
                GitRuntimePolicyDenialV2ResourceField,
                maximum: UInt64,
                actual: UInt64
            )
        ] = [
            (
                .cpuTimeoutSeconds,
                ceilings.cpuTimeoutSeconds,
                fields.cpuTimeoutSeconds
            ),
            (
                .maxAddressSpaceBytes,
                ceilings.maxAddressSpaceBytes,
                fields.maxAddressSpaceBytes
            ),
            (
                .maxFileSizeBytes,
                ceilings.maxFileSizeBytes,
                fields.maxFileSizeBytes
            ),
            (
                .maxOpenFiles,
                ceilings.maxOpenFiles,
                fields.maxOpenFiles
            ),
            (
                .maxObjectDatabaseBytes,
                ceilings.maxObjectDatabaseBytes,
                fields.maxObjectDatabaseBytes
            ),
            (
                .maxStdoutBytes,
                ceilings.maxStdoutBytes,
                fields.maxStdoutBytes
            ),
            (
                .maxStderrBytes,
                ceilings.maxStderrBytes,
                fields.maxStderrBytes
            ),
            (
                .wallTimeoutMilliseconds,
                ceilings.wallTimeoutMilliseconds,
                fields.wallTimeoutMilliseconds
            ),
            (
                .terminationGraceMilliseconds,
                terminationGraceMaximumMilliseconds,
                fields.terminationGraceMilliseconds
            ),
            (
                .reapTimeoutMilliseconds,
                reapTimeoutMaximumMilliseconds,
                fields.reapTimeoutMilliseconds
            ),
        ]
        for (field, maximum, actual) in values {
            guard actual <= maximum else {
                throw GitRuntimePolicyDenialV2Error
                    .resourceLimitExceedsCompileTimeCeiling(
                        field: field,
                        maximum: maximum,
                        actual: actual
                    )
            }
        }

        guard
            fields.terminationGraceMilliseconds <
                fields.reapTimeoutMilliseconds,
            fields.reapTimeoutMilliseconds <=
                fields.wallTimeoutMilliseconds,
            cleanupDeadline <= maximumCleanupDeadlineMilliseconds
        else {
            throw GitRuntimePolicyDenialV2Error
                .invalidWatchdogInterval
        }
    }

    static func validate(
        _ trustAnchor: GitRuntimePolicyDenialV2TrustAnchor
    ) throws {
        guard isLowercaseHex(
            trustAnchor.expectedCurrentPolicySHA256,
            count: 64
        ) else {
            throw GitRuntimePolicyDenialV2Error.invalidTrustAnchor(
                .expectedCurrentPolicySHA256
            )
        }
        guard trustAnchor.expectedCurrentPolicyBytes > 0 else {
            throw GitRuntimePolicyDenialV2Error.invalidTrustAnchor(
                .expectedCurrentPolicyBytes
            )
        }
        guard trustAnchor.minimumPolicyGeneration > 0 else {
            throw GitRuntimePolicyDenialV2Error.invalidTrustAnchor(
                .minimumPolicyGeneration
            )
        }
    }

    static func validate(
        _ reference: SignedClaimGitToolPolicyV2Reference
    ) throws {
        guard reference.policyDocument.policySHA256 ==
            reference.signedClaim.fields.toolManifest.sha256
        else {
            throw GitRuntimePolicyDenialV2Error
                .toolPolicyReferenceDiscontinuity(
                    .toolManifestDigest
                )
        }
        guard reference.policyDocument.policyBytes ==
            reference.signedClaim.fields.toolManifest.byteCount
        else {
            throw GitRuntimePolicyDenialV2Error
                .toolPolicyReferenceDiscontinuity(
                    .toolManifestByteCount
                )
        }
        guard reference.policyDocument.runtimePolicySHA256 ==
            reference.signedClaim.fields.policies.runtimeSHA256
        else {
            throw GitRuntimePolicyDenialV2Error
                .toolPolicyReferenceDiscontinuity(
                    .runtimePolicyDigest
                )
        }
    }

    static func validate(
        _ fields: GitRuntimePolicyDenialV2Fields,
        matches limits: GitToolPolicyResourceLimits
    ) throws {
        let values: [
            (
                GitRuntimePolicyDenialV2ResourceField,
                expected: UInt64,
                actual: UInt64
            )
        ] = [
            (
                .cpuTimeoutSeconds,
                limits.cpuTimeoutSeconds,
                fields.cpuTimeoutSeconds
            ),
            (
                .maxAddressSpaceBytes,
                limits.maxAddressSpaceBytes,
                fields.maxAddressSpaceBytes
            ),
            (
                .maxFileSizeBytes,
                limits.maxFileSizeBytes,
                fields.maxFileSizeBytes
            ),
            (
                .maxOpenFiles,
                limits.maxOpenFiles,
                fields.maxOpenFiles
            ),
            (
                .maxObjectDatabaseBytes,
                limits.maxObjectDatabaseBytes,
                fields.maxObjectDatabaseBytes
            ),
            (
                .maxStdoutBytes,
                limits.maxStdoutBytes,
                fields.maxStdoutBytes
            ),
            (
                .maxStderrBytes,
                limits.maxStderrBytes,
                fields.maxStderrBytes
            ),
            (
                .wallTimeoutMilliseconds,
                limits.wallTimeoutMilliseconds,
                fields.wallTimeoutMilliseconds
            ),
        ]
        for (field, expected, actual) in values {
            guard actual == expected else {
                throw GitRuntimePolicyDenialV2Error
                    .toolPolicyResourceMismatch(field)
            }
        }
    }

    static func isCanonicalDecimal(_ value: String) -> Bool {
        guard !value.isEmpty else { return false }
        if value.count > 1, value.first == "0" { return false }
        return value.utf8.allSatisfy { (0x30...0x39).contains($0) }
    }

    static func isLowercaseHex(_ value: String, count: Int) -> Bool {
        guard value.utf8.count == count else { return false }
        return value.utf8.allSatisfy {
            (0x30...0x39).contains($0) ||
                (0x61...0x66).contains($0)
        }
    }
}
