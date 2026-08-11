import Foundation

struct GitRuntimePolicyDenialFields: Equatable, Sendable {
    let policyGeneration: UInt64
    let validFromUnixSeconds: UInt64
    let validUntilUnixSeconds: UInt64
    let selfGuardSHA256: String
    let selfGuardBytes: UInt64
    let selfGuardMachOUUID: String
    let selfGuardCodeDirectorySHA256: String
    let selfGuardRuntimeClosureID: String
    let gitRuntimeClosureID: String
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
        selfGuardRuntimeClosureID: String,
        gitRuntimeClosureID: String,
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
        self.selfGuardRuntimeClosureID =
            selfGuardRuntimeClosureID
        self.gitRuntimeClosureID = gitRuntimeClosureID
        self.cpuTimeoutSeconds = limits.cpuTimeoutSeconds
        self.maxAddressSpaceBytes = limits.maxAddressSpaceBytes
        self.maxFileSizeBytes = limits.maxFileSizeBytes
        self.maxOpenFiles = limits.maxOpenFiles
        self.maxObjectDatabaseBytes = limits.maxObjectDatabaseBytes
        self.maxStdoutBytes = limits.maxStdoutBytes
        self.maxStderrBytes = limits.maxStderrBytes
        self.wallTimeoutMilliseconds =
            limits.wallTimeoutMilliseconds
        self.terminationGraceMilliseconds =
            terminationGraceMilliseconds
        self.reapTimeoutMilliseconds = reapTimeoutMilliseconds
    }
}

/// Immutable current runtime-policy context supplied outside both source
/// roles. This value establishes comparison context only.
struct GitRuntimePolicyDenialTrustAnchor: Equatable, Sendable {
    let expectedCurrentPolicySHA256: String
    let expectedCurrentPolicyBytes: UInt64
    let minimumPolicyGeneration: UInt64
    let verificationUnixSeconds: UInt64

    init(
        expectedCurrentPolicySHA256: String,
        expectedCurrentPolicyBytes: UInt64,
        minimumPolicyGeneration: UInt64,
        verificationUnixSeconds: UInt64
    ) {
        self.expectedCurrentPolicySHA256 =
            expectedCurrentPolicySHA256
        self.expectedCurrentPolicyBytes =
            expectedCurrentPolicyBytes
        self.minimumPolicyGeneration = minimumPolicyGeneration
        self.verificationUnixSeconds = verificationUnixSeconds
    }
}

enum GitRuntimePolicyDenialTrustAnchorField:
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

enum GitRuntimePolicyDenialResourceField:
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

enum GitRuntimePolicyReferenceField:
    String,
    Equatable,
    Sendable
{
    case toolManifestDigest = "tool-manifest-digest"
    case toolManifestByteCount = "tool-manifest-byte-count"
    case runtimePolicyDigest = "runtime-policy-digest"
}

enum GitRuntimePolicyDenialError: Error, Equatable, Sendable {
    case nonCanonicalPolicy
    case invalidTrustAnchor(
        GitRuntimePolicyDenialTrustAnchorField
    )
    case policyDigestMismatch
    case policyByteCountMismatch(expected: UInt64, actual: UInt64)
    case policyGenerationRollback(minimum: UInt64, actual: UInt64)
    case policyNotYetValid
    case policyExpired
    case resourceLimitExceedsCompileTimeCeiling(
        field: GitRuntimePolicyDenialResourceField,
        maximum: UInt64,
        actual: UInt64
    )
    case cleanupDeadlineOverflow
    case invalidWatchdogInterval
    case toolPolicyReferenceDiscontinuity(
        GitRuntimePolicyReferenceField
    )
    case runtimePolicyDigestMismatch
    case toolPolicyResourceMismatch(
        GitRuntimePolicyDenialResourceField
    )
    case supportedSandboxMechanismUnavailable
}

extension GitRuntimePolicyDenialError: CustomStringConvertible {
    var description: String {
        switch self {
        case .nonCanonicalPolicy:
            "Git runtime policy is not the canonical denial document"
        case .invalidTrustAnchor(let field):
            "Git runtime policy trust anchor \(field.rawValue) is not canonical"
        case .policyDigestMismatch:
            "Git runtime policy does not match the external digest"
        case .policyByteCountMismatch(let expected, let actual):
            "Git runtime policy has \(actual) bytes, expected \(expected)"
        case .policyGenerationRollback(let minimum, let actual):
            "Git runtime policy generation \(actual) is below \(minimum)"
        case .policyNotYetValid:
            "Git runtime policy is not valid at the verification time"
        case .policyExpired:
            "Git runtime policy has expired"
        case .resourceLimitExceedsCompileTimeCeiling(
            let field,
            let maximum,
            let actual
        ):
            "Git runtime policy \(field.rawValue) \(actual) exceeds \(maximum)"
        case .cleanupDeadlineOverflow:
            "Git runtime policy cleanup deadline overflows UInt64"
        case .invalidWatchdogInterval:
            "Git runtime policy watchdog intervals are inconsistent"
        case .toolPolicyReferenceDiscontinuity(let field):
            "Git tool policy reference \(field.rawValue) is discontinuous"
        case .runtimePolicyDigestMismatch:
            "Git runtime policy digest does not match the tool/claim reference"
        case .toolPolicyResourceMismatch(let field):
            "Git runtime policy \(field.rawValue) does not match the Git tool policy"
        case .supportedSandboxMechanismUnavailable:
            "No supported sandbox mechanism satisfies the required runtime policy"
        }
    }
}

/// Exact current runtime-policy bytes anchored to external comparison
/// context. This document records a denial only and grants no authority.
struct AnchoredGitRuntimePolicyDenialDocument:
    Equatable,
    Sendable
{
    let policyFile: AdmittedFile
    let policySHA256: String
    let policyBytes: UInt64
    let policyGeneration: UInt64
    let validFromUnixSeconds: UInt64
    let validUntilUnixSeconds: UInt64
    let fields: GitRuntimePolicyDenialFields
    let canImportGitObjects = false
    let canMutateFileSystem = false
    let canAccessNetwork = false
    let canConsumePack = false
    let canSpawn = false
    let canBuild = false
    let canLoadModel = false
    let canReserveOutput = false
    let canPublish = false

    fileprivate init(
        policyFile: AdmittedFile,
        fields: GitRuntimePolicyDenialFields
    ) {
        self.policyFile = policyFile
        self.policySHA256 = policyFile.sha256
        self.policyBytes = UInt64(policyFile.bytes.count)
        self.policyGeneration = fields.policyGeneration
        self.validFromUnixSeconds = fields.validFromUnixSeconds
        self.validUntilUnixSeconds = fields.validUntilUnixSeconds
        self.fields = fields
    }
}

enum GitRuntimePolicyDenialVerifier {
    static let policyDomain =
        "fast-mlx-proof-control-git-runtime-policy-v1"
    static let policySubject =
        "absorbed-mla-source-import-git-runtime"
    static let terminationGraceMaximumMilliseconds: UInt64 = 2_000
    static let reapTimeoutMaximumMilliseconds: UInt64 = 5_000
    static let maximumCleanupDeadlineMilliseconds: UInt64 = 127_000

    static func policyBytes(
        fields: GitRuntimePolicyDenialFields
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
            "self_guard_runtime_closure_id=\(fields.selfGuardRuntimeClosureID)",
            "git_runtime_closure_id=\(fields.gitRuntimeClosureID)",
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
        return Data((lines.joined(separator: "\n") + "\n").utf8)
    }

    static func anchor(
        policyFile: AdmittedFile,
        trustAnchor: GitRuntimePolicyDenialTrustAnchor
    ) throws -> AnchoredGitRuntimePolicyDenialDocument {
        try validate(trustAnchor)
        guard policyFile.sha256 ==
            trustAnchor.expectedCurrentPolicySHA256
        else {
            throw GitRuntimePolicyDenialError.policyDigestMismatch
        }

        let actualBytes = UInt64(policyFile.bytes.count)
        guard actualBytes == trustAnchor.expectedCurrentPolicyBytes else {
            throw GitRuntimePolicyDenialError.policyByteCountMismatch(
                expected: trustAnchor.expectedCurrentPolicyBytes,
                actual: actualBytes
            )
        }

        let fields = try parsePolicy(policyFile.bytes)
        guard fields.policyGeneration >=
            trustAnchor.minimumPolicyGeneration
        else {
            throw GitRuntimePolicyDenialError
                .policyGenerationRollback(
                    minimum: trustAnchor.minimumPolicyGeneration,
                    actual: fields.policyGeneration
                )
        }
        guard trustAnchor.verificationUnixSeconds >=
            fields.validFromUnixSeconds
        else {
            throw GitRuntimePolicyDenialError.policyNotYetValid
        }
        guard trustAnchor.verificationUnixSeconds <=
            fields.validUntilUnixSeconds
        else {
            throw GitRuntimePolicyDenialError.policyExpired
        }

        return AnchoredGitRuntimePolicyDenialDocument(
            policyFile: policyFile,
            fields: fields
        )
    }

    static func requireUnavailable(
        policyDocument: AnchoredGitRuntimePolicyDenialDocument,
        toolPolicyReference: SignedClaimGitToolPolicyReference
    ) throws -> Never {
        try validate(toolPolicyReference)
        guard policyDocument.policySHA256 ==
            toolPolicyReference.policyDocument.runtimePolicySHA256,
            policyDocument.policySHA256 ==
            toolPolicyReference.signedClaim.fields.policies.runtimeSHA256
        else {
            throw GitRuntimePolicyDenialError
                .runtimePolicyDigestMismatch
        }
        try validate(
            policyDocument.fields,
            matches: toolPolicyReference.policyDocument.fields.limits
        )
        throw GitRuntimePolicyDenialError
            .supportedSandboxMechanismUnavailable
    }
}

private extension GitRuntimePolicyDenialVerifier {
    struct CanonicalLineParser {
        let lines: [Substring]
        var index = 0

        mutating func require(_ expected: String) throws {
            guard
                index < lines.count,
                lines[index] == Substring(expected)
            else {
                throw GitRuntimePolicyDenialError
                    .nonCanonicalPolicy
            }
            index += 1
        }

        mutating func value(prefix: String) throws -> String {
            guard index < lines.count, lines[index].hasPrefix(prefix) else {
                throw GitRuntimePolicyDenialError
                    .nonCanonicalPolicy
            }
            let result = String(
                lines[index].dropFirst(prefix.utf8.count)
            )
            index += 1
            return result
        }

        mutating func sha256(prefix: String) throws -> String {
            let result = try value(prefix: prefix)
            guard isLowercaseHex(result, count: 64) else {
                throw GitRuntimePolicyDenialError
                    .nonCanonicalPolicy
            }
            return result
        }

        mutating func uuid(prefix: String) throws -> String {
            let result = try value(prefix: prefix)
            guard isLowercaseHex(result, count: 32) else {
                throw GitRuntimePolicyDenialError
                    .nonCanonicalPolicy
            }
            return result
        }

        mutating func uint64(prefix: String) throws -> UInt64 {
            let result = try value(prefix: prefix)
            guard
                isCanonicalDecimal(result),
                let value = UInt64(result)
            else {
                throw GitRuntimePolicyDenialError
                    .nonCanonicalPolicy
            }
            return value
        }

        var isAtEnd: Bool {
            index == lines.count
        }
    }

    static func parsePolicy(_ bytes: Data) throws
        -> GitRuntimePolicyDenialFields
    {
        guard
            let text = String(data: bytes, encoding: .utf8),
            Data(text.utf8) == bytes
        else {
            throw GitRuntimePolicyDenialError.nonCanonicalPolicy
        }

        let splitLines = text.split(
            separator: "\n",
            omittingEmptySubsequences: false
        )
        guard splitLines.count == 58, splitLines.last?.isEmpty == true else {
            throw GitRuntimePolicyDenialError.nonCanonicalPolicy
        }

        var parser = CanonicalLineParser(
            lines: Array(splitLines.dropLast())
        )
        try parser.require(policyDomain)
        try parser.require("subject=\(policySubject)")
        let policyGeneration = try parser.uint64(
            prefix: "policy_generation="
        )
        let validFromUnixSeconds = try parser.uint64(
            prefix: "valid_from_unix_seconds="
        )
        let validUntilUnixSeconds = try parser.uint64(
            prefix: "valid_until_unix_seconds="
        )
        try parser.require("platform_architecture=arm64")
        try parser.require("platform_hardware_model=Mac15,14")
        try parser.require("platform_os_version=26.5.2")
        try parser.require("platform_os_build=25F84")
        try parser.require("platform_xcode_version=26.6")
        try parser.require("platform_xcode_build=17F113")
        try parser.require("platform_sdk_version=26.5")
        let selfGuardSHA256 = try parser.sha256(
            prefix: "self_guard_sha256="
        )
        let selfGuardBytes = try parser.uint64(
            prefix: "self_guard_bytes="
        )
        let selfGuardMachOUUID = try parser.uuid(
            prefix: "self_guard_macho_uuid="
        )
        let selfGuardCodeDirectorySHA256 = try parser.sha256(
            prefix: "self_guard_code_directory_sha256="
        )
        let selfGuardRuntimeClosureID = try parser.sha256(
            prefix: "self_guard_runtime_closure_id="
        )
        let gitRuntimeClosureID = try parser.sha256(
            prefix: "git_runtime_closure_id="
        )
        try parser.require(
            "spawn_profile=posix-spawn-self-guard-v1"
        )
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
        try parser.require(
            "argv_profile=git-source-pack-ingress-v1"
        )
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
        let cpuTimeoutSeconds = try parser.uint64(
            prefix: "cpu_timeout_seconds="
        )
        let maxAddressSpaceBytes = try parser.uint64(
            prefix: "max_address_space_bytes="
        )
        let maxFileSizeBytes = try parser.uint64(
            prefix: "max_file_size_bytes="
        )
        let maxOpenFiles = try parser.uint64(
            prefix: "max_open_files="
        )
        let maxObjectDatabaseBytes = try parser.uint64(
            prefix: "max_odb_bytes="
        )
        let maxStdoutBytes = try parser.uint64(
            prefix: "max_stdout_bytes="
        )
        let maxStderrBytes = try parser.uint64(
            prefix: "max_stderr_bytes="
        )
        let wallTimeoutMilliseconds = try parser.uint64(
            prefix: "wall_timeout_milliseconds="
        )
        let terminationGraceMilliseconds = try parser.uint64(
            prefix: "termination_grace_milliseconds="
        )
        let reapTimeoutMilliseconds = try parser.uint64(
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
        guard parser.isAtEnd else {
            throw GitRuntimePolicyDenialError.nonCanonicalPolicy
        }

        let limits = GitToolPolicyResourceLimits(
            maxPackBytes:
                GitToolPolicyVerifier.phase1ResourceCeilings.maxPackBytes,
            maxPackObjects:
                GitToolPolicyVerifier.phase1ResourceCeilings.maxPackObjects,
            maxCommitBytes:
                GitToolPolicyVerifier.phase1ResourceCeilings.maxCommitBytes,
            maxSingleInflatedObjectBytes:
                GitToolPolicyVerifier.phase1ResourceCeilings
                .maxSingleInflatedObjectBytes,
            maxTotalInflatedBytes:
                GitToolPolicyVerifier.phase1ResourceCeilings
                .maxTotalInflatedBytes,
            maxCompressionRatio:
                GitToolPolicyVerifier.phase1ResourceCeilings
                .maxCompressionRatio,
            maxTreeDepth:
                GitToolPolicyVerifier.phase1ResourceCeilings.maxTreeDepth,
            maxTreeCount:
                GitToolPolicyVerifier.phase1ResourceCeilings.maxTreeCount,
            maxObjectDatabaseBytes: maxObjectDatabaseBytes,
            maxStdoutBytes: maxStdoutBytes,
            maxStderrBytes: maxStderrBytes,
            wallTimeoutMilliseconds: wallTimeoutMilliseconds,
            cpuTimeoutSeconds: cpuTimeoutSeconds,
            maxAddressSpaceBytes: maxAddressSpaceBytes,
            maxFileSizeBytes: maxFileSizeBytes,
            maxOpenFiles: maxOpenFiles
        )
        let fields = GitRuntimePolicyDenialFields(
            policyGeneration: policyGeneration,
            validFromUnixSeconds: validFromUnixSeconds,
            validUntilUnixSeconds: validUntilUnixSeconds,
            selfGuardSHA256: selfGuardSHA256,
            selfGuardBytes: selfGuardBytes,
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
        try validate(fields)
        guard try policyBytes(fields: fields) == bytes else {
            throw GitRuntimePolicyDenialError.nonCanonicalPolicy
        }
        return fields
    }

    static func validate(
        _ fields: GitRuntimePolicyDenialFields
    ) throws {
        guard
            fields.policyGeneration > 0,
            fields.validFromUnixSeconds <=
                fields.validUntilUnixSeconds,
            isLowercaseHex(fields.selfGuardSHA256, count: 64),
            fields.selfGuardBytes > 0,
            isLowercaseHex(fields.selfGuardMachOUUID, count: 32),
            isLowercaseHex(
                fields.selfGuardCodeDirectorySHA256,
                count: 64
            ),
            isLowercaseHex(
                fields.selfGuardRuntimeClosureID,
                count: 64
            ),
            isLowercaseHex(fields.gitRuntimeClosureID, count: 64),
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
            throw GitRuntimePolicyDenialError.nonCanonicalPolicy
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
            throw GitRuntimePolicyDenialError
                .cleanupDeadlineOverflow
        }

        let ceilings = GitToolPolicyVerifier.phase1ResourceCeilings
        let values: [
            (
                GitRuntimePolicyDenialResourceField,
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
                throw GitRuntimePolicyDenialError
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
            throw GitRuntimePolicyDenialError
                .invalidWatchdogInterval
        }
    }

    static func validate(
        _ trustAnchor: GitRuntimePolicyDenialTrustAnchor
    ) throws {
        guard isLowercaseHex(
            trustAnchor.expectedCurrentPolicySHA256,
            count: 64
        ) else {
            throw GitRuntimePolicyDenialError.invalidTrustAnchor(
                .expectedCurrentPolicySHA256
            )
        }
        guard trustAnchor.expectedCurrentPolicyBytes > 0 else {
            throw GitRuntimePolicyDenialError.invalidTrustAnchor(
                .expectedCurrentPolicyBytes
            )
        }
        guard trustAnchor.minimumPolicyGeneration > 0 else {
            throw GitRuntimePolicyDenialError.invalidTrustAnchor(
                .minimumPolicyGeneration
            )
        }
    }

    static func validate(
        _ reference: SignedClaimGitToolPolicyReference
    ) throws {
        guard reference.policyDocument.policySHA256 ==
            reference.signedClaim.fields.toolManifest.sha256
        else {
            throw GitRuntimePolicyDenialError
                .toolPolicyReferenceDiscontinuity(
                    .toolManifestDigest
                )
        }
        guard reference.policyDocument.policyBytes ==
            reference.signedClaim.fields.toolManifest.byteCount
        else {
            throw GitRuntimePolicyDenialError
                .toolPolicyReferenceDiscontinuity(
                    .toolManifestByteCount
                )
        }
        guard reference.policyDocument.runtimePolicySHA256 ==
            reference.signedClaim.fields.policies.runtimeSHA256
        else {
            throw GitRuntimePolicyDenialError
                .toolPolicyReferenceDiscontinuity(
                    .runtimePolicyDigest
                )
        }
    }

    static func validate(
        _ fields: GitRuntimePolicyDenialFields,
        matches limits: GitToolPolicyResourceLimits
    ) throws {
        let values: [
            (
                GitRuntimePolicyDenialResourceField,
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
                throw GitRuntimePolicyDenialError
                    .toolPolicyResourceMismatch(field)
            }
        }
    }

    static func isLowercaseHex(_ value: String, count: Int) -> Bool {
        value.utf8.count == count && value.utf8.allSatisfy {
            (0x30...0x39).contains($0) ||
                (0x61...0x66).contains($0)
        }
    }

    static func isCanonicalDecimal(_ value: String) -> Bool {
        guard !value.isEmpty, value.utf8.allSatisfy({
            (0x30...0x39).contains($0)
        }) else {
            return false
        }
        return value == "0" || value.utf8.first != 0x30
    }
}
