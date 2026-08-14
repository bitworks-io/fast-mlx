import Foundation

public struct GitToolPolicyResourceLimits: Equatable, Sendable {
    public let maxPackBytes: UInt64
    public let maxPackObjects: UInt64
    public let maxCommitBytes: UInt64
    public let maxSingleInflatedObjectBytes: UInt64
    public let maxTotalInflatedBytes: UInt64
    public let maxCompressionRatio: UInt64
    public let maxTreeDepth: UInt64
    public let maxTreeCount: UInt64
    public let maxObjectDatabaseBytes: UInt64
    public let maxStdoutBytes: UInt64
    public let maxStderrBytes: UInt64
    public let wallTimeoutMilliseconds: UInt64
    public let cpuTimeoutSeconds: UInt64
    public let maxAddressSpaceBytes: UInt64
    public let maxFileSizeBytes: UInt64
    public let maxOpenFiles: UInt64

    public init(
        maxPackBytes: UInt64,
        maxPackObjects: UInt64,
        maxCommitBytes: UInt64,
        maxSingleInflatedObjectBytes: UInt64,
        maxTotalInflatedBytes: UInt64,
        maxCompressionRatio: UInt64,
        maxTreeDepth: UInt64,
        maxTreeCount: UInt64,
        maxObjectDatabaseBytes: UInt64,
        maxStdoutBytes: UInt64,
        maxStderrBytes: UInt64,
        wallTimeoutMilliseconds: UInt64,
        cpuTimeoutSeconds: UInt64,
        maxAddressSpaceBytes: UInt64,
        maxFileSizeBytes: UInt64,
        maxOpenFiles: UInt64
    ) {
        self.maxPackBytes = maxPackBytes
        self.maxPackObjects = maxPackObjects
        self.maxCommitBytes = maxCommitBytes
        self.maxSingleInflatedObjectBytes =
            maxSingleInflatedObjectBytes
        self.maxTotalInflatedBytes = maxTotalInflatedBytes
        self.maxCompressionRatio = maxCompressionRatio
        self.maxTreeDepth = maxTreeDepth
        self.maxTreeCount = maxTreeCount
        self.maxObjectDatabaseBytes = maxObjectDatabaseBytes
        self.maxStdoutBytes = maxStdoutBytes
        self.maxStderrBytes = maxStderrBytes
        self.wallTimeoutMilliseconds = wallTimeoutMilliseconds
        self.cpuTimeoutSeconds = cpuTimeoutSeconds
        self.maxAddressSpaceBytes = maxAddressSpaceBytes
        self.maxFileSizeBytes = maxFileSizeBytes
        self.maxOpenFiles = maxOpenFiles
    }
}

public struct GitToolPolicyFields: Equatable, Sendable {
    public let policyGeneration: UInt64
    public let validFromUnixSeconds: UInt64
    public let validUntilUnixSeconds: UInt64
    public let executableSHA256: String
    public let executableBytes: UInt64
    public let executableMachOUUID: String
    public let executableCodeDirectorySHA256: String
    public let runtimeClosureManifestSHA256: String
    public let runtimeClosureManifestBytes: UInt64
    public let runtimePolicySHA256: String
    public let limits: GitToolPolicyResourceLimits

    public init(
        policyGeneration: UInt64,
        validFromUnixSeconds: UInt64,
        validUntilUnixSeconds: UInt64,
        executableSHA256: String,
        executableBytes: UInt64,
        executableMachOUUID: String,
        executableCodeDirectorySHA256: String,
        runtimeClosureManifestSHA256: String,
        runtimeClosureManifestBytes: UInt64,
        runtimePolicySHA256: String,
        limits: GitToolPolicyResourceLimits
    ) {
        self.policyGeneration = policyGeneration
        self.validFromUnixSeconds = validFromUnixSeconds
        self.validUntilUnixSeconds = validUntilUnixSeconds
        self.executableSHA256 = executableSHA256
        self.executableBytes = executableBytes
        self.executableMachOUUID = executableMachOUUID
        self.executableCodeDirectorySHA256 =
            executableCodeDirectorySHA256
        self.runtimeClosureManifestSHA256 =
            runtimeClosureManifestSHA256
        self.runtimeClosureManifestBytes =
            runtimeClosureManifestBytes
        self.runtimePolicySHA256 = runtimePolicySHA256
        self.limits = limits
    }
}

/// Immutable current-policy context supplied outside candidate and baseline
/// source trees. This value does not establish provenance for that context.
public struct GitToolPolicyTrustAnchor: Equatable, Sendable {
    public let expectedCurrentPolicySHA256: String
    public let expectedCurrentPolicyBytes: UInt64
    public let minimumPolicyGeneration: UInt64
    public let verificationUnixSeconds: UInt64

    public init(
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

public enum GitToolPolicyTrustAnchorField:
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

public enum GitToolPolicyResourceField:
    String,
    Equatable,
    Sendable
{
    case maxPackBytes = "max-pack-bytes"
    case maxPackObjects = "max-pack-objects"
    case maxCommitBytes = "max-commit-bytes"
    case maxSingleInflatedObjectBytes =
        "max-single-inflated-object-bytes"
    case maxTotalInflatedBytes = "max-total-inflated-bytes"
    case maxCompressionRatio = "max-compression-ratio"
    case maxTreeDepth = "max-tree-depth"
    case maxTreeCount = "max-tree-count"
    case maxObjectDatabaseBytes = "max-object-database-bytes"
    case maxStdoutBytes = "max-stdout-bytes"
    case maxStderrBytes = "max-stderr-bytes"
    case wallTimeoutMilliseconds = "wall-timeout-milliseconds"
    case cpuTimeoutSeconds = "cpu-timeout-seconds"
    case maxAddressSpaceBytes = "max-address-space-bytes"
    case maxFileSizeBytes = "max-file-size-bytes"
    case maxOpenFiles = "max-open-files"
}

public enum GitToolPolicyError: Error, Equatable, Sendable {
    case nonCanonicalPolicy
    case invalidTrustAnchor(GitToolPolicyTrustAnchorField)
    case policyDigestMismatch
    case policyByteCountMismatch(expected: UInt64, actual: UInt64)
    case policyGenerationRollback(minimum: UInt64, actual: UInt64)
    case policyNotYetValid
    case policyExpired
    case resourceLimitExceedsCompileTimeCeiling(
        field: GitToolPolicyResourceField,
        maximum: UInt64,
        actual: UInt64
    )
}

extension GitToolPolicyError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .nonCanonicalPolicy:
            "Git tool policy is not the canonical fixed-order UTF-8 form"
        case .invalidTrustAnchor(let field):
            "Git tool policy trust anchor \(field.rawValue) is not canonical"
        case .policyDigestMismatch:
            "Git tool policy bytes do not match the externally pinned current policy"
        case .policyByteCountMismatch(let expected, let actual):
            "Git tool policy has \(actual) bytes, expected \(expected)"
        case .policyGenerationRollback(let minimum, let actual):
            "Git tool policy generation \(actual) is below external floor \(minimum)"
        case .policyNotYetValid:
            "Git tool policy is not valid at the external verification time"
        case .policyExpired:
            "Git tool policy has expired at the external verification time"
        case .resourceLimitExceedsCompileTimeCeiling(
            let field,
            let maximum,
            let actual
        ):
            "Git tool policy \(field.rawValue) \(actual) exceeds compile-time ceiling \(maximum)"
        }
    }
}

enum GitToolPolicyClaimReferenceError: Error, Equatable, Sendable {
    case toolManifestDigestMismatch
    case toolManifestByteCountMismatch
    case runtimePolicyDigestMismatch
}

extension GitToolPolicyClaimReferenceError: CustomStringConvertible {
    var description: String {
        switch self {
        case .toolManifestDigestMismatch:
            "Git tool policy digest does not match the signed claim"
        case .toolManifestByteCountMismatch:
            "Git tool policy byte count does not match the signed claim"
        case .runtimePolicyDigestMismatch:
            "Git tool runtime policy digest does not match the signed claim"
        }
    }
}

/// Canonical policy bytes matched to one external current-policy anchor only.
///
/// This document does not admit the named executable or runtime closure,
/// match a signed run claim, authorize Git, import objects, mutate the
/// filesystem, materialize source, build, spawn, load a model, reserve output,
/// or publish.
struct AnchoredGitToolPolicyDocument: Equatable, Sendable {
    let policyFile: AdmittedFile
    let policySHA256: String
    let policyBytes: UInt64
    let policyGeneration: UInt64
    let validFromUnixSeconds: UInt64
    let validUntilUnixSeconds: UInt64
    let executableSHA256: String
    let runtimeClosureManifestSHA256: String
    let runtimePolicySHA256: String
    let fields: GitToolPolicyFields
    let canImportGitObjects = false
    let canMutateFileSystem = false
    let canSpawn = false
    let canBuild = false
    let canLoadModel = false
    let canReserveOutput = false
    let canPublish = false

    fileprivate init(
        policyFile: AdmittedFile,
        fields: GitToolPolicyFields
    ) {
        self.policyFile = policyFile
        self.policySHA256 = policyFile.sha256
        self.policyBytes = UInt64(policyFile.bytes.count)
        self.policyGeneration = fields.policyGeneration
        self.validFromUnixSeconds = fields.validFromUnixSeconds
        self.validUntilUnixSeconds = fields.validUntilUnixSeconds
        self.executableSHA256 = fields.executableSHA256
        self.runtimeClosureManifestSHA256 =
            fields.runtimeClosureManifestSHA256
        self.runtimePolicySHA256 = fields.runtimePolicySHA256
        self.fields = fields
    }
}

/// One exact signed-claim reference to one anchored policy document only.
///
/// This value has no ID and grants no executable, runtime, process,
/// filesystem, source-import, build, model, result, or publication authority.
struct SignedClaimGitToolPolicyReference: Equatable, Sendable {
    let signedClaim: OperatorSignedRunClaim
    let policyDocument: AnchoredGitToolPolicyDocument
    let canImportGitObjects = false
    let canMutateFileSystem = false
    let canSpawn = false
    let canBuild = false
    let canLoadModel = false
    let canReserveOutput = false
    let canPublish = false

    fileprivate init(
        signedClaim: OperatorSignedRunClaim,
        policyDocument: AnchoredGitToolPolicyDocument
    ) {
        self.signedClaim = signedClaim
        self.policyDocument = policyDocument
    }
}

public enum GitToolPolicyVerifier {
    public static let policyDomain =
        "fast-mlx-proof-control-git-tool-policy-v1"
    public static let policySubject =
        "absorbed-mla-source-import-git"
    public static let claimSubject =
        OperatorRunClaimSubject.absorbedMLALoadedResultPair.rawValue

    public static let phase1ResourceCeilings =
        GitToolPolicyResourceLimits(
            maxPackBytes: 134_217_728,
            maxPackObjects: 20_000,
            maxCommitBytes: 1_048_576,
            maxSingleInflatedObjectBytes: 16_777_216,
            maxTotalInflatedBytes: 268_435_456,
            maxCompressionRatio: 64,
            maxTreeDepth: 32,
            maxTreeCount: 10_000,
            maxObjectDatabaseBytes: 167_772_160,
            maxStdoutBytes: 67_108_864,
            maxStderrBytes: 1_048_576,
            wallTimeoutMilliseconds: 120_000,
            cpuTimeoutSeconds: 120,
            maxAddressSpaceBytes: 1_073_741_824,
            maxFileSizeBytes: 167_772_160,
            maxOpenFiles: 64
        )

    public static func policyBytes(
        fields: GitToolPolicyFields
    ) throws -> Data {
        try validate(fields)
        let limits = fields.limits
        let lines = [
            policyDomain,
            "subject=\(policySubject)",
            "policy_generation=\(fields.policyGeneration)",
            "valid_from_unix_seconds=\(fields.validFromUnixSeconds)",
            "valid_until_unix_seconds=\(fields.validUntilUnixSeconds)",
            "claim_subject=\(claimSubject)",
            "executable_sha256=\(fields.executableSHA256)",
            "executable_bytes=\(fields.executableBytes)",
            "executable_architecture=arm64",
            "executable_macho_uuid=\(fields.executableMachOUUID)",
            "executable_code_directory_sha256=\(fields.executableCodeDirectorySHA256)",
            "runtime_closure_manifest_sha256=\(fields.runtimeClosureManifestSHA256)",
            "runtime_closure_manifest_bytes=\(fields.runtimeClosureManifestBytes)",
            "runtime_policy_sha256=\(fields.runtimePolicySHA256)",
            "argv_profile=git-source-pack-ingress-v1",
            "environment_profile=git-source-pack-environment-v1",
            "fd_profile=self-guard-fchdir-then-stdio-v1",
            "sandbox_profile=git-source-pack-sandbox-v1",
            "cwd_profile=descriptor-anchored-git-directory-v1",
            "umask=0077",
            "cloexec_default=true",
            "network_policy=deny-all-v1",
            "descendant_process_policy=deny-all-v1",
            "config_policy=none-v1",
            "hook_policy=deny-v1",
            "attributes_policy=raw-only-v1",
            "filter_policy=deny-v1",
            "textconv_policy=deny-v1",
            "replace_refs_policy=deny-v1",
            "alternates_policy=deny-v1",
            "promisor_lazy_fetch_policy=deny-v1",
            "transport_policy=deny-v1",
            "pack_version=2",
            "object_format=sha1",
            "delta_depth=0",
            "max_invocations_per_role=4",
            "max_pack_bytes=\(limits.maxPackBytes)",
            "max_pack_objects=\(limits.maxPackObjects)",
            "max_commit_bytes=\(limits.maxCommitBytes)",
            "max_single_inflated_object_bytes=\(limits.maxSingleInflatedObjectBytes)",
            "max_total_inflated_bytes=\(limits.maxTotalInflatedBytes)",
            "max_compression_ratio=\(limits.maxCompressionRatio)",
            "max_tree_depth=\(limits.maxTreeDepth)",
            "max_tree_count=\(limits.maxTreeCount)",
            "max_odb_bytes=\(limits.maxObjectDatabaseBytes)",
            "max_stdout_bytes=\(limits.maxStdoutBytes)",
            "max_stderr_bytes=\(limits.maxStderrBytes)",
            "wall_timeout_milliseconds=\(limits.wallTimeoutMilliseconds)",
            "cpu_timeout_seconds=\(limits.cpuTimeoutSeconds)",
            "max_address_space_bytes=\(limits.maxAddressSpaceBytes)",
            "max_file_size_bytes=\(limits.maxFileSizeBytes)",
            "max_open_files=\(limits.maxOpenFiles)",
        ]
        return Data((lines.joined(separator: "\n") + "\n").utf8)
    }

    static func anchor(
        policyFile: AdmittedFile,
        trustAnchor: GitToolPolicyTrustAnchor
    ) throws -> AnchoredGitToolPolicyDocument {
        try validate(trustAnchor)
        guard policyFile.sha256 ==
            trustAnchor.expectedCurrentPolicySHA256
        else {
            throw GitToolPolicyError.policyDigestMismatch
        }

        let actualBytes = UInt64(policyFile.bytes.count)
        guard actualBytes == trustAnchor.expectedCurrentPolicyBytes else {
            throw GitToolPolicyError.policyByteCountMismatch(
                expected: trustAnchor.expectedCurrentPolicyBytes,
                actual: actualBytes
            )
        }

        let fields = try parsePolicy(policyFile.bytes)
        guard fields.policyGeneration >=
            trustAnchor.minimumPolicyGeneration
        else {
            throw GitToolPolicyError.policyGenerationRollback(
                minimum: trustAnchor.minimumPolicyGeneration,
                actual: fields.policyGeneration
            )
        }
        guard trustAnchor.verificationUnixSeconds >=
            fields.validFromUnixSeconds
        else {
            throw GitToolPolicyError.policyNotYetValid
        }
        guard trustAnchor.verificationUnixSeconds <=
            fields.validUntilUnixSeconds
        else {
            throw GitToolPolicyError.policyExpired
        }

        return AnchoredGitToolPolicyDocument(
            policyFile: policyFile,
            fields: fields
        )
    }

    static func reference(
        signedClaim: OperatorSignedRunClaim,
        policyDocument: AnchoredGitToolPolicyDocument
    ) throws -> SignedClaimGitToolPolicyReference {
        guard policyDocument.policySHA256 ==
            signedClaim.fields.toolManifest.sha256
        else {
            throw GitToolPolicyClaimReferenceError
                .toolManifestDigestMismatch
        }
        guard policyDocument.policyBytes ==
            signedClaim.fields.toolManifest.byteCount
        else {
            throw GitToolPolicyClaimReferenceError
                .toolManifestByteCountMismatch
        }
        guard policyDocument.runtimePolicySHA256 ==
            signedClaim.fields.policies.runtimeSHA256
        else {
            throw GitToolPolicyClaimReferenceError
                .runtimePolicyDigestMismatch
        }

        return SignedClaimGitToolPolicyReference(
            signedClaim: signedClaim,
            policyDocument: policyDocument
        )
    }
}

private extension GitToolPolicyVerifier {
    struct CanonicalLineParser {
        let lines: [Substring]
        var index = 0

        mutating func require(_ expected: String) throws {
            guard
                index < lines.count,
                lines[index] == Substring(expected)
            else {
                throw GitToolPolicyError.nonCanonicalPolicy
            }
            index += 1
        }

        mutating func value(prefix: String) throws -> String {
            guard index < lines.count, lines[index].hasPrefix(prefix) else {
                throw GitToolPolicyError.nonCanonicalPolicy
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
                throw GitToolPolicyError.nonCanonicalPolicy
            }
            return result
        }

        mutating func uuid(prefix: String) throws -> String {
            let result = try value(prefix: prefix)
            guard isLowercaseHex(result, count: 32) else {
                throw GitToolPolicyError.nonCanonicalPolicy
            }
            return result
        }

        mutating func uint64(prefix: String) throws -> UInt64 {
            let result = try value(prefix: prefix)
            guard
                isCanonicalDecimal(result),
                let value = UInt64(result)
            else {
                throw GitToolPolicyError.nonCanonicalPolicy
            }
            return value
        }

        var isAtEnd: Bool {
            index == lines.count
        }
    }

    static func parsePolicy(_ bytes: Data) throws
        -> GitToolPolicyFields
    {
        guard
            let text = String(data: bytes, encoding: .utf8),
            Data(text.utf8) == bytes
        else {
            throw GitToolPolicyError.nonCanonicalPolicy
        }

        let splitLines = text.split(
            separator: "\n",
            omittingEmptySubsequences: false
        )
        guard splitLines.count == 53, splitLines.last?.isEmpty == true else {
            throw GitToolPolicyError.nonCanonicalPolicy
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
        try parser.require("claim_subject=\(claimSubject)")
        let executableSHA256 = try parser.sha256(
            prefix: "executable_sha256="
        )
        let executableBytes = try parser.uint64(
            prefix: "executable_bytes="
        )
        try parser.require("executable_architecture=arm64")
        let executableMachOUUID = try parser.uuid(
            prefix: "executable_macho_uuid="
        )
        let executableCodeDirectorySHA256 = try parser.sha256(
            prefix: "executable_code_directory_sha256="
        )
        let runtimeClosureManifestSHA256 = try parser.sha256(
            prefix: "runtime_closure_manifest_sha256="
        )
        let runtimeClosureManifestBytes = try parser.uint64(
            prefix: "runtime_closure_manifest_bytes="
        )
        let runtimePolicySHA256 = try parser.sha256(
            prefix: "runtime_policy_sha256="
        )
        try parser.require("argv_profile=git-source-pack-ingress-v1")
        try parser.require(
            "environment_profile=git-source-pack-environment-v1"
        )
        try parser.require(
            "fd_profile=self-guard-fchdir-then-stdio-v1"
        )
        try parser.require("sandbox_profile=git-source-pack-sandbox-v1")
        try parser.require(
            "cwd_profile=descriptor-anchored-git-directory-v1"
        )
        try parser.require("umask=0077")
        try parser.require("cloexec_default=true")
        try parser.require("network_policy=deny-all-v1")
        try parser.require("descendant_process_policy=deny-all-v1")
        try parser.require("config_policy=none-v1")
        try parser.require("hook_policy=deny-v1")
        try parser.require("attributes_policy=raw-only-v1")
        try parser.require("filter_policy=deny-v1")
        try parser.require("textconv_policy=deny-v1")
        try parser.require("replace_refs_policy=deny-v1")
        try parser.require("alternates_policy=deny-v1")
        try parser.require("promisor_lazy_fetch_policy=deny-v1")
        try parser.require("transport_policy=deny-v1")
        try parser.require("pack_version=2")
        try parser.require("object_format=sha1")
        try parser.require("delta_depth=0")
        try parser.require("max_invocations_per_role=4")

        let limits = GitToolPolicyResourceLimits(
            maxPackBytes: try parser.uint64(
                prefix: "max_pack_bytes="
            ),
            maxPackObjects: try parser.uint64(
                prefix: "max_pack_objects="
            ),
            maxCommitBytes: try parser.uint64(
                prefix: "max_commit_bytes="
            ),
            maxSingleInflatedObjectBytes: try parser.uint64(
                prefix: "max_single_inflated_object_bytes="
            ),
            maxTotalInflatedBytes: try parser.uint64(
                prefix: "max_total_inflated_bytes="
            ),
            maxCompressionRatio: try parser.uint64(
                prefix: "max_compression_ratio="
            ),
            maxTreeDepth: try parser.uint64(
                prefix: "max_tree_depth="
            ),
            maxTreeCount: try parser.uint64(
                prefix: "max_tree_count="
            ),
            maxObjectDatabaseBytes: try parser.uint64(
                prefix: "max_odb_bytes="
            ),
            maxStdoutBytes: try parser.uint64(
                prefix: "max_stdout_bytes="
            ),
            maxStderrBytes: try parser.uint64(
                prefix: "max_stderr_bytes="
            ),
            wallTimeoutMilliseconds: try parser.uint64(
                prefix: "wall_timeout_milliseconds="
            ),
            cpuTimeoutSeconds: try parser.uint64(
                prefix: "cpu_timeout_seconds="
            ),
            maxAddressSpaceBytes: try parser.uint64(
                prefix: "max_address_space_bytes="
            ),
            maxFileSizeBytes: try parser.uint64(
                prefix: "max_file_size_bytes="
            ),
            maxOpenFiles: try parser.uint64(
                prefix: "max_open_files="
            )
        )
        guard parser.isAtEnd else {
            throw GitToolPolicyError.nonCanonicalPolicy
        }

        let fields = GitToolPolicyFields(
            policyGeneration: policyGeneration,
            validFromUnixSeconds: validFromUnixSeconds,
            validUntilUnixSeconds: validUntilUnixSeconds,
            executableSHA256: executableSHA256,
            executableBytes: executableBytes,
            executableMachOUUID: executableMachOUUID,
            executableCodeDirectorySHA256:
                executableCodeDirectorySHA256,
            runtimeClosureManifestSHA256:
                runtimeClosureManifestSHA256,
            runtimeClosureManifestBytes:
                runtimeClosureManifestBytes,
            runtimePolicySHA256: runtimePolicySHA256,
            limits: limits
        )
        try validate(fields)
        return fields
    }

    static func validate(_ fields: GitToolPolicyFields) throws {
        guard
            fields.policyGeneration > 0,
            fields.validFromUnixSeconds <=
                fields.validUntilUnixSeconds,
            isLowercaseHex(fields.executableSHA256, count: 64),
            fields.executableBytes > 0,
            isLowercaseHex(fields.executableMachOUUID, count: 32),
            isLowercaseHex(
                fields.executableCodeDirectorySHA256,
                count: 64
            ),
            isLowercaseHex(
                fields.runtimeClosureManifestSHA256,
                count: 64
            ),
            fields.runtimeClosureManifestBytes > 0,
            isLowercaseHex(fields.runtimePolicySHA256, count: 64)
        else {
            throw GitToolPolicyError.nonCanonicalPolicy
        }
        try validate(fields.limits)
    }

    static func validate(_ limits: GitToolPolicyResourceLimits) throws {
        let ceilings = phase1ResourceCeilings
        let values: [
            (
                GitToolPolicyResourceField,
                maximum: UInt64,
                actual: UInt64
            )
        ] = [
            (.maxPackBytes, ceilings.maxPackBytes, limits.maxPackBytes),
            (
                .maxPackObjects,
                ceilings.maxPackObjects,
                limits.maxPackObjects
            ),
            (
                .maxCommitBytes,
                ceilings.maxCommitBytes,
                limits.maxCommitBytes
            ),
            (
                .maxSingleInflatedObjectBytes,
                ceilings.maxSingleInflatedObjectBytes,
                limits.maxSingleInflatedObjectBytes
            ),
            (
                .maxTotalInflatedBytes,
                ceilings.maxTotalInflatedBytes,
                limits.maxTotalInflatedBytes
            ),
            (
                .maxCompressionRatio,
                ceilings.maxCompressionRatio,
                limits.maxCompressionRatio
            ),
            (
                .maxTreeDepth,
                ceilings.maxTreeDepth,
                limits.maxTreeDepth
            ),
            (
                .maxTreeCount,
                ceilings.maxTreeCount,
                limits.maxTreeCount
            ),
            (
                .maxObjectDatabaseBytes,
                ceilings.maxObjectDatabaseBytes,
                limits.maxObjectDatabaseBytes
            ),
            (
                .maxStdoutBytes,
                ceilings.maxStdoutBytes,
                limits.maxStdoutBytes
            ),
            (
                .maxStderrBytes,
                ceilings.maxStderrBytes,
                limits.maxStderrBytes
            ),
            (
                .wallTimeoutMilliseconds,
                ceilings.wallTimeoutMilliseconds,
                limits.wallTimeoutMilliseconds
            ),
            (
                .cpuTimeoutSeconds,
                ceilings.cpuTimeoutSeconds,
                limits.cpuTimeoutSeconds
            ),
            (
                .maxAddressSpaceBytes,
                ceilings.maxAddressSpaceBytes,
                limits.maxAddressSpaceBytes
            ),
            (
                .maxFileSizeBytes,
                ceilings.maxFileSizeBytes,
                limits.maxFileSizeBytes
            ),
            (
                .maxOpenFiles,
                ceilings.maxOpenFiles,
                limits.maxOpenFiles
            ),
        ]

        for (field, maximum, actual) in values {
            guard actual > 0 else {
                throw GitToolPolicyError.nonCanonicalPolicy
            }
            guard actual <= maximum else {
                throw GitToolPolicyError
                    .resourceLimitExceedsCompileTimeCeiling(
                        field: field,
                        maximum: maximum,
                        actual: actual
                    )
            }
        }
    }

    static func validate(
        _ trustAnchor: GitToolPolicyTrustAnchor
    ) throws {
        guard isLowercaseHex(
            trustAnchor.expectedCurrentPolicySHA256,
            count: 64
        ) else {
            throw GitToolPolicyError.invalidTrustAnchor(
                .expectedCurrentPolicySHA256
            )
        }
        guard trustAnchor.expectedCurrentPolicyBytes > 0 else {
            throw GitToolPolicyError.invalidTrustAnchor(
                .expectedCurrentPolicyBytes
            )
        }
        guard trustAnchor.minimumPolicyGeneration > 0 else {
            throw GitToolPolicyError.invalidTrustAnchor(
                .minimumPolicyGeneration
            )
        }
    }

    static func isLowercaseHex(_ value: String, count: Int) -> Bool {
        value.utf8.count == count && value.utf8.allSatisfy {
            (0x30...0x39).contains($0) || (0x61...0x66).contains($0)
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
