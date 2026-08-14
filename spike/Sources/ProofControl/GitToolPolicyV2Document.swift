import Foundation

struct GitToolPolicyV2Fields: Equatable, Sendable {
    let policyGeneration: UInt64
    let validFromUnixSeconds: UInt64
    let validUntilUnixSeconds: UInt64
    let executableSHA256: String
    let executableBytes: UInt64
    let executableMachOUUID: String
    let executableCodeDirectorySHA256: String
    let gitRuntimeClosureManifestSHA256: String
    let gitRuntimeClosureManifestBytes: UInt64
    let gitFileImageRuntimeClosureContentEvidenceID: String
    let runtimePolicySHA256: String
    let limits: GitToolPolicyResourceLimits
}

/// Immutable current-policy context supplied outside both source roles.
/// This value authenticates comparison bytes only and grants no authority.
struct GitToolPolicyV2TrustAnchor: Equatable, Sendable {
    let expectedCurrentPolicySHA256: String
    let expectedCurrentPolicyBytes: UInt64
    let minimumPolicyGeneration: UInt64
    let verificationUnixSeconds: UInt64
}

enum GitToolPolicyV2TrustAnchorField:
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

enum GitToolPolicyV2ResourceField:
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

enum GitToolPolicyV2Error: Error, Equatable, Sendable {
    case policySize
    case nonCanonicalPolicy
    case invalidTrustAnchor(GitToolPolicyV2TrustAnchorField)
    case policyDigestMismatch
    case policyByteCountMismatch(expected: UInt64, actual: UInt64)
    case policyGenerationRollback(minimum: UInt64, actual: UInt64)
    case policyNotYetValid
    case policyExpired
    case resourceLimitExceedsCompileTimeCeiling(
        field: GitToolPolicyV2ResourceField,
        maximum: UInt64,
        actual: UInt64
    )
}

enum GitToolPolicyV2ClaimReferenceError:
    Error,
    Equatable,
    Sendable
{
    case toolManifestDigestMismatch
    case toolManifestByteCountMismatch
    case runtimePolicyDigestMismatch
}

/// Exact v2 policy bytes matched to one external current-policy anchor.
/// "Anchored" means comparison-only; this value grants no runtime authority.
struct AnchoredGitToolPolicyV2Document: Equatable, Sendable {
    let policyFile: AdmittedFile
    let policySHA256: String
    let policyBytes: UInt64
    let policyGeneration: UInt64
    let validFromUnixSeconds: UInt64
    let validUntilUnixSeconds: UInt64
    let executableSHA256: String
    let gitRuntimeClosureManifestSHA256: String
    let gitFileImageRuntimeClosureContentEvidenceID: String
    let runtimePolicySHA256: String
    let fields: GitToolPolicyV2Fields

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
        fields: GitToolPolicyV2Fields
    ) {
        self.policyFile = policyFile
        self.policySHA256 = policyFile.sha256
        self.policyBytes = UInt64(policyFile.bytes.count)
        self.policyGeneration = fields.policyGeneration
        self.validFromUnixSeconds = fields.validFromUnixSeconds
        self.validUntilUnixSeconds = fields.validUntilUnixSeconds
        self.executableSHA256 = fields.executableSHA256
        self.gitRuntimeClosureManifestSHA256 =
            fields.gitRuntimeClosureManifestSHA256
        self.gitFileImageRuntimeClosureContentEvidenceID =
            fields.gitFileImageRuntimeClosureContentEvidenceID
        self.runtimePolicySHA256 = fields.runtimePolicySHA256
        self.fields = fields
    }
}

/// One already verified signed claim bound to one exact anchored v2 policy.
/// This reference has no ID and grants no operational authority.
struct SignedClaimGitToolPolicyV2Reference: Equatable, Sendable {
    let signedClaim: OperatorSignedRunClaim
    let policyDocument: AnchoredGitToolPolicyV2Document

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
        signedClaim: OperatorSignedRunClaim,
        policyDocument: AnchoredGitToolPolicyV2Document
    ) {
        self.signedClaim = signedClaim
        self.policyDocument = policyDocument
    }
}

enum GitToolPolicyV2Verifier {
    static let policyDomain =
        "fast-mlx-proof-control-git-tool-policy-v2"
    static let policySubject =
        "absorbed-mla-source-import-git"
    static let claimSubject =
        OperatorRunClaimSubject.absorbedMLALoadedResultPair.rawValue
    static let maximumPolicyBytes = 2_199
    static let maximumLineBytes = 115

    static func policyBytes(
        fields: GitToolPolicyV2Fields
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
            "git_runtime_closure_manifest_sha256=\(fields.gitRuntimeClosureManifestSHA256)",
            "git_runtime_closure_manifest_bytes=\(fields.gitRuntimeClosureManifestBytes)",
            "git_file_image_runtime_closure_content_evidence_id=\(fields.gitFileImageRuntimeClosureContentEvidenceID)",
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
        let bytes = Data((lines.joined(separator: "\n") + "\n").utf8)
        guard bytes.count <= maximumPolicyBytes else {
            throw GitToolPolicyV2Error.policySize
        }
        return bytes
    }

    static func anchor(
        policyFile: AdmittedFile,
        trustAnchor: GitToolPolicyV2TrustAnchor
    ) throws -> AnchoredGitToolPolicyV2Document {
        try validate(trustAnchor)

        guard policyFile.bytes.count <= maximumPolicyBytes else {
            throw GitToolPolicyV2Error.policySize
        }
        guard policyFile.sha256 ==
            trustAnchor.expectedCurrentPolicySHA256
        else {
            throw GitToolPolicyV2Error.policyDigestMismatch
        }

        let actualBytes = UInt64(policyFile.bytes.count)
        guard actualBytes == trustAnchor.expectedCurrentPolicyBytes else {
            throw GitToolPolicyV2Error.policyByteCountMismatch(
                expected: trustAnchor.expectedCurrentPolicyBytes,
                actual: actualBytes
            )
        }

        let fields = try parsePolicy(policyFile.bytes)
        guard fields.policyGeneration >=
            trustAnchor.minimumPolicyGeneration
        else {
            throw GitToolPolicyV2Error.policyGenerationRollback(
                minimum: trustAnchor.minimumPolicyGeneration,
                actual: fields.policyGeneration
            )
        }
        guard trustAnchor.verificationUnixSeconds >=
            fields.validFromUnixSeconds
        else {
            throw GitToolPolicyV2Error.policyNotYetValid
        }
        guard trustAnchor.verificationUnixSeconds <=
            fields.validUntilUnixSeconds
        else {
            throw GitToolPolicyV2Error.policyExpired
        }

        return AnchoredGitToolPolicyV2Document(
            policyFile: policyFile,
            fields: fields
        )
    }

    static func reference(
        signedClaim: OperatorSignedRunClaim,
        policyDocument: AnchoredGitToolPolicyV2Document
    ) throws -> SignedClaimGitToolPolicyV2Reference {
        guard policyDocument.policySHA256 ==
            signedClaim.fields.toolManifest.sha256
        else {
            throw GitToolPolicyV2ClaimReferenceError
                .toolManifestDigestMismatch
        }
        guard policyDocument.policyBytes ==
            signedClaim.fields.toolManifest.byteCount
        else {
            throw GitToolPolicyV2ClaimReferenceError
                .toolManifestByteCountMismatch
        }
        guard policyDocument.runtimePolicySHA256 ==
            signedClaim.fields.policies.runtimeSHA256
        else {
            throw GitToolPolicyV2ClaimReferenceError
                .runtimePolicyDigestMismatch
        }

        return SignedClaimGitToolPolicyV2Reference(
            signedClaim: signedClaim,
            policyDocument: policyDocument
        )
    }
}

private extension GitToolPolicyV2Verifier {
    struct CheckedByteLineCursor {
        let bytes: Data
        var offset: Int
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
                throw GitToolPolicyV2Error.nonCanonicalPolicy
            }
        }

        mutating func value(prefix: String) throws -> String {
            let line = try nextLine()
            guard line.hasPrefix(prefix) else {
                throw GitToolPolicyV2Error.nonCanonicalPolicy
            }
            return String(line.dropFirst(prefix.utf8.count))
        }

        mutating func canonicalUInt64(
            prefix: String
        ) throws -> UInt64 {
            let value = try value(prefix: prefix)
            guard
                isCanonicalDecimal(value),
                let result = UInt64(value)
            else {
                throw GitToolPolicyV2Error.nonCanonicalPolicy
            }
            return result
        }

        mutating func lowercaseHex(
            prefix: String,
            count: Int
        ) throws -> String {
            let value = try value(prefix: prefix)
            guard isLowercaseHex(value, count: count) else {
                throw GitToolPolicyV2Error.nonCanonicalPolicy
            }
            return value
        }

        mutating func nextLine() throws -> String {
            guard offset < bytes.endIndex else {
                throw GitToolPolicyV2Error.nonCanonicalPolicy
            }
            let start = offset
            var cursor = offset
            while cursor < bytes.endIndex {
                let byte = bytes[cursor]
                if byte == 0x0a {
                    let count = cursor - start
                    guard count > 0, count <= maximumLineBytes else {
                        throw GitToolPolicyV2Error.nonCanonicalPolicy
                    }
                    let lineBytes = bytes[start..<cursor]
                    guard lineBytes.allSatisfy({
                        (0x20...0x7e).contains($0)
                    }) else {
                        throw GitToolPolicyV2Error.nonCanonicalPolicy
                    }
                    offset = cursor + 1
                    lineCount += 1
                    return String(decoding: lineBytes, as: UTF8.self)
                }
                guard cursor - start < maximumLineBytes else {
                    throw GitToolPolicyV2Error.nonCanonicalPolicy
                }
                cursor += 1
            }
            throw GitToolPolicyV2Error.nonCanonicalPolicy
        }
    }

    static func parsePolicy(_ bytes: Data) throws
        -> GitToolPolicyV2Fields
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
        try parser.require("claim_subject=\(claimSubject)")
        let executableSHA256 = try parser.lowercaseHex(
            prefix: "executable_sha256=",
            count: 64
        )
        let executableBytes = try parser.canonicalUInt64(
            prefix: "executable_bytes="
        )
        try parser.require("executable_architecture=arm64")
        let executableMachOUUID = try parser.lowercaseHex(
            prefix: "executable_macho_uuid=",
            count: 32
        )
        let executableCodeDirectorySHA256 = try parser.lowercaseHex(
            prefix: "executable_code_directory_sha256=",
            count: 64
        )
        let gitRuntimeClosureManifestSHA256 = try parser.lowercaseHex(
            prefix: "git_runtime_closure_manifest_sha256=",
            count: 64
        )
        let gitRuntimeClosureManifestBytes = try parser.canonicalUInt64(
            prefix: "git_runtime_closure_manifest_bytes="
        )
        let gitFileImageRuntimeClosureContentEvidenceID =
            try parser.lowercaseHex(
                prefix:
                    "git_file_image_runtime_closure_content_evidence_id=",
                count: 64
            )
        let runtimePolicySHA256 = try parser.lowercaseHex(
            prefix: "runtime_policy_sha256=",
            count: 64
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
            maxPackBytes: try parser.canonicalUInt64(
                prefix: "max_pack_bytes="
            ),
            maxPackObjects: try parser.canonicalUInt64(
                prefix: "max_pack_objects="
            ),
            maxCommitBytes: try parser.canonicalUInt64(
                prefix: "max_commit_bytes="
            ),
            maxSingleInflatedObjectBytes: try parser.canonicalUInt64(
                prefix: "max_single_inflated_object_bytes="
            ),
            maxTotalInflatedBytes: try parser.canonicalUInt64(
                prefix: "max_total_inflated_bytes="
            ),
            maxCompressionRatio: try parser.canonicalUInt64(
                prefix: "max_compression_ratio="
            ),
            maxTreeDepth: try parser.canonicalUInt64(
                prefix: "max_tree_depth="
            ),
            maxTreeCount: try parser.canonicalUInt64(
                prefix: "max_tree_count="
            ),
            maxObjectDatabaseBytes: try parser.canonicalUInt64(
                prefix: "max_odb_bytes="
            ),
            maxStdoutBytes: try parser.canonicalUInt64(
                prefix: "max_stdout_bytes="
            ),
            maxStderrBytes: try parser.canonicalUInt64(
                prefix: "max_stderr_bytes="
            ),
            wallTimeoutMilliseconds: try parser.canonicalUInt64(
                prefix: "wall_timeout_milliseconds="
            ),
            cpuTimeoutSeconds: try parser.canonicalUInt64(
                prefix: "cpu_timeout_seconds="
            ),
            maxAddressSpaceBytes: try parser.canonicalUInt64(
                prefix: "max_address_space_bytes="
            ),
            maxFileSizeBytes: try parser.canonicalUInt64(
                prefix: "max_file_size_bytes="
            ),
            maxOpenFiles: try parser.canonicalUInt64(
                prefix: "max_open_files="
            )
        )
        guard parser.isAtEnd, parser.lineCount == 53 else {
            throw GitToolPolicyV2Error.nonCanonicalPolicy
        }

        let fields = GitToolPolicyV2Fields(
            policyGeneration: policyGeneration,
            validFromUnixSeconds: validFromUnixSeconds,
            validUntilUnixSeconds: validUntilUnixSeconds,
            executableSHA256: executableSHA256,
            executableBytes: executableBytes,
            executableMachOUUID: executableMachOUUID,
            executableCodeDirectorySHA256:
                executableCodeDirectorySHA256,
            gitRuntimeClosureManifestSHA256:
                gitRuntimeClosureManifestSHA256,
            gitRuntimeClosureManifestBytes:
                gitRuntimeClosureManifestBytes,
            gitFileImageRuntimeClosureContentEvidenceID:
                gitFileImageRuntimeClosureContentEvidenceID,
            runtimePolicySHA256: runtimePolicySHA256,
            limits: limits
        )
        try validate(fields)
        return fields
    }

    static func validate(_ fields: GitToolPolicyV2Fields) throws {
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
                fields.gitRuntimeClosureManifestSHA256,
                count: 64
            ),
            fields.gitRuntimeClosureManifestBytes > 0,
            isLowercaseHex(
                fields.gitFileImageRuntimeClosureContentEvidenceID,
                count: 64
            ),
            isLowercaseHex(fields.runtimePolicySHA256, count: 64)
        else {
            throw GitToolPolicyV2Error.nonCanonicalPolicy
        }
        try validate(fields.limits)
    }

    static func validate(_ limits: GitToolPolicyResourceLimits) throws {
        let ceilings = GitToolPolicyVerifier.phase1ResourceCeilings
        let values: [
            (
                GitToolPolicyV2ResourceField,
                maximum: UInt64,
                actual: UInt64
            )
        ] = [
            (.maxPackBytes, ceilings.maxPackBytes, limits.maxPackBytes),
            (.maxPackObjects, ceilings.maxPackObjects, limits.maxPackObjects),
            (.maxCommitBytes, ceilings.maxCommitBytes, limits.maxCommitBytes),
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
            (.maxTreeDepth, ceilings.maxTreeDepth, limits.maxTreeDepth),
            (.maxTreeCount, ceilings.maxTreeCount, limits.maxTreeCount),
            (
                .maxObjectDatabaseBytes,
                ceilings.maxObjectDatabaseBytes,
                limits.maxObjectDatabaseBytes
            ),
            (.maxStdoutBytes, ceilings.maxStdoutBytes, limits.maxStdoutBytes),
            (.maxStderrBytes, ceilings.maxStderrBytes, limits.maxStderrBytes),
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
            (.maxOpenFiles, ceilings.maxOpenFiles, limits.maxOpenFiles),
        ]

        for (field, maximum, actual) in values {
            guard actual > 0 else {
                throw GitToolPolicyV2Error.nonCanonicalPolicy
            }
            guard actual <= maximum else {
                throw GitToolPolicyV2Error
                    .resourceLimitExceedsCompileTimeCeiling(
                        field: field,
                        maximum: maximum,
                        actual: actual
                    )
            }
        }
    }

    static func validate(
        _ trustAnchor: GitToolPolicyV2TrustAnchor
    ) throws {
        guard isLowercaseHex(
            trustAnchor.expectedCurrentPolicySHA256,
            count: 64
        ) else {
            throw GitToolPolicyV2Error.invalidTrustAnchor(
                .expectedCurrentPolicySHA256
            )
        }
        guard trustAnchor.expectedCurrentPolicyBytes > 0 else {
            throw GitToolPolicyV2Error.invalidTrustAnchor(
                .expectedCurrentPolicyBytes
            )
        }
        guard trustAnchor.minimumPolicyGeneration > 0 else {
            throw GitToolPolicyV2Error.invalidTrustAnchor(
                .minimumPolicyGeneration
            )
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
