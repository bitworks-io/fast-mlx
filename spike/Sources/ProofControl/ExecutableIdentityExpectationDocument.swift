import Foundation

enum ExecutableIdentityArtifactRole:
    String,
    Equatable,
    Sendable
{
    case git
    case selfGuard = "self-guard"
}

struct ExecutableIdentityCodeDirectoryExpectation:
    Equatable,
    Sendable
{
    let slot: UInt64
    let blobSHA256: String
    let blobBytes: UInt64
    // This slice canonicalizes the comparison scalar only. The later
    // synthetic Mach-O parser owns the source-grounded supported-value set.
    let hashType: UInt64
    let flags: String
    let signingIdentifierBytes: UInt64
    let signingIdentifierBase64URL: String
    let teamIdentifierBytes: UInt64
    let teamIdentifierBase64URL: String
}

struct ExecutableIdentityExpectationFields:
    Equatable,
    Sendable
{
    let evidenceGeneration: UInt64
    let validFromUnixSeconds: UInt64
    let validUntilUnixSeconds: UInt64
    let artifactRole: ExecutableIdentityArtifactRole
    let fileSHA256: String
    let fileBytes: UInt64
    let cpuSubtype: String
    let headerFlags: String
    let loadCommandCount: UInt64
    let loadCommandBytes: UInt64
    let loadCommandsSHA256: String
    let machOUUID: String
    let codeSignatureRegionSHA256: String
    let codeSignatureRegionBytes: UInt64
    let codeDirectories: [
        ExecutableIdentityCodeDirectoryExpectation
    ]
    let cmsBlobSHA256: String
    let cmsBlobBytes: UInt64
}

/// Immutable comparison context supplied outside both source roles.
/// This value is evidence input only and grants no runtime authority.
struct ExecutableIdentityExpectationTrustAnchor:
    Equatable,
    Sendable
{
    let expectedCurrentDocumentSHA256: String
    let expectedCurrentDocumentBytes: UInt64
    let minimumEvidenceGeneration: UInt64
    let verificationUnixSeconds: UInt64
    let expectedArtifactRole: ExecutableIdentityArtifactRole
}

enum ExecutableIdentityExpectationTrustAnchorField:
    String,
    Equatable,
    Sendable
{
    case expectedCurrentDocumentSHA256 =
        "expected-current-document-sha256"
    case expectedCurrentDocumentBytes =
        "expected-current-document-bytes"
    case minimumEvidenceGeneration =
        "minimum-evidence-generation"
}

enum ExecutableIdentityExpectationError:
    Error,
    Equatable,
    Sendable
{
    case nonCanonicalDocument
    case invalidTrustAnchor(
        ExecutableIdentityExpectationTrustAnchorField
    )
    case documentDigestMismatch
    case documentByteCountMismatch(expected: UInt64, actual: UInt64)
    case evidenceGenerationRollback(minimum: UInt64, actual: UInt64)
    case documentNotYetValid
    case documentExpired
    case artifactRoleMismatch(
        expected: ExecutableIdentityArtifactRole,
        actual: ExecutableIdentityArtifactRole
    )
}

extension ExecutableIdentityExpectationError:
    CustomStringConvertible
{
    var description: String {
        switch self {
        case .nonCanonicalDocument:
            "Executable identity expectation is not canonical"
        case .invalidTrustAnchor(let field):
            "Executable identity expectation trust anchor " +
                "\(field.rawValue) is not canonical"
        case .documentDigestMismatch:
            "Executable identity expectation does not match its digest anchor"
        case .documentByteCountMismatch(let expected, let actual):
            "Executable identity expectation has \(actual) bytes, " +
                "expected \(expected)"
        case .evidenceGenerationRollback(let minimum, let actual):
            "Executable identity expectation generation \(actual) " +
                "is below \(minimum)"
        case .documentNotYetValid:
            "Executable identity expectation is not yet valid"
        case .documentExpired:
            "Executable identity expectation has expired"
        case .artifactRoleMismatch(let expected, let actual):
            "Executable identity expectation role \(actual.rawValue) " +
                "does not match \(expected.rawValue)"
        }
    }
}

/// Exact retained expectation bytes matched to external comparison context.
/// "Anchored" means comparison-only; this value cannot authorize execution.
/// The later six-anchor slice owns the first transcript ID; this narrow value
/// deliberately retains the exact typed anchor without inventing an ID.
struct AnchoredExecutableIdentityExpectationDocument:
    Equatable,
    Sendable
{
    let expectationFile: AdmittedFile
    let documentSHA256: String
    let documentBytes: UInt64
    let fields: ExecutableIdentityExpectationFields
    let trustAnchor: ExecutableIdentityExpectationTrustAnchor
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
        expectationFile: AdmittedFile,
        fields: ExecutableIdentityExpectationFields,
        trustAnchor: ExecutableIdentityExpectationTrustAnchor
    ) {
        self.expectationFile = expectationFile
        self.documentSHA256 = expectationFile.sha256
        self.documentBytes = UInt64(expectationFile.bytes.count)
        self.fields = fields
        self.trustAnchor = trustAnchor
    }
}

enum ExecutableIdentityExpectationVerifier {
    static let documentDomain =
        "fast-mlx-proof-control-executable-identity-expectation-v1"
    static let documentSubject =
        "absorbed-mla-source-import-executable-identity"
    static let zeroSHA256 = String(repeating: "0", count: 64)

    static func documentBytes(
        fields: ExecutableIdentityExpectationFields
    ) throws -> Data {
        try validate(fields)

        var lines = [
            documentDomain,
            "subject=\(documentSubject)",
            "evidence_generation=\(fields.evidenceGeneration)",
            "valid_from_unix_seconds=\(fields.validFromUnixSeconds)",
            "valid_until_unix_seconds=\(fields.validUntilUnixSeconds)",
            "artifact_role=\(fields.artifactRole.rawValue)",
            "platform_architecture=arm64",
            "platform_os_build=25F84",
            "container_format=thin-macho64-little-endian-v1",
            "file_sha256=\(fields.fileSHA256)",
            "file_bytes=\(fields.fileBytes)",
            "mach_header_magic=feedfacf",
            "cpu_type=0100000c",
            "cpu_subtype=\(fields.cpuSubtype)",
            "file_type=00000002",
            "header_flags=\(fields.headerFlags)",
            "load_command_count=\(fields.loadCommandCount)",
            "load_command_bytes=\(fields.loadCommandBytes)",
            "load_commands_sha256=\(fields.loadCommandsSHA256)",
            "macho_uuid=\(fields.machOUUID)",
            "code_signature_region_sha256=" +
                fields.codeSignatureRegionSHA256,
            "code_signature_region_bytes=" +
                "\(fields.codeSignatureRegionBytes)",
            "code_directory_count=\(fields.codeDirectories.count)",
        ]
        for (index, codeDirectory) in
            fields.codeDirectories.enumerated()
        {
            let record = indexedRecord(index)
            lines.append(
                "code_directory_\(record)_slot=" +
                    "\(codeDirectory.slot)"
            )
            lines.append(
                "code_directory_\(record)_blob_sha256=" +
                    codeDirectory.blobSHA256
            )
            lines.append(
                "code_directory_\(record)_blob_bytes=" +
                    "\(codeDirectory.blobBytes)"
            )
            lines.append(
                "code_directory_\(record)_hash_type=" +
                    "\(codeDirectory.hashType)"
            )
            lines.append(
                "code_directory_\(record)_flags=" +
                    codeDirectory.flags
            )
            lines.append(
                "code_directory_\(record)_signing_identifier_bytes=" +
                    "\(codeDirectory.signingIdentifierBytes)"
            )
            lines.append(
                "code_directory_\(record)_" +
                    "signing_identifier_base64url=" +
                    codeDirectory.signingIdentifierBase64URL
            )
            lines.append(
                "code_directory_\(record)_team_identifier_bytes=" +
                    "\(codeDirectory.teamIdentifierBytes)"
            )
            lines.append(
                "code_directory_\(record)_" +
                    "team_identifier_base64url=" +
                    codeDirectory.teamIdentifierBase64URL
            )
        }
        lines.append("cms_blob_sha256=\(fields.cmsBlobSHA256)")
        lines.append("cms_blob_bytes=\(fields.cmsBlobBytes)")
        lines.append("signer_semantics=metadata-only-no-trust-v1")
        lines.append("runtime_authority=none")
        return Data((lines.joined(separator: "\n") + "\n").utf8)
    }

    static func anchor(
        expectationFile: AdmittedFile,
        trustAnchor: ExecutableIdentityExpectationTrustAnchor
    ) throws -> AnchoredExecutableIdentityExpectationDocument {
        try validate(trustAnchor)
        guard expectationFile.sha256 ==
            trustAnchor.expectedCurrentDocumentSHA256
        else {
            throw ExecutableIdentityExpectationError
                .documentDigestMismatch
        }

        let actualBytes = UInt64(expectationFile.bytes.count)
        guard actualBytes ==
            trustAnchor.expectedCurrentDocumentBytes
        else {
            throw ExecutableIdentityExpectationError
                .documentByteCountMismatch(
                    expected:
                        trustAnchor.expectedCurrentDocumentBytes,
                    actual: actualBytes
                )
        }

        let fields = try parseDocument(expectationFile.bytes)
        guard fields.evidenceGeneration >=
            trustAnchor.minimumEvidenceGeneration
        else {
            throw ExecutableIdentityExpectationError
                .evidenceGenerationRollback(
                    minimum:
                        trustAnchor.minimumEvidenceGeneration,
                    actual: fields.evidenceGeneration
                )
        }
        guard trustAnchor.verificationUnixSeconds >=
            fields.validFromUnixSeconds
        else {
            throw ExecutableIdentityExpectationError
                .documentNotYetValid
        }
        guard trustAnchor.verificationUnixSeconds <=
            fields.validUntilUnixSeconds
        else {
            throw ExecutableIdentityExpectationError.documentExpired
        }
        guard fields.artifactRole ==
            trustAnchor.expectedArtifactRole
        else {
            throw ExecutableIdentityExpectationError
                .artifactRoleMismatch(
                    expected: trustAnchor.expectedArtifactRole,
                    actual: fields.artifactRole
                )
        }

        return AnchoredExecutableIdentityExpectationDocument(
            expectationFile: expectationFile,
            fields: fields,
            trustAnchor: trustAnchor
        )
    }
}

private extension ExecutableIdentityExpectationVerifier {
    struct CanonicalLineParser {
        let lines: [Substring]
        var index = 0

        mutating func require(_ expected: String) throws {
            guard
                index < lines.count,
                lines[index] == Substring(expected)
            else {
                throw ExecutableIdentityExpectationError
                    .nonCanonicalDocument
            }
            index += 1
        }

        mutating func value(prefix: String) throws -> String {
            guard
                index < lines.count,
                lines[index].hasPrefix(prefix)
            else {
                throw ExecutableIdentityExpectationError
                    .nonCanonicalDocument
            }
            let result = String(
                lines[index].dropFirst(prefix.utf8.count)
            )
            index += 1
            return result
        }

        mutating func uint64(prefix: String) throws -> UInt64 {
            let result = try value(prefix: prefix)
            guard
                isCanonicalDecimal(result),
                let value = UInt64(result)
            else {
                throw ExecutableIdentityExpectationError
                    .nonCanonicalDocument
            }
            return value
        }

        mutating func hex(
            prefix: String,
            count: Int
        ) throws -> String {
            let result = try value(prefix: prefix)
            guard isLowercaseHex(result, count: count) else {
                throw ExecutableIdentityExpectationError
                    .nonCanonicalDocument
            }
            return result
        }

        mutating func role(
            prefix: String
        ) throws -> ExecutableIdentityArtifactRole {
            let result = try value(prefix: prefix)
            guard let role = ExecutableIdentityArtifactRole(
                rawValue: result
            ) else {
                throw ExecutableIdentityExpectationError
                    .nonCanonicalDocument
            }
            return role
        }

        var isAtEnd: Bool {
            index == lines.count
        }
    }

    static func parseDocument(_ bytes: Data) throws
        -> ExecutableIdentityExpectationFields
    {
        guard
            let text = String(data: bytes, encoding: .utf8),
            Data(text.utf8) == bytes
        else {
            throw ExecutableIdentityExpectationError
                .nonCanonicalDocument
        }

        let splitLines = text.split(
            separator: "\n",
            omittingEmptySubsequences: false
        )
        guard
            splitLines.last?.isEmpty == true,
            splitLines.count >= 37
        else {
            throw ExecutableIdentityExpectationError
                .nonCanonicalDocument
        }

        var parser = CanonicalLineParser(
            lines: Array(splitLines.dropLast())
        )
        try parser.require(documentDomain)
        try parser.require("subject=\(documentSubject)")
        let evidenceGeneration = try parser.uint64(
            prefix: "evidence_generation="
        )
        let validFromUnixSeconds = try parser.uint64(
            prefix: "valid_from_unix_seconds="
        )
        let validUntilUnixSeconds = try parser.uint64(
            prefix: "valid_until_unix_seconds="
        )
        let artifactRole = try parser.role(
            prefix: "artifact_role="
        )
        try parser.require("platform_architecture=arm64")
        try parser.require("platform_os_build=25F84")
        try parser.require(
            "container_format=thin-macho64-little-endian-v1"
        )
        let fileSHA256 = try parser.hex(
            prefix: "file_sha256=",
            count: 64
        )
        let fileBytes = try parser.uint64(prefix: "file_bytes=")
        try parser.require("mach_header_magic=feedfacf")
        try parser.require("cpu_type=0100000c")
        let cpuSubtype = try parser.hex(
            prefix: "cpu_subtype=",
            count: 8
        )
        try parser.require("file_type=00000002")
        let headerFlags = try parser.hex(
            prefix: "header_flags=",
            count: 8
        )
        let loadCommandCount = try parser.uint64(
            prefix: "load_command_count="
        )
        let loadCommandBytes = try parser.uint64(
            prefix: "load_command_bytes="
        )
        let loadCommandsSHA256 = try parser.hex(
            prefix: "load_commands_sha256=",
            count: 64
        )
        let machOUUID = try parser.hex(
            prefix: "macho_uuid=",
            count: 32
        )
        let codeSignatureRegionSHA256 = try parser.hex(
            prefix: "code_signature_region_sha256=",
            count: 64
        )
        let codeSignatureRegionBytes = try parser.uint64(
            prefix: "code_signature_region_bytes="
        )
        let codeDirectoryCount = try parser.uint64(
            prefix: "code_directory_count="
        )
        guard
            (1...6).contains(codeDirectoryCount),
            let count = Int(exactly: codeDirectoryCount)
        else {
            throw ExecutableIdentityExpectationError
                .nonCanonicalDocument
        }

        var codeDirectories: [
            ExecutableIdentityCodeDirectoryExpectation
        ] = []
        codeDirectories.reserveCapacity(count)
        for index in 0..<count {
            let record = indexedRecord(index)
            codeDirectories.append(
                ExecutableIdentityCodeDirectoryExpectation(
                    slot: try parser.uint64(
                        prefix:
                            "code_directory_\(record)_slot="
                    ),
                    blobSHA256: try parser.hex(
                        prefix:
                            "code_directory_\(record)_blob_sha256=",
                        count: 64
                    ),
                    blobBytes: try parser.uint64(
                        prefix:
                            "code_directory_\(record)_blob_bytes="
                    ),
                    hashType: try parser.uint64(
                        prefix:
                            "code_directory_\(record)_hash_type="
                    ),
                    flags: try parser.hex(
                        prefix:
                            "code_directory_\(record)_flags=",
                        count: 8
                    ),
                    signingIdentifierBytes: try parser.uint64(
                        prefix:
                            "code_directory_\(record)_" +
                            "signing_identifier_bytes="
                    ),
                    signingIdentifierBase64URL: try parser.value(
                        prefix:
                            "code_directory_\(record)_" +
                            "signing_identifier_base64url="
                    ),
                    teamIdentifierBytes: try parser.uint64(
                        prefix:
                            "code_directory_\(record)_" +
                            "team_identifier_bytes="
                    ),
                    teamIdentifierBase64URL: try parser.value(
                        prefix:
                            "code_directory_\(record)_" +
                            "team_identifier_base64url="
                    )
                )
            )
        }
        let cmsBlobSHA256 = try parser.hex(
            prefix: "cms_blob_sha256=",
            count: 64
        )
        let cmsBlobBytes = try parser.uint64(
            prefix: "cms_blob_bytes="
        )
        try parser.require(
            "signer_semantics=metadata-only-no-trust-v1"
        )
        try parser.require("runtime_authority=none")
        guard parser.isAtEnd else {
            throw ExecutableIdentityExpectationError
                .nonCanonicalDocument
        }

        let fields = ExecutableIdentityExpectationFields(
            evidenceGeneration: evidenceGeneration,
            validFromUnixSeconds: validFromUnixSeconds,
            validUntilUnixSeconds: validUntilUnixSeconds,
            artifactRole: artifactRole,
            fileSHA256: fileSHA256,
            fileBytes: fileBytes,
            cpuSubtype: cpuSubtype,
            headerFlags: headerFlags,
            loadCommandCount: loadCommandCount,
            loadCommandBytes: loadCommandBytes,
            loadCommandsSHA256: loadCommandsSHA256,
            machOUUID: machOUUID,
            codeSignatureRegionSHA256:
                codeSignatureRegionSHA256,
            codeSignatureRegionBytes:
                codeSignatureRegionBytes,
            codeDirectories: codeDirectories,
            cmsBlobSHA256: cmsBlobSHA256,
            cmsBlobBytes: cmsBlobBytes
        )
        try validate(fields)
        guard try documentBytes(fields: fields) == bytes else {
            throw ExecutableIdentityExpectationError
                .nonCanonicalDocument
        }
        return fields
    }

    static func validate(
        _ fields: ExecutableIdentityExpectationFields
    ) throws {
        guard
            fields.evidenceGeneration > 0,
            fields.validFromUnixSeconds <=
                fields.validUntilUnixSeconds,
            isLowercaseHex(fields.fileSHA256, count: 64),
            fields.fileBytes > 0,
            isLowercaseHex(fields.cpuSubtype, count: 8),
            isLowercaseHex(fields.headerFlags, count: 8),
            fields.loadCommandCount > 0,
            fields.loadCommandBytes > 0,
            isLowercaseHex(
                fields.loadCommandsSHA256,
                count: 64
            ),
            isLowercaseHex(fields.machOUUID, count: 32),
            fields.machOUUID != String(repeating: "0", count: 32),
            isLowercaseHex(
                fields.codeSignatureRegionSHA256,
                count: 64
            ),
            fields.codeSignatureRegionBytes > 0,
            (1...6).contains(fields.codeDirectories.count),
            isLowercaseHex(fields.cmsBlobSHA256, count: 64),
            (fields.cmsBlobSHA256 == zeroSHA256) ==
                (fields.cmsBlobBytes == 0)
        else {
            throw ExecutableIdentityExpectationError
                .nonCanonicalDocument
        }

        for (index, codeDirectory) in
            fields.codeDirectories.enumerated()
        {
            let validSlot: Bool
            if index == 0 {
                validSlot = codeDirectory.slot == 0
            } else {
                let previous = fields.codeDirectories[index - 1].slot
                validSlot =
                    (4_096...4_100).contains(codeDirectory.slot) &&
                    codeDirectory.slot > previous
            }
            guard
                validSlot,
                isLowercaseHex(
                    codeDirectory.blobSHA256,
                    count: 64
                ),
                codeDirectory.blobBytes > 0,
                isLowercaseHex(codeDirectory.flags, count: 8),
                isCanonicalBase64URL(
                    codeDirectory.signingIdentifierBase64URL,
                    decodedBytes:
                        codeDirectory.signingIdentifierBytes
                ),
                isCanonicalBase64URL(
                    codeDirectory.teamIdentifierBase64URL,
                    decodedBytes:
                        codeDirectory.teamIdentifierBytes
                )
            else {
                throw ExecutableIdentityExpectationError
                    .nonCanonicalDocument
            }
        }
    }

    static func validate(
        _ trustAnchor: ExecutableIdentityExpectationTrustAnchor
    ) throws {
        guard isLowercaseHex(
            trustAnchor.expectedCurrentDocumentSHA256,
            count: 64
        ) else {
            throw ExecutableIdentityExpectationError
                .invalidTrustAnchor(
                    .expectedCurrentDocumentSHA256
                )
        }
        guard trustAnchor.expectedCurrentDocumentBytes > 0 else {
            throw ExecutableIdentityExpectationError
                .invalidTrustAnchor(
                    .expectedCurrentDocumentBytes
                )
        }
        guard trustAnchor.minimumEvidenceGeneration > 0 else {
            throw ExecutableIdentityExpectationError
                .invalidTrustAnchor(
                    .minimumEvidenceGeneration
                )
        }
    }

    static func indexedRecord(_ index: Int) -> String {
        let value = String(index)
        return String(repeating: "0", count: 4 - value.count) +
            value
    }

    static func isCanonicalDecimal(_ value: String) -> Bool {
        guard !value.isEmpty else { return false }
        guard value == "0" || value.first != "0" else {
            return false
        }
        return value.utf8.allSatisfy { (48...57).contains($0) }
    }

    static func isLowercaseHex(
        _ value: String,
        count: Int
    ) -> Bool {
        value.utf8.count == count &&
            value.utf8.allSatisfy {
                (48...57).contains($0) || (97...102).contains($0)
            }
    }

    static func isCanonicalBase64URL(
        _ value: String,
        decodedBytes: UInt64
    ) -> Bool {
        guard
            value.utf8.allSatisfy({
                (65...90).contains($0) ||
                    (97...122).contains($0) ||
                    (48...57).contains($0) ||
                    $0 == 45 ||
                    $0 == 95
            }),
            value.utf8.count % 4 != 1
        else {
            return false
        }

        var standard = value
            .replacingOccurrences(of: "-", with: "+")
            .replacingOccurrences(of: "_", with: "/")
        standard += String(
            repeating: "=",
            count: (4 - standard.utf8.count % 4) % 4
        )
        guard
            let decoded = Data(base64Encoded: standard),
            UInt64(decoded.count) == decodedBytes
        else {
            return false
        }

        let reencoded = decoded.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
        return reencoded == value
    }
}
