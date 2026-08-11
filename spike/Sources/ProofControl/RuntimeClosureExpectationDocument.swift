import Foundation

enum RuntimeClosureExpectationArtifactRole:
    String,
    Equatable,
    Sendable
{
    case git
    case selfGuard = "self-guard"
}

struct RuntimeClosureExpectationMemberFields:
    Equatable,
    Sendable
{
    let contentEvidenceID: String
    let storage: SyntheticRuntimeClosureMemberStorage
    let installName: SyntheticRuntimeClosureInstallName
    let decodedInstallName: Data
    let machOUUID: String
    let primaryCodeDirectoryBlobSHA256: String
    let loadCommandsSHA256: String
}

struct RuntimeClosureExpectationEdgeFields:
    Equatable,
    Sendable
{
    let parentContentEvidenceID: String
    let loadCommandOrdinal: UInt64
    let kind: SyntheticRuntimeClosureEdgeKind
    let installName: SyntheticRuntimeClosureInstallName
    let decodedInstallName: Data
    let resolvedContentEvidenceID: String
}

struct RuntimeClosureExpectationFields:
    Equatable,
    Sendable
{
    let evidenceGeneration: UInt64
    let validFromUnixSeconds: UInt64
    let validUntilUnixSeconds: UInt64
    let artifactRole: RuntimeClosureExpectationArtifactRole
    let platformArchitecture: String
    let platformHardwareModel: String
    let platformOSVersion: String
    let platformOSBuild: String
    let resolutionProfile: String
    let environmentProfile: String
    let rootExecutableContentEvidenceID: String
    let dynamicLoaderContentEvidenceID: String
    let members: [RuntimeClosureExpectationMemberFields]
    let edges: [RuntimeClosureExpectationEdgeFields]
}

/// Immutable comparison context supplied outside both source roles.
/// This value is evidence input only and grants no runtime authority.
struct RuntimeClosureExpectationTrustAnchor:
    Equatable,
    Sendable
{
    let expectedCurrentDocumentSHA256: String
    let expectedCurrentDocumentBytes: UInt64
    let minimumEvidenceGeneration: UInt64
    let verificationUnixSeconds: UInt64
    let expectedArtifactRole: RuntimeClosureExpectationArtifactRole
}

enum RuntimeClosureExpectationTrustAnchorField:
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

enum RuntimeClosureExpectationCacheRecordField:
    Equatable,
    Sendable
{
    case suffixBytes
    case suffixBase64URL
    case fileSHA256
    case fileBytes
    case headerUUID
}

enum RuntimeClosureExpectationCacheSetFailure:
    Error,
    Equatable,
    Sendable
{
    case predecessor(SyntheticSharedCacheSetIdentityFailure)
    case identityMismatch
}

enum RuntimeClosureExpectationMemberRecordField:
    Equatable,
    Sendable
{
    case contentEvidenceID
    case storage
    case installNameBytes
    case installNameBase64URL
    case installNameSyntax
    case machOUUID
    case primaryCodeDirectoryBlobSHA256
    case loadCommandsSHA256
}

enum RuntimeClosureExpectationMemberUniquenessKind:
    Equatable,
    Sendable
{
    case contentEvidenceID
    case installName
}

enum RuntimeClosureExpectationPrefixCollisionKind:
    Equatable,
    Sendable
{
    case rootAndDynamicLoader
    case rootAndMember
    case dynamicLoaderAndMember
}

enum RuntimeClosureExpectationEdgeRecordField:
    Equatable,
    Sendable
{
    case parentContentEvidenceID
    case loadCommandOrdinal
    case kind
    case installNameBytes
    case installNameBase64URL
    case installNameSyntax
    case resolvedContentEvidenceID
}

enum RuntimeClosureExpectationError:
    Error,
    Equatable,
    Sendable
{
    case documentSize
    case nonCanonicalDocument
    case invalidTrustAnchor(RuntimeClosureExpectationTrustAnchorField)
    case documentDigestMismatch
    case documentByteCountMismatch(expected: UInt64, actual: UInt64)
    case cacheRecord(
        index: Int,
        field: RuntimeClosureExpectationCacheRecordField
    )
    case cacheSet(RuntimeClosureExpectationCacheSetFailure)
    case memberRecord(
        index: Int,
        field: RuntimeClosureExpectationMemberRecordField
    )
    case memberOrdering(index: Int)
    case memberUniqueness(
        index: Int,
        kind: RuntimeClosureExpectationMemberUniquenessKind
    )
    case sharedCacheMemberIdentityMismatch(index: Int)
    case prefixIdentityCollision(
        RuntimeClosureExpectationPrefixCollisionKind
    )
    case edgeRecord(
        index: Int,
        field: RuntimeClosureExpectationEdgeRecordField
    )
    case edgeOrdering(index: Int)
    case edgeUniqueness(index: Int)
    case edgeParentMissing(index: Int)
    case edgeResolvedMissing(index: Int)
    case edgeInstallNameMismatch(index: Int)
    case generationRollback(minimum: UInt64, actual: UInt64)
    case documentNotYetValid
    case documentExpired
    case artifactRoleMismatch(
        expected: RuntimeClosureExpectationArtifactRole,
        actual: RuntimeClosureExpectationArtifactRole
    )
}

enum RuntimeClosureExpectationRuntimeDecision:
    String,
    Equatable,
    Sendable
{
    case noGo = "no-go"
}

fileprivate enum RuntimeClosureExpectationConstructionSeal:
    Equatable,
    Sendable
{
    case verified
}

/// Exact retained expectation bytes matched to external comparison context.
/// "Anchored" means comparison-only; this value cannot authorize execution.
/// The later six-anchor slice owns the first transcript-like identifier.
struct AnchoredRuntimeClosureExpectationDocument:
    Equatable,
    Sendable
{
    let expectationFile: AdmittedFile
    let documentSHA256: String
    let documentBytes: UInt64
    let fields: RuntimeClosureExpectationFields
    let sharedCacheSetEvidence:
        SyntheticSharedCacheSetIdentityEvidence
    let canonicalGraphSectionRange: Range<Int>
    let trustAnchor: RuntimeClosureExpectationTrustAnchor
    let runtimeDecision = RuntimeClosureExpectationRuntimeDecision.noGo
    fileprivate let constructionSeal:
        RuntimeClosureExpectationConstructionSeal

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
        fields: RuntimeClosureExpectationFields,
        sharedCacheSetEvidence:
            SyntheticSharedCacheSetIdentityEvidence,
        canonicalGraphSectionRange: Range<Int>,
        trustAnchor: RuntimeClosureExpectationTrustAnchor
    ) {
        self.expectationFile = expectationFile
        self.documentSHA256 = expectationFile.sha256
        self.documentBytes = UInt64(expectationFile.bytes.count)
        self.fields = fields
        self.sharedCacheSetEvidence = sharedCacheSetEvidence
        self.canonicalGraphSectionRange =
            canonicalGraphSectionRange
        self.trustAnchor = trustAnchor
        self.constructionSeal = .verified
    }
}

enum RuntimeClosureExpectationVerifier {
    static let documentDomain =
        "fast-mlx-proof-control-runtime-closure-expectation-v1"
    static let documentSubject =
        "absorbed-mla-source-import-runtime-closure-identity"
    static let maximumDocumentBytes = 25_687_913
    static let maximumLineBytes = 5_502
    static let maximumCacheSuffixBytes = 4_096

    static func anchor(
        expectationFile: AdmittedFile,
        trustAnchor: RuntimeClosureExpectationTrustAnchor
    ) throws -> AnchoredRuntimeClosureExpectationDocument {
        try validate(trustAnchor)

        guard expectationFile.bytes.count <= maximumDocumentBytes else {
            throw RuntimeClosureExpectationError.documentSize
        }
        guard expectationFile.sha256 ==
            trustAnchor.expectedCurrentDocumentSHA256
        else {
            throw RuntimeClosureExpectationError
                .documentDigestMismatch
        }

        let actualBytes = UInt64(expectationFile.bytes.count)
        guard actualBytes ==
            trustAnchor.expectedCurrentDocumentBytes
        else {
            throw RuntimeClosureExpectationError
                .documentByteCountMismatch(
                    expected:
                        trustAnchor.expectedCurrentDocumentBytes,
                    actual: actualBytes
                )
        }

        let parsed = try parseDocument(expectationFile.bytes)
        guard parsed.fields.evidenceGeneration >=
            trustAnchor.minimumEvidenceGeneration
        else {
            throw RuntimeClosureExpectationError
                .generationRollback(
                    minimum: trustAnchor.minimumEvidenceGeneration,
                    actual: parsed.fields.evidenceGeneration
                )
        }
        guard trustAnchor.verificationUnixSeconds >=
            parsed.fields.validFromUnixSeconds
        else {
            throw RuntimeClosureExpectationError
                .documentNotYetValid
        }
        guard trustAnchor.verificationUnixSeconds <=
            parsed.fields.validUntilUnixSeconds
        else {
            throw RuntimeClosureExpectationError.documentExpired
        }
        guard parsed.fields.artifactRole ==
            trustAnchor.expectedArtifactRole
        else {
            throw RuntimeClosureExpectationError
                .artifactRoleMismatch(
                    expected: trustAnchor.expectedArtifactRole,
                    actual: parsed.fields.artifactRole
                )
        }

        return AnchoredRuntimeClosureExpectationDocument(
            expectationFile: expectationFile,
            fields: parsed.fields,
            sharedCacheSetEvidence: parsed.cacheSetEvidence,
            canonicalGraphSectionRange: parsed.graphSectionRange,
            trustAnchor: trustAnchor
        )
    }
}

private extension RuntimeClosureExpectationVerifier {
    static let platformArchitecture = "arm64"
    static let platformHardwareModel = "Mac15,14"
    static let platformOSVersion = "26.5.2"
    static let platformOSBuild = "25F84"
    static let resolutionProfile = "absolute-static-graph-v1"
    static let environmentProfile = "no-dyld-environment-v1"
    static let zeroSHA256 = String(repeating: "0", count: 64)
    static let maximumEncodedCacheSuffixBytes =
        (maximumCacheSuffixBytes * 4 + 2) / 3

    struct ParsedDocument {
        let fields: RuntimeClosureExpectationFields
        let cacheSetEvidence:
            SyntheticSharedCacheSetIdentityEvidence
        let graphSectionRange: Range<Int>
    }

    struct EdgePair: Hashable {
        let parentContentEvidenceID: String
        let loadCommandOrdinal: UInt64
    }

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
                throw RuntimeClosureExpectationError
                    .nonCanonicalDocument
            }
        }

        mutating func value(prefix: String) throws -> String {
            let line = try nextLine()
            guard line.hasPrefix(prefix) else {
                throw RuntimeClosureExpectationError
                    .nonCanonicalDocument
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
                throw RuntimeClosureExpectationError
                    .nonCanonicalDocument
            }
            return result
        }

        mutating func lowercaseHex(
            prefix: String,
            count: Int
        ) throws -> String {
            let value = try value(prefix: prefix)
            guard isLowercaseHex(value, count: count) else {
                throw RuntimeClosureExpectationError
                    .nonCanonicalDocument
            }
            return value
        }

        mutating func nextLine() throws -> String {
            guard offset < bytes.endIndex else {
                throw RuntimeClosureExpectationError
                    .nonCanonicalDocument
            }
            let start = offset
            var cursor = offset
            while cursor < bytes.endIndex {
                let byte = bytes[cursor]
                if byte == 0x0a {
                    let count = cursor - start
                    guard
                        count > 0,
                        count <= maximumLineBytes
                    else {
                        throw RuntimeClosureExpectationError
                            .nonCanonicalDocument
                    }
                    let lineBytes = bytes[start..<cursor]
                    guard lineBytes.allSatisfy({
                        (0x20...0x7e).contains($0)
                    }) else {
                        throw RuntimeClosureExpectationError
                            .nonCanonicalDocument
                    }
                    offset = cursor + 1
                    lineCount += 1
                    return String(
                        decoding: lineBytes,
                        as: UTF8.self
                    )
                }
                guard cursor - start < maximumLineBytes else {
                    throw RuntimeClosureExpectationError
                        .nonCanonicalDocument
                }
                cursor += 1
            }
            throw RuntimeClosureExpectationError
                .nonCanonicalDocument
        }
    }

    static func parseDocument(_ bytes: Data) throws
        -> ParsedDocument
    {
        var parser = CheckedByteLineCursor(bytes: bytes)
        try parser.require(documentDomain)
        try parser.require("subject=\(documentSubject)")
        let evidenceGeneration = try parser.canonicalUInt64(
            prefix: "evidence_generation="
        )
        guard evidenceGeneration > 0 else {
            throw RuntimeClosureExpectationError
                .nonCanonicalDocument
        }
        let validFromUnixSeconds = try parser.canonicalUInt64(
            prefix: "valid_from_unix_seconds="
        )
        let validUntilUnixSeconds = try parser.canonicalUInt64(
            prefix: "valid_until_unix_seconds="
        )
        guard validFromUnixSeconds <= validUntilUnixSeconds else {
            throw RuntimeClosureExpectationError
                .nonCanonicalDocument
        }
        let artifactRole = try parseArtifactRole(
            try parser.value(prefix: "artifact_role=")
        )
        try parser.require(
            "platform_architecture=\(platformArchitecture)"
        )
        try parser.require(
            "platform_hardware_model=\(platformHardwareModel)"
        )
        try parser.require(
            "platform_os_version=\(platformOSVersion)"
        )
        try parser.require("platform_os_build=\(platformOSBuild)")
        try parser.require(
            "resolution_profile=\(resolutionProfile)"
        )
        try parser.require(
            "environment_profile=\(environmentProfile)"
        )
        let rootID = try parser.lowercaseHex(
            prefix: "root_executable_content_evidence_id=",
            count: 64
        )
        let dynamicLoaderID = try parser.lowercaseHex(
            prefix: "dynamic_loader_content_evidence_id=",
            count: 64
        )

        let cacheCountValue = try parser.canonicalUInt64(
            prefix: "shared_cache_file_count="
        )
        guard
            (1...64).contains(cacheCountValue),
            let cacheCount = Int(exactly: cacheCountValue)
        else {
            throw RuntimeClosureExpectationError
                .nonCanonicalDocument
        }
        var cacheRecords: [SyntheticSharedCacheFileRecord] = []
        cacheRecords.reserveCapacity(cacheCount)
        for index in 0..<cacheCount {
            cacheRecords.append(
                try parseCacheRecord(index: index, parser: &parser)
            )
        }
        let declaredCacheSetID = try parser.lowercaseHex(
            prefix: "shared_cache_set_id=",
            count: 64
        )
        let cacheSetEvidence: SyntheticSharedCacheSetIdentityEvidence
        do {
            cacheSetEvidence =
                try SyntheticSharedCacheSetIdentityVerifier.derive(
                    records: cacheRecords
                )
        } catch let failure as SyntheticSharedCacheSetIdentityFailure {
            throw RuntimeClosureExpectationError.cacheSet(
                .predecessor(failure)
            )
        }
        guard cacheSetEvidence.sharedCacheSetID.sha256 ==
            declaredCacheSetID
        else {
            throw RuntimeClosureExpectationError
                .cacheSet(.identityMismatch)
        }

        let graphStart = parser.offset
        let memberCountValue = try parser.canonicalUInt64(
            prefix: "member_count="
        )
        guard
            (1...256).contains(memberCountValue),
            let memberCount = Int(exactly: memberCountValue)
        else {
            throw RuntimeClosureExpectationError
                .nonCanonicalDocument
        }
        var members: [RuntimeClosureExpectationMemberFields] = []
        members.reserveCapacity(memberCount)
        var memberRowLengths: [Int] = []
        memberRowLengths.reserveCapacity(memberCount)
        for index in 0..<memberCount {
            let start = parser.offset
            members.append(
                try parseMemberRecord(index: index, parser: &parser)
            )
            memberRowLengths.append(parser.offset - start)
        }
        try validateMembers(
            members,
            rootID: rootID,
            dynamicLoaderID: dynamicLoaderID,
            cacheSetEvidence: cacheSetEvidence
        )

        let edgeCountValue = try parser.canonicalUInt64(
            prefix: "edge_count="
        )
        guard
            (1...4_096).contains(edgeCountValue),
            let edgeCount = Int(exactly: edgeCountValue)
        else {
            throw RuntimeClosureExpectationError
                .nonCanonicalDocument
        }
        var edges: [RuntimeClosureExpectationEdgeFields] = []
        edges.reserveCapacity(edgeCount)
        var edgeRowLengths: [Int] = []
        edgeRowLengths.reserveCapacity(edgeCount)
        for index in 0..<edgeCount {
            let start = parser.offset
            edges.append(
                try parseEdgeRecord(index: index, parser: &parser)
            )
            edgeRowLengths.append(parser.offset - start)
        }
        let graphEnd = parser.offset
        try validateEdges(edges, rootID: rootID, members: members)

        try parser.require(
            "runtime_resolution_outcome=" +
                "unproved-static-comparison-only"
        )
        try parser.require("runtime_authority=none")
        guard parser.isAtEnd else {
            throw RuntimeClosureExpectationError
                .nonCanonicalDocument
        }

        let expectedLineCount = try checkedLineCount(
            cacheCount: cacheCount,
            memberCount: memberCount,
            edgeCount: edgeCount
        )
        guard parser.lineCount == expectedLineCount else {
            throw RuntimeClosureExpectationError
                .nonCanonicalDocument
        }
        let graphRange = graphStart..<graphEnd
        let expectedGraphLength: Int
        do {
            expectedGraphLength = try
                SyntheticRuntimeClosureRecordCollectionVerifier
                .checkedCanonicalSectionLength(
                    memberCount: memberCount,
                    memberRowLengths: memberRowLengths,
                    edgeCount: edgeCount,
                    edgeRowLengths: edgeRowLengths
                )
        } catch {
            throw RuntimeClosureExpectationError
                .nonCanonicalDocument
        }
        guard graphRange.count == expectedGraphLength else {
            throw RuntimeClosureExpectationError
                .nonCanonicalDocument
        }

        let fields = RuntimeClosureExpectationFields(
            evidenceGeneration: evidenceGeneration,
            validFromUnixSeconds: validFromUnixSeconds,
            validUntilUnixSeconds: validUntilUnixSeconds,
            artifactRole: artifactRole,
            platformArchitecture: platformArchitecture,
            platformHardwareModel: platformHardwareModel,
            platformOSVersion: platformOSVersion,
            platformOSBuild: platformOSBuild,
            resolutionProfile: resolutionProfile,
            environmentProfile: environmentProfile,
            rootExecutableContentEvidenceID: rootID,
            dynamicLoaderContentEvidenceID: dynamicLoaderID,
            members: members,
            edges: edges
        )
        return ParsedDocument(
            fields: fields,
            cacheSetEvidence: cacheSetEvidence,
            graphSectionRange: graphRange
        )
    }

    static func parseCacheRecord(
        index: Int,
        parser: inout CheckedByteLineCursor
    ) throws -> SyntheticSharedCacheFileRecord {
        let record = fourDigit(index)
        let prefix = "shared_cache_file_\(record)_"
        let suffixBytes = try cacheUInt64(
            parser: &parser,
            prefix: prefix + "suffix_bytes=",
            index: index,
            field: .suffixBytes
        )
        guard suffixBytes <= UInt64(maximumCacheSuffixBytes) else {
            throw RuntimeClosureExpectationError.cacheRecord(
                index: index,
                field: .suffixBytes
            )
        }
        let suffixBase64URL = try parser.value(
            prefix: prefix + "suffix_base64url="
        )
        guard
            suffixBase64URL.utf8.count <=
                maximumEncodedCacheSuffixBytes,
            let decodedSuffix = decodeCanonicalBase64URL(
                suffixBase64URL
            )
        else {
            throw RuntimeClosureExpectationError.cacheRecord(
                index: index,
                field: .suffixBase64URL
            )
        }
        guard UInt64(decodedSuffix.count) == suffixBytes else {
            throw RuntimeClosureExpectationError.cacheRecord(
                index: index,
                field: .suffixBytes
            )
        }
        let fileSHA256 = try parser.value(
            prefix: prefix + "sha256="
        )
        guard isLowercaseHex(fileSHA256, count: 64) else {
            throw RuntimeClosureExpectationError.cacheRecord(
                index: index,
                field: .fileSHA256
            )
        }
        let fileBytes = try cacheUInt64(
            parser: &parser,
            prefix: prefix + "bytes=",
            index: index,
            field: .fileBytes
        )
        guard fileBytes > 0 else {
            throw RuntimeClosureExpectationError.cacheRecord(
                index: index,
                field: .fileBytes
            )
        }
        let headerUUID = try parser.value(
            prefix: prefix + "header_uuid="
        )
        guard isLowercaseHex(headerUUID, count: 32) else {
            throw RuntimeClosureExpectationError.cacheRecord(
                index: index,
                field: .headerUUID
            )
        }
        return SyntheticSharedCacheFileRecord(
            suffixBytes: suffixBytes,
            suffixBase64URL: suffixBase64URL,
            fileSHA256: fileSHA256,
            fileBytes: fileBytes,
            headerUUID: headerUUID
        )
    }

    static func parseMemberRecord(
        index: Int,
        parser: inout CheckedByteLineCursor
    ) throws -> RuntimeClosureExpectationMemberFields {
        let record = fourDigit(index)
        let prefix = "member_\(record)_"
        let contentEvidenceID = try memberHex(
            parser: &parser,
            prefix: prefix + "content_evidence_id=",
            count: 64,
            index: index,
            field: .contentEvidenceID
        )
        let storageText = try parser.value(
            prefix: prefix + "storage="
        )
        guard let storage = SyntheticRuntimeClosureMemberStorage(
            rawValue: storageText
        ) else {
            throw RuntimeClosureExpectationError.memberRecord(
                index: index,
                field: .storage
            )
        }
        let installNameBytes = try memberUInt64(
            parser: &parser,
            prefix: prefix + "install_name_bytes=",
            index: index,
            field: .installNameBytes
        )
        let installNameBase64URL = try parser.value(
            prefix: prefix + "install_name_base64url="
        )
        let installName = SyntheticRuntimeClosureInstallName(
            bytes: installNameBytes,
            base64URL: installNameBase64URL
        )
        let decodedInstallName: Data
        do {
            decodedInstallName =
                try SyntheticRuntimeClosureInstallNameVerifier
                .validate(installName)
        } catch let failure as SyntheticRuntimeClosureInstallNameFailure {
            throw RuntimeClosureExpectationError.memberRecord(
                index: index,
                field: memberField(failure)
            )
        }
        let machOUUID = try memberHex(
            parser: &parser,
            prefix: prefix + "macho_uuid=",
            count: 32,
            index: index,
            field: .machOUUID
        )
        let codeDirectorySHA256 = try memberHex(
            parser: &parser,
            prefix: prefix +
                "primary_code_directory_blob_sha256=",
            count: 64,
            index: index,
            field: .primaryCodeDirectoryBlobSHA256
        )
        if storage == .file, codeDirectorySHA256 == zeroSHA256 {
            throw RuntimeClosureExpectationError.memberRecord(
                index: index,
                field: .primaryCodeDirectoryBlobSHA256
            )
        }
        let loadCommandsSHA256 = try memberHex(
            parser: &parser,
            prefix: prefix + "load_commands_sha256=",
            count: 64,
            index: index,
            field: .loadCommandsSHA256
        )
        return RuntimeClosureExpectationMemberFields(
            contentEvidenceID: contentEvidenceID,
            storage: storage,
            installName: installName,
            decodedInstallName: decodedInstallName,
            machOUUID: machOUUID,
            primaryCodeDirectoryBlobSHA256:
                codeDirectorySHA256,
            loadCommandsSHA256: loadCommandsSHA256
        )
    }

    static func parseEdgeRecord(
        index: Int,
        parser: inout CheckedByteLineCursor
    ) throws -> RuntimeClosureExpectationEdgeFields {
        let record = fourDigit(index)
        let prefix = "edge_\(record)_"
        let parentID = try edgeHex(
            parser: &parser,
            prefix: prefix + "parent_content_evidence_id=",
            count: 64,
            index: index,
            field: .parentContentEvidenceID
        )
        let loadCommandOrdinal = try edgeUInt64(
            parser: &parser,
            prefix: prefix + "load_command_ordinal=",
            index: index,
            field: .loadCommandOrdinal
        )
        let kindText = try parser.value(prefix: prefix + "kind=")
        guard let kind = SyntheticRuntimeClosureEdgeKind(
            rawValue: kindText
        ) else {
            throw RuntimeClosureExpectationError.edgeRecord(
                index: index,
                field: .kind
            )
        }
        let installNameBytes = try edgeUInt64(
            parser: &parser,
            prefix: prefix + "install_name_bytes=",
            index: index,
            field: .installNameBytes
        )
        let installNameBase64URL = try parser.value(
            prefix: prefix + "install_name_base64url="
        )
        let installName = SyntheticRuntimeClosureInstallName(
            bytes: installNameBytes,
            base64URL: installNameBase64URL
        )
        let decodedInstallName: Data
        do {
            decodedInstallName =
                try SyntheticRuntimeClosureInstallNameVerifier
                .validate(installName)
        } catch let failure as SyntheticRuntimeClosureInstallNameFailure {
            throw RuntimeClosureExpectationError.edgeRecord(
                index: index,
                field: edgeField(failure)
            )
        }
        let resolvedID = try edgeHex(
            parser: &parser,
            prefix: prefix + "resolved_content_evidence_id=",
            count: 64,
            index: index,
            field: .resolvedContentEvidenceID
        )
        return RuntimeClosureExpectationEdgeFields(
            parentContentEvidenceID: parentID,
            loadCommandOrdinal: loadCommandOrdinal,
            kind: kind,
            installName: installName,
            decodedInstallName: decodedInstallName,
            resolvedContentEvidenceID: resolvedID
        )
    }

    static func validateMembers(
        _ members: [RuntimeClosureExpectationMemberFields],
        rootID: String,
        dynamicLoaderID: String,
        cacheSetEvidence: SyntheticSharedCacheSetIdentityEvidence
    ) throws {
        var memberIDs = Set<String>()
        var installNames = Set<Data>()
        var previousID: String?
        for (index, member) in members.enumerated() {
            guard memberIDs.insert(member.contentEvidenceID).inserted else {
                throw RuntimeClosureExpectationError.memberUniqueness(
                    index: index,
                    kind: .contentEvidenceID
                )
            }
            if let previousID,
               !asciiLess(previousID, member.contentEvidenceID)
            {
                throw RuntimeClosureExpectationError
                    .memberOrdering(index: index)
            }
            guard installNames.insert(member.decodedInstallName).inserted
            else {
                throw RuntimeClosureExpectationError.memberUniqueness(
                    index: index,
                    kind: .installName
                )
            }
            previousID = member.contentEvidenceID
        }

        for (index, member) in members.enumerated()
            where member.storage == .sharedCache
        {
            let primaryCodeDirectory:
                SyntheticSharedCacheImagePrimaryCodeDirectory
            if member.primaryCodeDirectoryBlobSHA256 == zeroSHA256 {
                primaryCodeDirectory = .absent
            } else {
                primaryCodeDirectory = .present(
                    blobSHA256:
                        member.primaryCodeDirectoryBlobSHA256
                )
            }
            let facts = SyntheticSharedCacheImageContentFacts(
                installNameBytes: member.installName.bytes,
                installNameBase64URL: member.installName.base64URL,
                machOUUID: member.machOUUID,
                primaryCodeDirectory: primaryCodeDirectory,
                loadCommandsSHA256: member.loadCommandsSHA256
            )
            guard
                let rederived = try?
                    SyntheticSharedCacheImageContentIdentityVerifier
                    .derive(
                        cacheSetEvidence: cacheSetEvidence,
                        facts: facts
                    ),
                rederived.contentEvidenceID.sha256 ==
                    member.contentEvidenceID
            else {
                throw RuntimeClosureExpectationError
                    .sharedCacheMemberIdentityMismatch(index: index)
            }
        }

        guard rootID != dynamicLoaderID else {
            throw RuntimeClosureExpectationError
                .prefixIdentityCollision(.rootAndDynamicLoader)
        }
        for member in members {
            guard member.contentEvidenceID != rootID else {
                throw RuntimeClosureExpectationError
                    .prefixIdentityCollision(.rootAndMember)
            }
            guard member.contentEvidenceID != dynamicLoaderID else {
                throw RuntimeClosureExpectationError
                    .prefixIdentityCollision(.dynamicLoaderAndMember)
            }
        }
    }

    static func validateEdges(
        _ edges: [RuntimeClosureExpectationEdgeFields],
        rootID: String,
        members: [RuntimeClosureExpectationMemberFields]
    ) throws {
        let membersByID = Dictionary(
            uniqueKeysWithValues: members.map {
                ($0.contentEvidenceID, $0)
            }
        )
        var pairs = Set<EdgePair>()
        var previousPair: EdgePair?
        for (index, edge) in edges.enumerated() {
            let pair = EdgePair(
                parentContentEvidenceID:
                    edge.parentContentEvidenceID,
                loadCommandOrdinal: edge.loadCommandOrdinal
            )
            guard pairs.insert(pair).inserted else {
                throw RuntimeClosureExpectationError
                    .edgeUniqueness(index: index)
            }
            if let previousPair, !edgePairLess(previousPair, pair) {
                throw RuntimeClosureExpectationError
                    .edgeOrdering(index: index)
            }
            guard edge.parentContentEvidenceID == rootID ||
                    membersByID[edge.parentContentEvidenceID] != nil
            else {
                throw RuntimeClosureExpectationError
                    .edgeParentMissing(index: index)
            }
            guard let resolved =
                membersByID[edge.resolvedContentEvidenceID]
            else {
                throw RuntimeClosureExpectationError
                    .edgeResolvedMissing(index: index)
            }
            guard edge.decodedInstallName ==
                resolved.decodedInstallName
            else {
                throw RuntimeClosureExpectationError
                    .edgeInstallNameMismatch(index: index)
            }
            previousPair = pair
        }
    }

    static func memberHex(
        parser: inout CheckedByteLineCursor,
        prefix: String,
        count: Int,
        index: Int,
        field: RuntimeClosureExpectationMemberRecordField
    ) throws -> String {
        let value = try parser.value(prefix: prefix)
        guard isLowercaseHex(value, count: count) else {
            throw RuntimeClosureExpectationError.memberRecord(
                index: index,
                field: field
            )
        }
        return value
    }

    static func cacheUInt64(
        parser: inout CheckedByteLineCursor,
        prefix: String,
        index: Int,
        field: RuntimeClosureExpectationCacheRecordField
    ) throws -> UInt64 {
        do {
            return try parser.canonicalUInt64(prefix: prefix)
        } catch RuntimeClosureExpectationError.nonCanonicalDocument {
            throw RuntimeClosureExpectationError.cacheRecord(
                index: index,
                field: field
            )
        }
    }

    static func memberUInt64(
        parser: inout CheckedByteLineCursor,
        prefix: String,
        index: Int,
        field: RuntimeClosureExpectationMemberRecordField
    ) throws -> UInt64 {
        do {
            return try parser.canonicalUInt64(prefix: prefix)
        } catch RuntimeClosureExpectationError.nonCanonicalDocument {
            throw RuntimeClosureExpectationError.memberRecord(
                index: index,
                field: field
            )
        }
    }

    static func edgeUInt64(
        parser: inout CheckedByteLineCursor,
        prefix: String,
        index: Int,
        field: RuntimeClosureExpectationEdgeRecordField
    ) throws -> UInt64 {
        do {
            return try parser.canonicalUInt64(prefix: prefix)
        } catch RuntimeClosureExpectationError.nonCanonicalDocument {
            throw RuntimeClosureExpectationError.edgeRecord(
                index: index,
                field: field
            )
        }
    }

    static func edgeHex(
        parser: inout CheckedByteLineCursor,
        prefix: String,
        count: Int,
        index: Int,
        field: RuntimeClosureExpectationEdgeRecordField
    ) throws -> String {
        let value = try parser.value(prefix: prefix)
        guard isLowercaseHex(value, count: count) else {
            throw RuntimeClosureExpectationError.edgeRecord(
                index: index,
                field: field
            )
        }
        return value
    }

    static func memberField(
        _ failure: SyntheticRuntimeClosureInstallNameFailure
    ) -> RuntimeClosureExpectationMemberRecordField {
        switch failure {
        case .bytes:
            .installNameBytes
        case .base64URL:
            .installNameBase64URL
        case .syntax:
            .installNameSyntax
        }
    }

    static func edgeField(
        _ failure: SyntheticRuntimeClosureInstallNameFailure
    ) -> RuntimeClosureExpectationEdgeRecordField {
        switch failure {
        case .bytes:
            .installNameBytes
        case .base64URL:
            .installNameBase64URL
        case .syntax:
            .installNameSyntax
        }
    }

    static func parseArtifactRole(
        _ value: String
    ) throws -> RuntimeClosureExpectationArtifactRole {
        guard let role = RuntimeClosureExpectationArtifactRole(
            rawValue: value
        ) else {
            throw RuntimeClosureExpectationError
                .nonCanonicalDocument
        }
        return role
    }

    static func validate(
        _ anchor: RuntimeClosureExpectationTrustAnchor
    ) throws {
        guard isLowercaseHex(
            anchor.expectedCurrentDocumentSHA256,
            count: 64
        ) else {
            throw RuntimeClosureExpectationError.invalidTrustAnchor(
                .expectedCurrentDocumentSHA256
            )
        }
        guard anchor.expectedCurrentDocumentBytes > 0 else {
            throw RuntimeClosureExpectationError.invalidTrustAnchor(
                .expectedCurrentDocumentBytes
            )
        }
        guard anchor.minimumEvidenceGeneration > 0 else {
            throw RuntimeClosureExpectationError.invalidTrustAnchor(
                .minimumEvidenceGeneration
            )
        }
    }

    static func checkedLineCount(
        cacheCount: Int,
        memberCount: Int,
        edgeCount: Int
    ) throws -> Int {
        let cacheLines = try checkedMultiply(cacheCount, 5)
        let memberLines = try checkedMultiply(memberCount, 7)
        let edgeLines = try checkedMultiply(edgeCount, 6)
        let first = try checkedAdd(20, cacheLines)
        let second = try checkedAdd(first, memberLines)
        return try checkedAdd(second, edgeLines)
    }

    static func checkedMultiply(_ lhs: Int, _ rhs: Int) throws
        -> Int
    {
        let (value, overflow) = lhs.multipliedReportingOverflow(
            by: rhs
        )
        guard !overflow else {
            throw RuntimeClosureExpectationError
                .nonCanonicalDocument
        }
        return value
    }

    static func checkedAdd(_ lhs: Int, _ rhs: Int) throws -> Int {
        let (value, overflow) = lhs.addingReportingOverflow(rhs)
        guard !overflow else {
            throw RuntimeClosureExpectationError
                .nonCanonicalDocument
        }
        return value
    }

    static func decodeCanonicalBase64URL(
        _ encoded: String
    ) -> Data? {
        let bytes = Array(encoded.utf8)
        guard
            bytes.allSatisfy({
                (0x41...0x5a).contains($0) ||
                    (0x61...0x7a).contains($0) ||
                    (0x30...0x39).contains($0) ||
                    $0 == 0x2d ||
                    $0 == 0x5f
            }),
            bytes.count % 4 != 1
        else {
            return nil
        }
        let standard = encoded
            .replacingOccurrences(of: "-", with: "+")
            .replacingOccurrences(of: "_", with: "/")
        let padded = standard + String(
            repeating: "=",
            count: (4 - standard.utf8.count % 4) % 4
        )
        guard let decoded = Data(base64Encoded: padded) else {
            return nil
        }
        let canonical = decoded.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
        return canonical == encoded ? decoded : nil
    }

    static func isCanonicalDecimal(_ value: String) -> Bool {
        let bytes = Array(value.utf8)
        guard
            !bytes.isEmpty,
            bytes.allSatisfy({ (0x30...0x39).contains($0) })
        else {
            return false
        }
        return bytes.count == 1 || bytes[0] != 0x30
    }

    static func isLowercaseHex(
        _ value: String,
        count: Int
    ) -> Bool {
        let bytes = Array(value.utf8)
        return bytes.count == count && bytes.allSatisfy {
            (0x30...0x39).contains($0) ||
                (0x61...0x66).contains($0)
        }
    }

    static func asciiLess(_ lhs: String, _ rhs: String) -> Bool {
        lhs.utf8.lexicographicallyPrecedes(rhs.utf8)
    }

    static func edgePairLess(_ lhs: EdgePair, _ rhs: EdgePair) -> Bool {
        if lhs.parentContentEvidenceID != rhs.parentContentEvidenceID {
            return asciiLess(
                lhs.parentContentEvidenceID,
                rhs.parentContentEvidenceID
            )
        }
        return lhs.loadCommandOrdinal < rhs.loadCommandOrdinal
    }

    static func fourDigit(_ value: Int) -> String {
        let text = String(value)
        return String(repeating: "0", count: 4 - text.count) + text
    }
}
