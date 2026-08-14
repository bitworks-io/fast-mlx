import Foundation

fileprivate enum SyntheticRuntimeClosureRecordConstructionSeal:
    Equatable
{
    case verified
}

private extension SyntheticRuntimeClosureInstallNameFailure {
    var recordField: SyntheticRuntimeClosureRecordField {
        switch self {
        case .bytes:
            .installNameBytes
        case .base64URL:
            .installNameBase64URL
        case .syntax:
            .installNameSyntax
        }
    }
}

enum SyntheticRuntimeClosureMemberStorage:
    String,
    Equatable,
    Sendable
{
    case file
    case sharedCache = "shared-cache"
}

enum SyntheticRuntimeClosureMemberSource: Equatable {
    case file(ExecutableContentIdentityEvidence)
    case fileImage(
        SyntheticFileImageRuntimeClosureMemberSnapshot
    )
    case sharedCache(
        SyntheticSharedCacheImageContentIdentityEvidence
    )
}

enum SyntheticRuntimeClosureEdgeKind:
    String,
    Equatable,
    Sendable
{
    case load
    case reexport
}

enum SyntheticRuntimeClosureEdgeParent: Equatable {
    case root(ExecutableContentIdentityEvidence)
    case member(SyntheticRuntimeClosureMemberRecordComparison)
}

enum SyntheticRuntimeClosureRecordField:
    Equatable,
    Sendable
{
    case installNameBytes
    case installNameBase64URL
    case installNameSyntax
    case primaryCodeDirectoryBlobSHA256
}

enum SyntheticRuntimeClosureRecordSchemaFailure:
    Error,
    Equatable
{
    case memberIndexOutOfRange(Int)
    case edgeIndexOutOfRange(Int)
    case unsupportedFileMemberRole(
        ExecutableContentArtifactRole
    )
    case unsupportedRootRole(
        ExecutableContentArtifactRole
    )
    case memberEvidenceMismatch
    case fileImageSnapshot(
        SyntheticFileImageRuntimeClosureMemberSnapshotFailure
    )
    case fileImageInstallNameMismatch
    case sharedCacheInstallNameMismatch
    case invalidMember(
        index: Int,
        field: SyntheticRuntimeClosureRecordField
    )
    case edgeParentEvidenceMismatch
    case edgeResolvedEvidenceMismatch
    case edgeInstallNameMismatch
    case invalidEdge(
        index: Int,
        field: SyntheticRuntimeClosureRecordField
    )
}

/// One exact indexed member-row comparison only. The install name is an
/// inert dependency label, not a filesystem locator. Collection ordering,
/// uniqueness, resolution, and reachability remain deliberately absent.
struct SyntheticRuntimeClosureMemberRecordComparison: Equatable {
    let index: Int
    let source: SyntheticRuntimeClosureMemberSource
    let storage: SyntheticRuntimeClosureMemberStorage
    let installName: SyntheticRuntimeClosureInstallName
    let decodedInstallName: Data
    let contentEvidenceID: String
    let machOUUID: String
    let primaryCodeDirectoryBlobSHA256: String
    let loadCommandsSHA256: String
    let canonicalRecordBytes: Data
    fileprivate let constructionSeal:
        SyntheticRuntimeClosureRecordConstructionSeal

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
        index: Int,
        source: SyntheticRuntimeClosureMemberSource,
        storage: SyntheticRuntimeClosureMemberStorage,
        installName: SyntheticRuntimeClosureInstallName,
        decodedInstallName: Data,
        contentEvidenceID: String,
        machOUUID: String,
        primaryCodeDirectoryBlobSHA256: String,
        loadCommandsSHA256: String,
        canonicalRecordBytes: Data
    ) {
        self.index = index
        self.source = source
        self.storage = storage
        self.installName = installName
        self.decodedInstallName = decodedInstallName
        self.contentEvidenceID = contentEvidenceID
        self.machOUUID = machOUUID
        self.primaryCodeDirectoryBlobSHA256 =
            primaryCodeDirectoryBlobSHA256
        self.loadCommandsSHA256 = loadCommandsSHA256
        self.canonicalRecordBytes = canonicalRecordBytes
        self.constructionSeal = .verified
    }
}

/// One exact indexed edge-row comparison only. It proves record-local typed
/// continuity and exact label equality with the resolved member; it does not
/// prove graph membership, uniqueness, traversal, cycles, or reachability.
struct SyntheticRuntimeClosureEdgeRecordComparison: Equatable {
    let index: Int
    let parent: SyntheticRuntimeClosureEdgeParent
    let resolved: SyntheticRuntimeClosureMemberRecordComparison
    let parentContentEvidenceID: String
    let loadCommandOrdinal: UInt64
    let kind: SyntheticRuntimeClosureEdgeKind
    let installName: SyntheticRuntimeClosureInstallName
    let decodedInstallName: Data
    let resolvedContentEvidenceID: String
    let canonicalRecordBytes: Data
    fileprivate let constructionSeal:
        SyntheticRuntimeClosureRecordConstructionSeal

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
        index: Int,
        parent: SyntheticRuntimeClosureEdgeParent,
        resolved: SyntheticRuntimeClosureMemberRecordComparison,
        parentContentEvidenceID: String,
        loadCommandOrdinal: UInt64,
        kind: SyntheticRuntimeClosureEdgeKind,
        installName: SyntheticRuntimeClosureInstallName,
        decodedInstallName: Data,
        canonicalRecordBytes: Data
    ) {
        self.index = index
        self.parent = parent
        self.resolved = resolved
        self.parentContentEvidenceID = parentContentEvidenceID
        self.loadCommandOrdinal = loadCommandOrdinal
        self.kind = kind
        self.installName = installName
        self.decodedInstallName = decodedInstallName
        self.resolvedContentEvidenceID =
            resolved.contentEvidenceID
        self.canonicalRecordBytes = canonicalRecordBytes
        self.constructionSeal = .verified
    }
}

enum SyntheticRuntimeClosureRecordSchemaVerifier {
    static let maximumMemberCount = 256
    static let maximumEdgeCount = 4_096

    static func member(
        index: Int,
        source: SyntheticRuntimeClosureMemberSource,
        installName: SyntheticRuntimeClosureInstallName
    ) throws -> SyntheticRuntimeClosureMemberRecordComparison {
        guard (0..<maximumMemberCount).contains(index) else {
            throw SyntheticRuntimeClosureRecordSchemaFailure
                .memberIndexOutOfRange(index)
        }

        let decodedInstallName: Data
        do {
            decodedInstallName =
                try SyntheticRuntimeClosureInstallNameVerifier
                .validate(
                    installName
            )
        } catch let failure as
            SyntheticRuntimeClosureInstallNameFailure {
            throw SyntheticRuntimeClosureRecordSchemaFailure
                .invalidMember(
                    index: index,
                    field: failure.recordField
                )
        }

        let storage: SyntheticRuntimeClosureMemberStorage
        let contentEvidenceID: String
        let machOUUID: String
        let primaryCodeDirectoryBlobSHA256: String
        let loadCommandsSHA256: String

        switch source {
        case let .file(evidence):
            guard evidence.artifactRole == .fileImage else {
                throw SyntheticRuntimeClosureRecordSchemaFailure
                    .unsupportedFileMemberRole(
                        evidence.artifactRole
                    )
            }
            guard rederive(evidence) == evidence else {
                throw SyntheticRuntimeClosureRecordSchemaFailure
                    .memberEvidenceMismatch
            }
            guard
                evidence.primaryCodeDirectoryBlobSHA256 !=
                    absentCodeDirectorySHA256
            else {
                throw SyntheticRuntimeClosureRecordSchemaFailure
                    .invalidMember(
                        index: index,
                        field:
                            .primaryCodeDirectoryBlobSHA256
                    )
            }
            storage = .file
            contentEvidenceID =
                evidence.contentEvidenceID.sha256
            machOUUID = hex(evidence.comparison.machOUUID)
            primaryCodeDirectoryBlobSHA256 =
                evidence.primaryCodeDirectoryBlobSHA256
            loadCommandsSHA256 =
                evidence.comparison.loadCommandsSHA256

        case let .fileImage(snapshot):
            let rederived:
                SyntheticFileImageRuntimeClosureMemberSnapshot
            do {
                rederived = try
                    SyntheticFileImageRuntimeClosureMemberSnapshotVerifier
                    .derive(
                        fileImageEvidence: snapshot.fileImageEvidence
                    )
            } catch let failure as
                SyntheticFileImageRuntimeClosureMemberSnapshotFailure
            {
                throw SyntheticRuntimeClosureRecordSchemaFailure
                    .fileImageSnapshot(failure)
            }
            guard rederived == snapshot else {
                throw SyntheticRuntimeClosureRecordSchemaFailure
                    .memberEvidenceMismatch
            }
            guard
                installName == snapshot.installName,
                decodedInstallName == snapshot.decodedInstallName
            else {
                throw SyntheticRuntimeClosureRecordSchemaFailure
                    .fileImageInstallNameMismatch
            }
            storage = .file
            contentEvidenceID =
                snapshot.fileImageEvidence.contentEvidenceID.sha256
            machOUUID = hex(
                snapshot.fileImageEvidence.comparison.machOUUID
            )
            primaryCodeDirectoryBlobSHA256 =
                snapshot.fileImageEvidence
                    .primaryCodeDirectoryBlobSHA256
            loadCommandsSHA256 =
                snapshot.fileImageEvidence.comparison
                    .loadCommandsSHA256

        case let .sharedCache(evidence):
            guard rederive(evidence) == evidence else {
                throw SyntheticRuntimeClosureRecordSchemaFailure
                    .memberEvidenceMismatch
            }
            guard
                evidence.facts.installNameBytes ==
                    installName.bytes,
                evidence.facts.installNameBase64URL ==
                    installName.base64URL,
                evidence.decodedInstallName ==
                    decodedInstallName
            else {
                throw SyntheticRuntimeClosureRecordSchemaFailure
                    .sharedCacheInstallNameMismatch
            }
            storage = .sharedCache
            contentEvidenceID =
                evidence.contentEvidenceID.sha256
            machOUUID = evidence.facts.machOUUID
            primaryCodeDirectoryBlobSHA256 =
                evidence.primaryCodeDirectoryBlobSHA256
            loadCommandsSHA256 =
                evidence.facts.loadCommandsSHA256
        }

        let bytes = memberRecordBytes(
            index: index,
            contentEvidenceID: contentEvidenceID,
            storage: storage,
            installName: installName,
            machOUUID: machOUUID,
            primaryCodeDirectoryBlobSHA256:
                primaryCodeDirectoryBlobSHA256,
            loadCommandsSHA256: loadCommandsSHA256
        )
        return SyntheticRuntimeClosureMemberRecordComparison(
            index: index,
            source: source,
            storage: storage,
            installName: installName,
            decodedInstallName: decodedInstallName,
            contentEvidenceID: contentEvidenceID,
            machOUUID: machOUUID,
            primaryCodeDirectoryBlobSHA256:
                primaryCodeDirectoryBlobSHA256,
            loadCommandsSHA256: loadCommandsSHA256,
            canonicalRecordBytes: bytes
        )
    }

    static func edge(
        index: Int,
        parent: SyntheticRuntimeClosureEdgeParent,
        loadCommandOrdinal: UInt64,
        kind: SyntheticRuntimeClosureEdgeKind,
        installName: SyntheticRuntimeClosureInstallName,
        resolved: SyntheticRuntimeClosureMemberRecordComparison
    ) throws -> SyntheticRuntimeClosureEdgeRecordComparison {
        guard (0..<maximumEdgeCount).contains(index) else {
            throw SyntheticRuntimeClosureRecordSchemaFailure
                .edgeIndexOutOfRange(index)
        }

        let decodedInstallName: Data
        do {
            decodedInstallName =
                try SyntheticRuntimeClosureInstallNameVerifier
                .validate(
                    installName
            )
        } catch let failure as
            SyntheticRuntimeClosureInstallNameFailure {
            throw SyntheticRuntimeClosureRecordSchemaFailure
                .invalidEdge(
                    index: index,
                    field: failure.recordField
                )
        }

        let parentContentEvidenceID: String
        switch parent {
        case let .root(evidence):
            guard evidence.artifactRole == .git ||
                    evidence.artifactRole == .selfGuard
            else {
                throw SyntheticRuntimeClosureRecordSchemaFailure
                    .unsupportedRootRole(
                        evidence.artifactRole
                    )
            }
            guard rederive(evidence) == evidence else {
                throw SyntheticRuntimeClosureRecordSchemaFailure
                    .edgeParentEvidenceMismatch
            }
            parentContentEvidenceID =
                evidence.contentEvidenceID.sha256
        case let .member(record):
            guard rederive(record) == record else {
                throw SyntheticRuntimeClosureRecordSchemaFailure
                    .edgeParentEvidenceMismatch
            }
            parentContentEvidenceID =
                record.contentEvidenceID
        }

        guard rederive(resolved) == resolved else {
            throw SyntheticRuntimeClosureRecordSchemaFailure
                .edgeResolvedEvidenceMismatch
        }
        guard
            installName == resolved.installName,
            decodedInstallName == resolved.decodedInstallName
        else {
            throw SyntheticRuntimeClosureRecordSchemaFailure
                .edgeInstallNameMismatch
        }

        let bytes = edgeRecordBytes(
            index: index,
            parentContentEvidenceID:
                parentContentEvidenceID,
            loadCommandOrdinal: loadCommandOrdinal,
            kind: kind,
            installName: installName,
            resolvedContentEvidenceID:
                resolved.contentEvidenceID
        )
        return SyntheticRuntimeClosureEdgeRecordComparison(
            index: index,
            parent: parent,
            resolved: resolved,
            parentContentEvidenceID:
                parentContentEvidenceID,
            loadCommandOrdinal: loadCommandOrdinal,
            kind: kind,
            installName: installName,
            decodedInstallName: decodedInstallName,
            canonicalRecordBytes: bytes
        )
    }
}

private extension SyntheticRuntimeClosureRecordSchemaVerifier {
    static let absentCodeDirectorySHA256 =
        String(repeating: "0", count: 64)

    static func rederive(
        _ evidence: ExecutableContentIdentityEvidence
    ) -> ExecutableContentIdentityEvidence? {
        try? ExecutableContentIdentityVerifier.derive(
            artifactRole: evidence.artifactRole,
            comparison: evidence.comparison
        )
    }

    static func rederive(
        _ evidence:
            SyntheticSharedCacheImageContentIdentityEvidence
    ) -> SyntheticSharedCacheImageContentIdentityEvidence? {
        try? SyntheticSharedCacheImageContentIdentityVerifier
            .derive(
                cacheSetEvidence: evidence.cacheSetEvidence,
                facts: evidence.facts
            )
    }

    static func rederive(
        _ record: SyntheticRuntimeClosureMemberRecordComparison
    ) -> SyntheticRuntimeClosureMemberRecordComparison? {
        try? member(
            index: record.index,
            source: record.source,
            installName: record.installName
        )
    }

    static func memberRecordBytes(
        index: Int,
        contentEvidenceID: String,
        storage: SyntheticRuntimeClosureMemberStorage,
        installName: SyntheticRuntimeClosureInstallName,
        machOUUID: String,
        primaryCodeDirectoryBlobSHA256: String,
        loadCommandsSHA256: String
    ) -> Data {
        let prefix = "member_\(fourDigit(index))"
        let lines = [
            "\(prefix)_content_evidence_id=\(contentEvidenceID)",
            "\(prefix)_storage=\(storage.rawValue)",
            "\(prefix)_install_name_bytes=\(installName.bytes)",
            "\(prefix)_install_name_base64url=" +
                installName.base64URL,
            "\(prefix)_macho_uuid=\(machOUUID)",
            "\(prefix)_primary_code_directory_blob_sha256=" +
                primaryCodeDirectoryBlobSHA256,
            "\(prefix)_load_commands_sha256=" +
                loadCommandsSHA256,
        ]
        return Data((lines.joined(separator: "\n") + "\n").utf8)
    }

    static func edgeRecordBytes(
        index: Int,
        parentContentEvidenceID: String,
        loadCommandOrdinal: UInt64,
        kind: SyntheticRuntimeClosureEdgeKind,
        installName: SyntheticRuntimeClosureInstallName,
        resolvedContentEvidenceID: String
    ) -> Data {
        let prefix = "edge_\(fourDigit(index))"
        let lines = [
            "\(prefix)_parent_content_evidence_id=" +
                parentContentEvidenceID,
            "\(prefix)_load_command_ordinal=" +
                String(loadCommandOrdinal),
            "\(prefix)_kind=\(kind.rawValue)",
            "\(prefix)_install_name_bytes=\(installName.bytes)",
            "\(prefix)_install_name_base64url=" +
                installName.base64URL,
            "\(prefix)_resolved_content_evidence_id=" +
                resolvedContentEvidenceID,
        ]
        return Data((lines.joined(separator: "\n") + "\n").utf8)
    }

    static func fourDigit(_ value: Int) -> String {
        let text = String(value)
        return String(
            repeating: "0",
            count: 4 - text.count
        ) + text
    }

    static func hex(_ value: Data) -> String {
        value.map { String(format: "%02x", $0) }.joined()
    }
}
