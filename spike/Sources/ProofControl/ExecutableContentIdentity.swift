import CryptoKit
import Foundation

enum ExecutableContentArtifactRole:
    String,
    Equatable,
    Sendable
{
    case git
    case selfGuard = "self-guard"
    case fileImage = "file-image"
    case dynamicLoader = "dynamic-loader"
}

struct ExecutableContentEvidenceID:
    Equatable,
    Sendable
{
    let sha256: String

    fileprivate init(sha256: String) {
        self.sha256 = sha256
    }
}

enum ExecutableContentCodeDirectoryField:
    Equatable,
    Sendable
{
    case slot
    case blobSHA256
    case blobBytes
    case hashType
    case flags
    case signingIdentifierBytes
    case signingIdentifierBase64URL
    case teamIdentifierBytes
    case teamIdentifierBase64URL
}

enum ExecutableContentExpectationField:
    Equatable,
    Sendable
{
    case artifactRole
    case fileSHA256
    case fileBytes
    case cpuSubtype
    case headerFlags
    case loadCommandCount
    case loadCommandBytes
    case loadCommandsSHA256
    case machOUUID
    case codeSignatureRegionSHA256
    case codeSignatureRegionBytes
    case codeDirectoryCount
    case codeDirectory(
        index: Int,
        field: ExecutableContentCodeDirectoryField
    )
    case cmsBlobSHA256
    case cmsBlobBytes
}

enum ExecutableContentIdentityFailure:
    Error,
    Equatable,
    Sendable
{
    case missingPrimaryCodeDirectory
    case parserFactMismatch
    case expectationRoleUnsupported(
        ExecutableContentArtifactRole
    )
    case expectationMismatch(
        ExecutableContentExpectationField
    )
}

/// Portable synthetic content facts only. The ID preimage deliberately omits
/// every expectation, policy, filesystem, runtime, and self-reference field.
struct ExecutableContentIdentityEvidence: Equatable {
    let artifactRole: ExecutableContentArtifactRole
    let identityPreimage: Data
    let contentEvidenceID: ExecutableContentEvidenceID
    let machHeaderSHA256: String
    let primaryCodeDirectoryBlobSHA256: String
    let comparison: SyntheticMachOIdentityComparison

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
        artifactRole: ExecutableContentArtifactRole,
        identityPreimage: Data,
        machHeaderSHA256: String,
        primaryCodeDirectoryBlobSHA256: String,
        comparison: SyntheticMachOIdentityComparison
    ) {
        self.artifactRole = artifactRole
        self.identityPreimage = identityPreimage
        self.contentEvidenceID = ExecutableContentEvidenceID(
            sha256: Self.sha256Hex(identityPreimage)
        )
        self.machHeaderSHA256 = machHeaderSHA256
        self.primaryCodeDirectoryBlobSHA256 =
            primaryCodeDirectoryBlobSHA256
        self.comparison = comparison
    }

    private static func sha256Hex(_ bytes: Data) -> String {
        SHA256.hash(data: bytes)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}

fileprivate enum DynamicLoaderContentIdentityConstructionSeal {
    case verifiedSyntheticDynamicLoader
}

fileprivate enum FileImageContentIdentityConstructionSeal {
    case verifiedSyntheticFileImage
}

enum DynamicLoaderContentIdentityFailure:
    Error,
    Equatable,
    Sendable
{
    case missingPrimaryCodeDirectory
    case parserFactMismatch
}

enum FileImageContentIdentityFailure:
    Error,
    Equatable,
    Sendable
{
    case missingPrimaryCodeDirectory
    case parserFactMismatch
}

/// Sealed file-type-specific evidence. The role is fixed by the verifier and
/// cannot be selected by a caller or substituted by generic execute evidence.
struct DynamicLoaderContentIdentityEvidence: Equatable {
    let artifactRole: ExecutableContentArtifactRole
    let identityPreimage: Data
    let contentEvidenceID: ExecutableContentEvidenceID
    let machHeaderSHA256: String
    let primaryCodeDirectoryBlobSHA256: String
    let comparison: SyntheticDynamicLoaderMachOIdentityComparison

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
        seal: DynamicLoaderContentIdentityConstructionSeal,
        identityPreimage: Data,
        machHeaderSHA256: String,
        primaryCodeDirectoryBlobSHA256: String,
        comparison: SyntheticDynamicLoaderMachOIdentityComparison
    ) {
        switch seal {
        case .verifiedSyntheticDynamicLoader:
            break
        }
        self.artifactRole = .dynamicLoader
        self.identityPreimage = identityPreimage
        self.contentEvidenceID = ExecutableContentEvidenceID(
            sha256: DynamicLoaderContentIdentityVerifier
                .sha256Hex(identityPreimage)
        )
        self.machHeaderSHA256 = machHeaderSHA256
        self.primaryCodeDirectoryBlobSHA256 =
            primaryCodeDirectoryBlobSHA256
        self.comparison = comparison
    }
}

/// Sealed file-type-specific evidence. The role is fixed by the verifier and
/// cannot be selected by a caller or substituted by generic execute evidence.
struct FileImageContentIdentityEvidence: Equatable {
    let artifactRole: ExecutableContentArtifactRole
    let identityPreimage: Data
    let contentEvidenceID: ExecutableContentEvidenceID
    let machHeaderSHA256: String
    let primaryCodeDirectoryBlobSHA256: String
    let comparison: SyntheticFileImageMachOIdentityComparison

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
        seal: FileImageContentIdentityConstructionSeal,
        identityPreimage: Data,
        machHeaderSHA256: String,
        primaryCodeDirectoryBlobSHA256: String,
        comparison: SyntheticFileImageMachOIdentityComparison
    ) {
        switch seal {
        case .verifiedSyntheticFileImage:
            break
        }
        self.artifactRole = .fileImage
        self.identityPreimage = identityPreimage
        self.contentEvidenceID = ExecutableContentEvidenceID(
            sha256: FileImageContentIdentityVerifier
                .sha256Hex(identityPreimage)
        )
        self.machHeaderSHA256 = machHeaderSHA256
        self.primaryCodeDirectoryBlobSHA256 =
            primaryCodeDirectoryBlobSHA256
        self.comparison = comparison
    }
}

/// Exact one-way equality between inert content facts and one already
/// canonical, already anchored expectation document. No transcript is created
/// here; the later six-anchor context owns the first transcript ID.
struct ExecutableContentExpectationComparison: Equatable {
    let contentEvidence: ExecutableContentIdentityEvidence
    let expectation: AnchoredExecutableIdentityExpectationDocument

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
        contentEvidence: ExecutableContentIdentityEvidence,
        expectation: AnchoredExecutableIdentityExpectationDocument
    ) {
        self.contentEvidence = contentEvidence
        self.expectation = expectation
    }
}

enum ExecutableContentIdentityVerifier {
    static let identityDomain =
        "fast-mlx-proof-control-executable-content-evidence-id-v1"

    static func derive(
        artifactRole: ExecutableContentArtifactRole,
        comparison: SyntheticMachOIdentityComparison
    ) throws -> ExecutableContentIdentityEvidence {
        guard let primary = comparison.codeDirectories.first,
              primary.slot == 0
        else {
            throw ExecutableContentIdentityFailure
                .missingPrimaryCodeDirectory
        }

        let reparsed: SyntheticMachOIdentityComparison
        do {
            reparsed = try SyntheticMachOIdentityParser.parse(
                comparison.retainedFileBytes
            )
        } catch {
            throw ExecutableContentIdentityFailure
                .parserFactMismatch
        }
        guard reparsed == comparison,
              let fileBytes = UInt64(
                  exactly: comparison.retainedFileBytes.count
              )
        else {
            throw ExecutableContentIdentityFailure
                .parserFactMismatch
        }

        let machHeader = Data(
            comparison.retainedFileBytes.prefix(32)
        )
        guard machHeader.count == 32 else {
            throw ExecutableContentIdentityFailure
                .parserFactMismatch
        }
        let machHeaderSHA256 = sha256Hex(machHeader)
        let preimage = makeExecutableContentIdentityPreimage(
            artifactRole: artifactRole,
            fileSHA256: comparison.fileSHA256,
            fileBytes: fileBytes,
            machHeaderSHA256: machHeaderSHA256,
            loadCommandsSHA256: comparison.loadCommandsSHA256,
            machOUUID: hex(comparison.machOUUID),
            primaryCodeDirectoryBlobSHA256: primary.blobSHA256,
            codeSignatureRegionSHA256:
                comparison.codeSignatureRegionSHA256
        )

        return ExecutableContentIdentityEvidence(
            artifactRole: artifactRole,
            identityPreimage: preimage,
            machHeaderSHA256: machHeaderSHA256,
            primaryCodeDirectoryBlobSHA256:
                primary.blobSHA256,
            comparison: comparison
        )
    }

    static func match(
        evidence: ExecutableContentIdentityEvidence,
        expectation: AnchoredExecutableIdentityExpectationDocument
    ) throws -> ExecutableContentExpectationComparison {
        let expectationRole:
            ExecutableIdentityArtifactRole
        switch evidence.artifactRole {
        case .git:
            expectationRole = .git
        case .selfGuard:
            expectationRole = .selfGuard
        case .fileImage, .dynamicLoader:
            throw ExecutableContentIdentityFailure
                .expectationRoleUnsupported(
                    evidence.artifactRole
                )
        }

        let actual = evidence.comparison
        let expected = expectation.fields
        try require(
            expectationRole,
            equals: expected.artifactRole,
            field: .artifactRole
        )
        try require(
            actual.fileSHA256,
            equals: expected.fileSHA256,
            field: .fileSHA256
        )
        try require(
            UInt64(actual.retainedFileBytes.count),
            equals: expected.fileBytes,
            field: .fileBytes
        )
        try require(
            hex8(actual.cpuSubtype),
            equals: expected.cpuSubtype,
            field: .cpuSubtype
        )
        try require(
            hex8(actual.headerFlags),
            equals: expected.headerFlags,
            field: .headerFlags
        )
        try require(
            UInt64(actual.loadCommandCount),
            equals: expected.loadCommandCount,
            field: .loadCommandCount
        )
        try require(
            UInt64(actual.loadCommandBytes.count),
            equals: expected.loadCommandBytes,
            field: .loadCommandBytes
        )
        try require(
            actual.loadCommandsSHA256,
            equals: expected.loadCommandsSHA256,
            field: .loadCommandsSHA256
        )
        try require(
            hex(actual.machOUUID),
            equals: expected.machOUUID,
            field: .machOUUID
        )
        try require(
            actual.codeSignatureRegionSHA256,
            equals: expected.codeSignatureRegionSHA256,
            field: .codeSignatureRegionSHA256
        )
        try require(
            UInt64(actual.codeSignatureRegion.count),
            equals: expected.codeSignatureRegionBytes,
            field: .codeSignatureRegionBytes
        )
        try require(
            actual.codeDirectories.count,
            equals: expected.codeDirectories.count,
            field: .codeDirectoryCount
        )

        for index in actual.codeDirectories.indices {
            let actualDirectory = actual.codeDirectories[index]
            let expectedDirectory =
                expected.codeDirectories[index]
            try require(
                UInt64(actualDirectory.slot),
                equals: expectedDirectory.slot,
                field: .codeDirectory(index: index, field: .slot)
            )
            try require(
                actualDirectory.blobSHA256,
                equals: expectedDirectory.blobSHA256,
                field: .codeDirectory(
                    index: index,
                    field: .blobSHA256
                )
            )
            try require(
                UInt64(actualDirectory.blob.count),
                equals: expectedDirectory.blobBytes,
                field: .codeDirectory(
                    index: index,
                    field: .blobBytes
                )
            )
            try require(
                UInt64(actualDirectory.hashType),
                equals: expectedDirectory.hashType,
                field: .codeDirectory(
                    index: index,
                    field: .hashType
                )
            )
            try require(
                hex8(actualDirectory.flags),
                equals: expectedDirectory.flags,
                field: .codeDirectory(
                    index: index,
                    field: .flags
                )
            )
            try require(
                UInt64(actualDirectory.signingIdentifier.count),
                equals: expectedDirectory.signingIdentifierBytes,
                field: .codeDirectory(
                    index: index,
                    field: .signingIdentifierBytes
                )
            )
            try require(
                base64URL(actualDirectory.signingIdentifier),
                equals:
                    expectedDirectory.signingIdentifierBase64URL,
                field: .codeDirectory(
                    index: index,
                    field: .signingIdentifierBase64URL
                )
            )
            try require(
                UInt64(actualDirectory.teamIdentifier.count),
                equals: expectedDirectory.teamIdentifierBytes,
                field: .codeDirectory(
                    index: index,
                    field: .teamIdentifierBytes
                )
            )
            try require(
                base64URL(actualDirectory.teamIdentifier),
                equals: expectedDirectory.teamIdentifierBase64URL,
                field: .codeDirectory(
                    index: index,
                    field: .teamIdentifierBase64URL
                )
            )
        }

        try require(
            actual.cmsBlobSHA256,
            equals: expected.cmsBlobSHA256,
            field: .cmsBlobSHA256
        )
        try require(
            UInt64(actual.cmsBlob?.count ?? 0),
            equals: expected.cmsBlobBytes,
            field: .cmsBlobBytes
        )

        return ExecutableContentExpectationComparison(
            contentEvidence: evidence,
            expectation: expectation
        )
    }
}

enum DynamicLoaderContentIdentityVerifier {
    static func derive(
        comparison: SyntheticDynamicLoaderMachOIdentityComparison
    ) throws -> DynamicLoaderContentIdentityEvidence {
        guard let primary = comparison.codeDirectories.first,
              primary.slot == 0
        else {
            throw DynamicLoaderContentIdentityFailure
                .missingPrimaryCodeDirectory
        }

        let reparsed: SyntheticDynamicLoaderMachOIdentityComparison
        do {
            reparsed = try SyntheticDynamicLoaderMachOIdentityParser
                .parse(comparison.retainedFileBytes)
        } catch {
            throw DynamicLoaderContentIdentityFailure
                .parserFactMismatch
        }
        guard reparsed == comparison,
              let fileBytes = UInt64(
                  exactly: comparison.retainedFileBytes.count
              )
        else {
            throw DynamicLoaderContentIdentityFailure
                .parserFactMismatch
        }

        let machHeader = Data(
            comparison.retainedFileBytes.prefix(32)
        )
        guard machHeader.count == 32 else {
            throw DynamicLoaderContentIdentityFailure
                .parserFactMismatch
        }
        let machHeaderSHA256 = sha256Hex(machHeader)
        let preimage = makeExecutableContentIdentityPreimage(
            artifactRole: .dynamicLoader,
            fileSHA256: comparison.fileSHA256,
            fileBytes: fileBytes,
            machHeaderSHA256: machHeaderSHA256,
            loadCommandsSHA256: comparison.loadCommandsSHA256,
            machOUUID: hex(comparison.machOUUID),
            primaryCodeDirectoryBlobSHA256: primary.blobSHA256,
            codeSignatureRegionSHA256:
                comparison.codeSignatureRegionSHA256
        )

        return DynamicLoaderContentIdentityEvidence(
            seal: .verifiedSyntheticDynamicLoader,
            identityPreimage: preimage,
            machHeaderSHA256: machHeaderSHA256,
            primaryCodeDirectoryBlobSHA256: primary.blobSHA256,
            comparison: comparison
        )
    }
}

enum FileImageContentIdentityVerifier {
    static func derive(
        comparison: SyntheticFileImageMachOIdentityComparison
    ) throws -> FileImageContentIdentityEvidence {
        guard let primary = comparison.codeDirectories.first,
              primary.slot == 0
        else {
            throw FileImageContentIdentityFailure
                .missingPrimaryCodeDirectory
        }

        let reparsed: SyntheticFileImageMachOIdentityComparison
        do {
            reparsed = try SyntheticFileImageMachOIdentityParser
                .parse(comparison.retainedFileBytes)
        } catch {
            throw FileImageContentIdentityFailure
                .parserFactMismatch
        }
        guard reparsed == comparison,
              let fileBytes = UInt64(
                  exactly: comparison.retainedFileBytes.count
              )
        else {
            throw FileImageContentIdentityFailure
                .parserFactMismatch
        }

        let machHeader = Data(
            comparison.retainedFileBytes.prefix(32)
        )
        guard machHeader.count == 32 else {
            throw FileImageContentIdentityFailure
                .parserFactMismatch
        }
        let machHeaderSHA256 = sha256Hex(machHeader)
        let preimage = makeExecutableContentIdentityPreimage(
            artifactRole: .fileImage,
            fileSHA256: comparison.fileSHA256,
            fileBytes: fileBytes,
            machHeaderSHA256: machHeaderSHA256,
            loadCommandsSHA256: comparison.loadCommandsSHA256,
            machOUUID: hex(comparison.machOUUID),
            primaryCodeDirectoryBlobSHA256: primary.blobSHA256,
            codeSignatureRegionSHA256:
                comparison.codeSignatureRegionSHA256
        )

        return FileImageContentIdentityEvidence(
            seal: .verifiedSyntheticFileImage,
            identityPreimage: preimage,
            machHeaderSHA256: machHeaderSHA256,
            primaryCodeDirectoryBlobSHA256: primary.blobSHA256,
            comparison: comparison
        )
    }
}

extension DynamicLoaderContentIdentityVerifier {
    fileprivate static func hex(_ data: Data) -> String {
        data.map { String(format: "%02x", $0) }.joined()
    }

    fileprivate static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}

extension FileImageContentIdentityVerifier {
    fileprivate static func hex(_ data: Data) -> String {
        data.map { String(format: "%02x", $0) }.joined()
    }

    fileprivate static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}

fileprivate func makeExecutableContentIdentityPreimage(
    artifactRole: ExecutableContentArtifactRole,
    fileSHA256: String,
    fileBytes: UInt64,
    machHeaderSHA256: String,
    loadCommandsSHA256: String,
    machOUUID: String,
    primaryCodeDirectoryBlobSHA256: String,
    codeSignatureRegionSHA256: String
) -> Data {
    Data(
        (
            [
                ExecutableContentIdentityVerifier.identityDomain,
                "artifact_role=\(artifactRole.rawValue)",
                "file_sha256=\(fileSHA256)",
                "file_bytes=\(fileBytes)",
                "mach_header_sha256=\(machHeaderSHA256)",
                "load_commands_sha256=\(loadCommandsSHA256)",
                "macho_uuid=\(machOUUID)",
                "primary_code_directory_blob_sha256=" +
                    primaryCodeDirectoryBlobSHA256,
                "code_signature_region_sha256=" +
                    codeSignatureRegionSHA256,
            ].joined(separator: "\n") + "\n"
        ).utf8
    )
}

private extension ExecutableContentIdentityVerifier {
    static func require<Value: Equatable>(
        _ actual: Value,
        equals expected: Value,
        field: ExecutableContentExpectationField
    ) throws {
        guard actual == expected else {
            throw ExecutableContentIdentityFailure
                .expectationMismatch(field)
        }
    }

    static func base64URL(_ data: Data) -> String {
        data.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    static func hex(_ data: Data) -> String {
        data.map { String(format: "%02x", $0) }.joined()
    }

    static func hex8(_ value: UInt32) -> String {
        String(format: "%08x", value)
    }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}
