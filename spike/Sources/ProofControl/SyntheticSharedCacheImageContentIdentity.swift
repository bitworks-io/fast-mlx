import CryptoKit
import Foundation

enum SyntheticSharedCacheImagePrimaryCodeDirectory:
    Equatable,
    Sendable
{
    case absent
    case present(blobSHA256: String)
}

struct SyntheticSharedCacheImageContentFacts:
    Equatable,
    Sendable
{
    let installNameBytes: UInt64
    let installNameBase64URL: String
    let machOUUID: String
    let primaryCodeDirectory:
        SyntheticSharedCacheImagePrimaryCodeDirectory
    let loadCommandsSHA256: String
}

enum SyntheticSharedCacheImageContentField:
    Equatable,
    Sendable
{
    case installNameBytes
    case installNameBase64URL
    case installNameSyntax
    case machOUUID
    case primaryCodeDirectoryBlobSHA256
    case loadCommandsSHA256
}

enum SyntheticSharedCacheImageContentIdentityFailure:
    Error,
    Equatable,
    Sendable
{
    case cacheSetEvidenceMismatch
    case invalidField(SyntheticSharedCacheImageContentField)
}

struct SharedCacheImageContentEvidenceID:
    Equatable,
    Sendable
{
    let sha256: String

    fileprivate init(sha256: String) {
        self.sha256 = sha256
    }
}

/// Retained synthetic comparison facts only. The zero CodeDirectory digest
/// means only that this synthetic image declares no embedded CodeDirectory;
/// it grants no signer, launch, capture, or runtime authority.
struct SyntheticSharedCacheImageContentIdentityEvidence: Equatable {
    let cacheSetEvidence: SyntheticSharedCacheSetIdentityEvidence
    let facts: SyntheticSharedCacheImageContentFacts
    let decodedInstallName: Data
    let primaryCodeDirectoryBlobSHA256: String
    let identityPreimage: Data
    let contentEvidenceID: SharedCacheImageContentEvidenceID

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
        cacheSetEvidence: SyntheticSharedCacheSetIdentityEvidence,
        facts: SyntheticSharedCacheImageContentFacts,
        decodedInstallName: Data,
        primaryCodeDirectoryBlobSHA256: String,
        identityPreimage: Data
    ) {
        self.cacheSetEvidence = cacheSetEvidence
        self.facts = facts
        self.decodedInstallName = decodedInstallName
        self.primaryCodeDirectoryBlobSHA256 =
            primaryCodeDirectoryBlobSHA256
        self.identityPreimage = identityPreimage
        self.contentEvidenceID = SharedCacheImageContentEvidenceID(
            sha256: Self.sha256Hex(identityPreimage)
        )
    }

    private static func sha256Hex(_ bytes: Data) -> String {
        SHA256.hash(data: bytes)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}

enum SyntheticSharedCacheImageContentIdentityVerifier {
    static let identityDomain =
        "fast-mlx-proof-control-" +
        "shared-cache-image-content-evidence-id-v1"
    static let absentCodeDirectorySHA256 =
        String(repeating: "0", count: 64)

    static func derive(
        cacheSetEvidence:
            SyntheticSharedCacheSetIdentityEvidence,
        facts: SyntheticSharedCacheImageContentFacts
    ) throws -> SyntheticSharedCacheImageContentIdentityEvidence {
        let rederivedCacheSet:
            SyntheticSharedCacheSetIdentityEvidence
        do {
            rederivedCacheSet =
                try SyntheticSharedCacheSetIdentityVerifier.derive(
                    records: cacheSetEvidence.records
                )
        } catch {
            throw SyntheticSharedCacheImageContentIdentityFailure
                .cacheSetEvidenceMismatch
        }
        guard rederivedCacheSet == cacheSetEvidence else {
            throw SyntheticSharedCacheImageContentIdentityFailure
                .cacheSetEvidenceMismatch
        }

        let decodedInstallName: Data
        do {
            decodedInstallName =
                try SyntheticRuntimeClosureInstallNameVerifier
                .validate(
                    SyntheticRuntimeClosureInstallName(
                        bytes: facts.installNameBytes,
                        base64URL: facts.installNameBase64URL
                    )
                )
        } catch let failure as
            SyntheticRuntimeClosureInstallNameFailure {
            throw SyntheticSharedCacheImageContentIdentityFailure
                .invalidField(failure.imageContentField)
        }
        guard isLowercaseHex(facts.machOUUID, count: 32) else {
            throw SyntheticSharedCacheImageContentIdentityFailure
                .invalidField(.machOUUID)
        }

        let codeDirectorySHA256: String
        switch facts.primaryCodeDirectory {
        case .absent:
            codeDirectorySHA256 = absentCodeDirectorySHA256
        case let .present(blobSHA256):
            guard
                isLowercaseHex(blobSHA256, count: 64),
                blobSHA256 != absentCodeDirectorySHA256
            else {
                throw SyntheticSharedCacheImageContentIdentityFailure
                    .invalidField(
                        .primaryCodeDirectoryBlobSHA256
                    )
            }
            codeDirectorySHA256 = blobSHA256
        }
        guard isLowercaseHex(
            facts.loadCommandsSHA256,
            count: 64
        ) else {
            throw SyntheticSharedCacheImageContentIdentityFailure
                .invalidField(.loadCommandsSHA256)
        }

        let preimage = identityPreimage(
            cacheSetID:
                cacheSetEvidence.sharedCacheSetID.sha256,
            facts: facts,
            codeDirectorySHA256: codeDirectorySHA256
        )
        return SyntheticSharedCacheImageContentIdentityEvidence(
            cacheSetEvidence: cacheSetEvidence,
            facts: facts,
            decodedInstallName: decodedInstallName,
            primaryCodeDirectoryBlobSHA256:
                codeDirectorySHA256,
            identityPreimage: preimage
        )
    }
}

private extension SyntheticSharedCacheImageContentIdentityVerifier {
    static func identityPreimage(
        cacheSetID: String,
        facts: SyntheticSharedCacheImageContentFacts,
        codeDirectorySHA256: String
    ) -> Data {
        Data(
            (
                [
                    identityDomain,
                    "shared_cache_set_id=\(cacheSetID)",
                    "install_name_bytes=" +
                        String(facts.installNameBytes),
                    "install_name_base64url=" +
                        facts.installNameBase64URL,
                    "macho_uuid=\(facts.machOUUID)",
                    "primary_code_directory_blob_sha256=" +
                        codeDirectorySHA256,
                    "load_commands_sha256=" +
                        facts.loadCommandsSHA256,
                ].joined(separator: "\n") + "\n"
            ).utf8
        )
    }

    static func isLowercaseHex(
        _ value: String,
        count: Int
    ) -> Bool {
        let bytes = Array(value.utf8)
        return bytes.count == count &&
            bytes.allSatisfy {
                (0x30...0x39).contains($0) ||
                    (0x61...0x66).contains($0)
            }
    }
}

private extension SyntheticRuntimeClosureInstallNameFailure {
    var imageContentField: SyntheticSharedCacheImageContentField {
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
