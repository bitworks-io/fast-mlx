import Foundation

fileprivate enum
    SyntheticFileImageRuntimeClosureMemberSnapshotConstructionSeal
{
    case verified
}

enum SyntheticFileImageRuntimeClosureMemberSnapshotFailure:
    Error,
    Equatable
{
    case sourceEvidence(FileImageContentIdentityFailure)
    case sourceEvidenceMismatch
    case identityName(SyntheticRuntimeClosureInstallNameFailure)
}

/// Exact sealed D2 evidence joined to one canonical runtime-closure label.
/// This is comparison data only; it is not a path, locator, or capability.
struct SyntheticFileImageRuntimeClosureMemberSnapshot: Equatable {
    let fileImageEvidence: FileImageContentIdentityEvidence
    let installName: SyntheticRuntimeClosureInstallName
    let decodedInstallName: Data
    fileprivate let constructionSeal:
        SyntheticFileImageRuntimeClosureMemberSnapshotConstructionSeal

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
        fileImageEvidence: FileImageContentIdentityEvidence,
        installName: SyntheticRuntimeClosureInstallName,
        decodedInstallName: Data,
        seal:
            SyntheticFileImageRuntimeClosureMemberSnapshotConstructionSeal
    ) {
        self.fileImageEvidence = fileImageEvidence
        self.installName = installName
        self.decodedInstallName = Data(decodedInstallName)
        self.constructionSeal = seal
    }
}

enum SyntheticFileImageRuntimeClosureMemberSnapshotVerifier {
    static func derive(
        fileImageEvidence: FileImageContentIdentityEvidence
    ) throws -> SyntheticFileImageRuntimeClosureMemberSnapshot {
        let rederived: FileImageContentIdentityEvidence
        do {
            rederived = try FileImageContentIdentityVerifier.derive(
                comparison: fileImageEvidence.comparison
            )
        } catch let failure as FileImageContentIdentityFailure {
            throw SyntheticFileImageRuntimeClosureMemberSnapshotFailure
                .sourceEvidence(failure)
        }
        guard rederived == fileImageEvidence else {
            throw SyntheticFileImageRuntimeClosureMemberSnapshotFailure
                .sourceEvidenceMismatch
        }

        let identityName = Data(
            fileImageEvidence.comparison.dylibIDName
        )
        guard let byteCount = UInt64(exactly: identityName.count) else {
            throw SyntheticFileImageRuntimeClosureMemberSnapshotFailure
                .identityName(.bytes)
        }
        let installName = SyntheticRuntimeClosureInstallName(
            bytes: byteCount,
            base64URL: base64URL(identityName)
        )
        let decoded: Data
        do {
            decoded = try SyntheticRuntimeClosureInstallNameVerifier
                .validate(installName)
        } catch let failure as SyntheticRuntimeClosureInstallNameFailure {
            throw SyntheticFileImageRuntimeClosureMemberSnapshotFailure
                .identityName(failure)
        }
        guard decoded == identityName else {
            throw SyntheticFileImageRuntimeClosureMemberSnapshotFailure
                .identityName(.base64URL)
        }

        return SyntheticFileImageRuntimeClosureMemberSnapshot(
            fileImageEvidence: fileImageEvidence,
            installName: installName,
            decodedInstallName: decoded,
            seal: .verified
        )
    }
}

private extension SyntheticFileImageRuntimeClosureMemberSnapshotVerifier {
    static func base64URL(_ data: Data) -> String {
        data.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }
}
