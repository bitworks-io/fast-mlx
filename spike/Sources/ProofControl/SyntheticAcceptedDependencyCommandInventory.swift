import CryptoKit
import Foundation

fileprivate enum SyntheticAcceptedDependencyCommandConstructionSeal:
    Equatable
{
    case verified
}

enum SyntheticAcceptedDependencyCommandBoundsField:
    Equatable,
    Sendable
{
    case regionBytesOutOfRange
    case commandCountOutOfRange(Int)
}

enum SyntheticAcceptedDependencyCommandFramingField:
    Equatable,
    Sendable
{
    case header
    case cmdsizeTooSmall
    case cmdsizeAlignment
    case cmdsizeRange
    case trailingBytes
}

enum SyntheticUnsupportedDependencyCommandKind:
    Equatable,
    Sendable
{
    case weakDylib
    case upwardDylib
    case lazyDylib
    case rpath
    case dyldEnvironment
}

enum SyntheticDependencyCommandLayoutField:
    Equatable,
    Sendable
{
    case dylibCommandSize
    case nameOffset
    case nameTerminator
    case zeroPadding
}

enum SyntheticAcceptedDependencyInstallNameField:
    Equatable,
    Sendable
{
    case installNameBytes
    case installNameBase64URL
    case installNameSyntax
}

enum SyntheticAcceptedDependencyCommandInventoryFailure:
    Error,
    Equatable
{
    case sourceRole(ExecutableContentArtifactRole)
    case unsupportedFileMemberIdentity(
        ExecutableContentArtifactRole
    )
    case fileImageSnapshot(
        SyntheticFileImageRuntimeClosureMemberSnapshotFailure
    )
    case sourceEvidenceMismatch
    case snapshotCountOrLength(
        SyntheticAcceptedDependencyCommandBoundsField
    )
    case snapshotDigestMismatch
    case commandCountOrRegionBounds(
        SyntheticAcceptedDependencyCommandBoundsField
    )
    case commandFraming(
        ordinal: UInt64,
        field: SyntheticAcceptedDependencyCommandFramingField
    )
    case unsupportedCommand(
        ordinal: UInt64,
        kind: SyntheticUnsupportedDependencyCommandKind
    )
    case dependencyCommandLayout(
        ordinal: UInt64,
        field: SyntheticDependencyCommandLayoutField
    )
    case installName(
        ordinal: UInt64,
        field: SyntheticAcceptedDependencyInstallNameField
    )
    case acceptedEntryCountOutOfRange(Int)
}

/// Exact synthetic bytes paired with one already-derived shared-cache image
/// identity. This is comparison input only, not a cache locator or capture
/// capability.
struct SyntheticSharedCacheImageLoadCommandSnapshot: Equatable {
    let imageEvidence:
        SyntheticSharedCacheImageContentIdentityEvidence
    let loadCommandBytes: Data
    let loadCommandCount: UInt32
    let loadCommandsSHA256: String
    fileprivate let constructionSeal:
        SyntheticAcceptedDependencyCommandConstructionSeal

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
        imageEvidence:
            SyntheticSharedCacheImageContentIdentityEvidence,
        loadCommandBytes: Data,
        loadCommandCount: UInt32,
        loadCommandsSHA256: String
    ) {
        self.imageEvidence = imageEvidence
        self.loadCommandBytes = Data(loadCommandBytes)
        self.loadCommandCount = loadCommandCount
        self.loadCommandsSHA256 = loadCommandsSHA256
        self.constructionSeal = .verified
    }
}

enum SyntheticAcceptedDependencyCommandInventorySource:
    Equatable
{
    case root(ExecutableContentIdentityEvidence)
    case fileImageMember(
        SyntheticFileImageRuntimeClosureMemberSnapshot
    )
    case sharedCacheMember(
        SyntheticSharedCacheImageLoadCommandSnapshot
    )
}

struct SyntheticAcceptedDependencyCommand: Equatable {
    let loadCommandOrdinal: UInt64
    let kind: SyntheticRuntimeClosureEdgeKind
    let installName: SyntheticRuntimeClosureInstallName
    let decodedInstallName: Data
    fileprivate let constructionSeal:
        SyntheticAcceptedDependencyCommandConstructionSeal

    fileprivate init(
        loadCommandOrdinal: UInt64,
        kind: SyntheticRuntimeClosureEdgeKind,
        installName: SyntheticRuntimeClosureInstallName,
        decodedInstallName: Data
    ) {
        self.loadCommandOrdinal = loadCommandOrdinal
        self.kind = kind
        self.installName = installName
        self.decodedInstallName = Data(decodedInstallName)
        self.constructionSeal = .verified
    }
}

/// Exact accepted dependency commands from one sealed synthetic source. It
/// has no manifest identity and grants no runtime or filesystem authority.
struct SyntheticAcceptedDependencyCommandInventoryComparison:
    Equatable
{
    let source: SyntheticAcceptedDependencyCommandInventorySource
    let parentContentEvidenceID: String
    let sourceLoadCommandsSHA256: String
    let commandCount: UInt32
    let entries: [SyntheticAcceptedDependencyCommand]
    fileprivate let constructionSeal:
        SyntheticAcceptedDependencyCommandConstructionSeal

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
        source: SyntheticAcceptedDependencyCommandInventorySource,
        parentContentEvidenceID: String,
        sourceLoadCommandsSHA256: String,
        commandCount: UInt32,
        entries: [SyntheticAcceptedDependencyCommand]
    ) {
        self.source = source
        self.parentContentEvidenceID = parentContentEvidenceID
        self.sourceLoadCommandsSHA256 = sourceLoadCommandsSHA256
        self.commandCount = commandCount
        self.entries = entries
        self.constructionSeal = .verified
    }
}

enum SyntheticSharedCacheImageLoadCommandSnapshotVerifier {
    static func derive(
        imageEvidence:
            SyntheticSharedCacheImageContentIdentityEvidence,
        loadCommandBytes: Data
    ) throws -> SyntheticSharedCacheImageLoadCommandSnapshot {
        let rederivedEvidence:
            SyntheticSharedCacheImageContentIdentityEvidence
        do {
            rederivedEvidence =
                try SyntheticSharedCacheImageContentIdentityVerifier
                    .derive(
                        cacheSetEvidence:
                            imageEvidence.cacheSetEvidence,
                        facts: imageEvidence.facts
                    )
        } catch {
            throw SyntheticAcceptedDependencyCommandInventoryFailure
                .sourceEvidenceMismatch
        }
        guard rederivedEvidence == imageEvidence else {
            throw SyntheticAcceptedDependencyCommandInventoryFailure
                .sourceEvidenceMismatch
        }

        let bytes = Data(loadCommandBytes)
        guard
            bytes.count <=
                SyntheticMachOIdentityParser.maximumLoadCommandBytes
        else {
            throw SyntheticAcceptedDependencyCommandInventoryFailure
                .snapshotCountOrLength(.regionBytesOutOfRange)
        }

        let digest = SyntheticAcceptedDependencyCommandParser
            .sha256Hex(bytes)
        guard digest == imageEvidence.facts.loadCommandsSHA256 else {
            throw SyntheticAcceptedDependencyCommandInventoryFailure
                .snapshotDigestMismatch
        }

        let frames = try SyntheticAcceptedDependencyCommandParser
            .frames(
                bytes,
                countFailure: { count in
                    .snapshotCountOrLength(
                        .commandCountOutOfRange(count)
                    )
                }
            )

        return SyntheticSharedCacheImageLoadCommandSnapshot(
            imageEvidence: imageEvidence,
            loadCommandBytes: bytes,
            loadCommandCount: UInt32(frames.count),
            loadCommandsSHA256: digest
        )
    }
}

enum SyntheticAcceptedDependencyCommandInventoryVerifier {
    static func root(
        _ evidence: ExecutableContentIdentityEvidence
    ) throws -> SyntheticAcceptedDependencyCommandInventoryComparison {
        guard evidence.artifactRole == .git ||
                evidence.artifactRole == .selfGuard
        else {
            throw SyntheticAcceptedDependencyCommandInventoryFailure
                .sourceRole(evidence.artifactRole)
        }

        let rederived: ExecutableContentIdentityEvidence
        do {
            rederived = try ExecutableContentIdentityVerifier.derive(
                artifactRole: evidence.artifactRole,
                comparison: evidence.comparison
            )
        } catch {
            throw SyntheticAcceptedDependencyCommandInventoryFailure
                .sourceEvidenceMismatch
        }
        guard rederived == evidence else {
            throw SyntheticAcceptedDependencyCommandInventoryFailure
                .sourceEvidenceMismatch
        }

        let comparison = evidence.comparison
        return try derive(
            source: .root(evidence),
            parentContentEvidenceID: evidence.contentEvidenceID.sha256,
            bytes: comparison.loadCommandBytes,
            commandCount: comparison.loadCommandCount,
            digest: comparison.loadCommandsSHA256
        )
    }

    static func sharedCacheMember(
        _ snapshot: SyntheticSharedCacheImageLoadCommandSnapshot
    ) throws -> SyntheticAcceptedDependencyCommandInventoryComparison {
        let rederived: SyntheticSharedCacheImageLoadCommandSnapshot
        do {
            rederived =
                try SyntheticSharedCacheImageLoadCommandSnapshotVerifier
                    .derive(
                        imageEvidence: snapshot.imageEvidence,
                        loadCommandBytes: snapshot.loadCommandBytes
                    )
        } catch let failure as
            SyntheticAcceptedDependencyCommandInventoryFailure
        {
            throw failure
        } catch {
            throw SyntheticAcceptedDependencyCommandInventoryFailure
                .sourceEvidenceMismatch
        }
        guard rederived == snapshot else {
            throw SyntheticAcceptedDependencyCommandInventoryFailure
                .sourceEvidenceMismatch
        }

        return try derive(
            source: .sharedCacheMember(snapshot),
            parentContentEvidenceID:
                snapshot.imageEvidence.contentEvidenceID.sha256,
            bytes: snapshot.loadCommandBytes,
            commandCount: snapshot.loadCommandCount,
            digest: snapshot.loadCommandsSHA256
        )
    }

    static func fileMember(
        _ evidence: ExecutableContentIdentityEvidence
    ) throws -> SyntheticAcceptedDependencyCommandInventoryComparison {
        throw SyntheticAcceptedDependencyCommandInventoryFailure
            .unsupportedFileMemberIdentity(evidence.artifactRole)
    }

    static func fileImageMember(
        _ snapshot: SyntheticFileImageRuntimeClosureMemberSnapshot
    ) throws -> SyntheticAcceptedDependencyCommandInventoryComparison {
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
            throw SyntheticAcceptedDependencyCommandInventoryFailure
                .fileImageSnapshot(failure)
        }
        guard rederived == snapshot else {
            throw SyntheticAcceptedDependencyCommandInventoryFailure
                .sourceEvidenceMismatch
        }

        let comparison = snapshot.fileImageEvidence.comparison
        return try derive(
            source: .fileImageMember(snapshot),
            parentContentEvidenceID:
                snapshot.fileImageEvidence.contentEvidenceID.sha256,
            bytes: comparison.loadCommandBytes,
            commandCount: comparison.loadCommandCount,
            digest: comparison.loadCommandsSHA256
        )
    }
}

private extension SyntheticAcceptedDependencyCommandInventoryVerifier {
    static func derive(
        source: SyntheticAcceptedDependencyCommandInventorySource,
        parentContentEvidenceID: String,
        bytes input: Data,
        commandCount: UInt32,
        digest: String
    ) throws -> SyntheticAcceptedDependencyCommandInventoryComparison {
        let bytes = Data(input)
        guard
            bytes.count <=
                SyntheticMachOIdentityParser.maximumLoadCommandBytes
        else {
            throw SyntheticAcceptedDependencyCommandInventoryFailure
                .commandCountOrRegionBounds(.regionBytesOutOfRange)
        }
        guard
            commandCount <=
                SyntheticMachOIdentityParser.maximumLoadCommandCount
        else {
            throw SyntheticAcceptedDependencyCommandInventoryFailure
                .commandCountOrRegionBounds(
                    .commandCountOutOfRange(Int(commandCount))
                )
        }
        guard
            SyntheticAcceptedDependencyCommandParser.sha256Hex(bytes) ==
                digest
        else {
            throw SyntheticAcceptedDependencyCommandInventoryFailure
                .sourceEvidenceMismatch
        }

        let frames = try SyntheticAcceptedDependencyCommandParser
            .frames(
                bytes,
                countFailure: { count in
                    .commandCountOrRegionBounds(
                        .commandCountOutOfRange(count)
                    )
                }
            )
        guard frames.count == Int(commandCount) else {
            throw SyntheticAcceptedDependencyCommandInventoryFailure
                .commandCountOrRegionBounds(
                    .commandCountOutOfRange(frames.count)
                )
        }

        let entries = try SyntheticAcceptedDependencyCommandParser
            .acceptedCommands(bytes, frames: frames)
        guard
            entries.count <=
                Int(
                    SyntheticMachOIdentityParser
                        .maximumLoadCommandCount
                )
        else {
            throw SyntheticAcceptedDependencyCommandInventoryFailure
                .acceptedEntryCountOutOfRange(entries.count)
        }

        return SyntheticAcceptedDependencyCommandInventoryComparison(
            source: source,
            parentContentEvidenceID: parentContentEvidenceID,
            sourceLoadCommandsSHA256: digest,
            commandCount: commandCount,
            entries: entries
        )
    }
}

private enum SyntheticAcceptedDependencyCommandParser {
    struct Frame {
        let range: Range<Int>
        let command: UInt32
    }

    struct AcceptedFrame {
        let ordinal: UInt64
        let frame: Frame
        let kind: SyntheticRuntimeClosureEdgeKind
    }

    struct ParsedInstallName {
        let ordinal: UInt64
        let kind: SyntheticRuntimeClosureEdgeKind
        let decoded: Data
    }

    static let loadCommandHeaderBytes = 8
    static let dylibCommandBytes = 24

    static let lcLoadDylib: UInt32 = 0x0c
    static let lcLoadWeakDylib: UInt32 = 0x80000018
    static let lcRpath: UInt32 = 0x8000001c
    static let lcReexportDylib: UInt32 = 0x8000001f
    static let lcLazyLoadDylib: UInt32 = 0x20
    static let lcLoadUpwardDylib: UInt32 = 0x80000023
    static let lcDyldEnvironment: UInt32 = 0x27

    static func frames(
        _ input: Data,
        countFailure: (Int) ->
            SyntheticAcceptedDependencyCommandInventoryFailure
    ) throws -> [Frame] {
        let bytes = Data(input)
        var cursor = 0
        var result: [Frame] = []
        result.reserveCapacity(
            min(
                bytes.count / loadCommandHeaderBytes,
                Int(
                    SyntheticMachOIdentityParser
                        .maximumLoadCommandCount
                )
            )
        )

        while cursor < bytes.count {
            let ordinal = UInt64(result.count)
            guard
                result.count <
                    Int(
                        SyntheticMachOIdentityParser
                            .maximumLoadCommandCount
                    )
            else {
                throw countFailure(result.count + 1)
            }
            guard
                let headerEnd = checkedAdd(
                    cursor,
                    loadCommandHeaderBytes
                ),
                headerEnd <= bytes.count
            else {
                throw SyntheticAcceptedDependencyCommandInventoryFailure
                    .commandFraming(
                        ordinal: ordinal,
                        field: result.isEmpty ? .header : .trailingBytes
                    )
            }

            let command = try requiredUInt32LE(
                bytes,
                at: cursor,
                ordinal: ordinal
            )
            let sizeValue = try requiredUInt32LE(
                bytes,
                at: cursor + 4,
                ordinal: ordinal
            )
            guard sizeValue >= UInt32(loadCommandHeaderBytes) else {
                throw SyntheticAcceptedDependencyCommandInventoryFailure
                    .commandFraming(
                        ordinal: ordinal,
                        field: .cmdsizeTooSmall
                    )
            }
            guard
                sizeValue <= UInt32(
                    SyntheticMachOIdentityParser
                        .maximumLoadCommandBytes
                )
            else {
                throw SyntheticAcceptedDependencyCommandInventoryFailure
                    .commandFraming(
                        ordinal: ordinal,
                        field: .cmdsizeRange
                    )
            }
            guard sizeValue.isMultiple(of: 8) else {
                throw SyntheticAcceptedDependencyCommandInventoryFailure
                    .commandFraming(
                        ordinal: ordinal,
                        field: .cmdsizeAlignment
                    )
            }

            let size = Int(sizeValue)
            guard
                let end = checkedAdd(cursor, size),
                end <= bytes.count
            else {
                throw SyntheticAcceptedDependencyCommandInventoryFailure
                    .commandFraming(
                        ordinal: ordinal,
                        field: .cmdsizeRange
                    )
            }
            result.append(
                Frame(range: cursor..<end, command: command)
            )
            cursor = end
        }

        return result
    }

    static func acceptedCommands(
        _ bytes: Data,
        frames: [Frame]
    ) throws -> [SyntheticAcceptedDependencyCommand] {
        var accepted: [AcceptedFrame] = []
        accepted.reserveCapacity(frames.count)

        for (index, frame) in frames.enumerated() {
            let ordinal = UInt64(index)
            let kind: SyntheticRuntimeClosureEdgeKind
            switch frame.command {
            case lcLoadDylib:
                kind = .load
            case lcReexportDylib:
                kind = .reexport
            case lcLoadWeakDylib:
                throw unsupported(ordinal, .weakDylib)
            case lcLoadUpwardDylib:
                throw unsupported(ordinal, .upwardDylib)
            case lcLazyLoadDylib:
                throw unsupported(ordinal, .lazyDylib)
            case lcRpath:
                throw unsupported(ordinal, .rpath)
            case lcDyldEnvironment:
                throw unsupported(ordinal, .dyldEnvironment)
            default:
                continue
            }
            accepted.append(
                AcceptedFrame(
                    ordinal: ordinal,
                    frame: frame,
                    kind: kind
                )
            )
        }

        var parsedNames: [ParsedInstallName] = []
        parsedNames.reserveCapacity(accepted.count)
        for acceptedFrame in accepted {
            parsedNames.append(
                ParsedInstallName(
                    ordinal: acceptedFrame.ordinal,
                    kind: acceptedFrame.kind,
                    decoded: try dependencyInstallNameBytes(
                        bytes,
                        range: acceptedFrame.frame.range,
                        ordinal: acceptedFrame.ordinal
                    )
                )
            )
        }

        var result: [SyntheticAcceptedDependencyCommand] = []
        result.reserveCapacity(parsedNames.count)
        for parsedName in parsedNames {
            let name = try validatedInstallName(
                parsedName.decoded,
                ordinal: parsedName.ordinal
            )
            result.append(
                SyntheticAcceptedDependencyCommand(
                    loadCommandOrdinal: parsedName.ordinal,
                    kind: parsedName.kind,
                    installName: name.value,
                    decodedInstallName: name.decoded
                )
            )
        }
        return result
    }

    static func dependencyInstallNameBytes(
        _ bytes: Data,
        range: Range<Int>,
        ordinal: UInt64
    ) throws -> Data {
        guard range.count >= dylibCommandBytes else {
            throw SyntheticAcceptedDependencyCommandInventoryFailure
                .dependencyCommandLayout(
                    ordinal: ordinal,
                    field: .dylibCommandSize
                )
        }
        let offsetValue = try requiredUInt32LE(
            bytes,
            at: range.lowerBound + 8,
            ordinal: ordinal
        )
        let offset = Int(offsetValue)
        guard
            offset >= dylibCommandBytes,
            offset < range.count,
            let nameStart = checkedAdd(range.lowerBound, offset),
            nameStart < range.upperBound
        else {
            throw SyntheticAcceptedDependencyCommandInventoryFailure
                .dependencyCommandLayout(
                    ordinal: ordinal,
                    field: .nameOffset
                )
        }
        guard
            bytes[(range.lowerBound + dylibCommandBytes)..<nameStart]
                .allSatisfy({ $0 == 0 })
        else {
            throw SyntheticAcceptedDependencyCommandInventoryFailure
                .dependencyCommandLayout(
                    ordinal: ordinal,
                    field: .zeroPadding
                )
        }

        var terminator: Int?
        var cursor = nameStart
        while cursor < range.upperBound {
            if bytes[cursor] == 0 {
                terminator = cursor
                break
            }
            cursor += 1
        }
        guard let terminator else {
            throw SyntheticAcceptedDependencyCommandInventoryFailure
                .dependencyCommandLayout(
                    ordinal: ordinal,
                    field: .nameTerminator
                )
        }
        guard
            bytes[(terminator + 1)..<range.upperBound]
                .allSatisfy({ $0 == 0 })
        else {
            throw SyntheticAcceptedDependencyCommandInventoryFailure
                .dependencyCommandLayout(
                    ordinal: ordinal,
                    field: .zeroPadding
                )
        }

        return Data(bytes[nameStart..<terminator])
    }

    static func validatedInstallName(
        _ decoded: Data,
        ordinal: UInt64
    ) throws -> (
        value: SyntheticRuntimeClosureInstallName,
        decoded: Data
    ) {
        let value = SyntheticRuntimeClosureInstallName(
            bytes: UInt64(decoded.count),
            base64URL: base64URL(decoded)
        )
        do {
            let validated =
                try SyntheticRuntimeClosureInstallNameVerifier
                    .validate(value)
            return (value, validated)
        } catch let failure as
            SyntheticRuntimeClosureInstallNameFailure
        {
            throw SyntheticAcceptedDependencyCommandInventoryFailure
                .installName(
                    ordinal: ordinal,
                    field: installNameField(failure)
                )
        } catch {
            throw SyntheticAcceptedDependencyCommandInventoryFailure
                .installName(
                    ordinal: ordinal,
                    field: .installNameSyntax
                )
        }
    }

    static func installNameField(
        _ failure: SyntheticRuntimeClosureInstallNameFailure
    ) -> SyntheticAcceptedDependencyInstallNameField {
        switch failure {
        case .bytes:
            .installNameBytes
        case .base64URL:
            .installNameBase64URL
        case .syntax:
            .installNameSyntax
        }
    }

    static func unsupported(
        _ ordinal: UInt64,
        _ kind: SyntheticUnsupportedDependencyCommandKind
    ) -> SyntheticAcceptedDependencyCommandInventoryFailure {
        .unsupportedCommand(ordinal: ordinal, kind: kind)
    }

    static func requiredUInt32LE(
        _ bytes: Data,
        at offset: Int,
        ordinal: UInt64
    ) throws -> UInt32 {
        guard
            let end = checkedAdd(offset, 4),
            end <= bytes.count
        else {
            throw SyntheticAcceptedDependencyCommandInventoryFailure
                .commandFraming(ordinal: ordinal, field: .header)
        }
        return UInt32(bytes[offset]) |
            UInt32(bytes[offset + 1]) << 8 |
            UInt32(bytes[offset + 2]) << 16 |
            UInt32(bytes[offset + 3]) << 24
    }

    static func checkedAdd(_ lhs: Int, _ rhs: Int) -> Int? {
        let (sum, overflow) = lhs.addingReportingOverflow(rhs)
        return overflow ? nil : sum
    }

    static func base64URL(_ data: Data) -> String {
        data.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    static func sha256Hex(_ bytes: Data) -> String {
        SHA256.hash(data: bytes)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}
