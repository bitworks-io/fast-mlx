import CryptoKit
import Foundation

enum SyntheticMachOIdentityFailure: Error, Equatable {
    case fileBounds
    case unsupportedContainer
    case unsupportedArchitecture
    case unsupportedFileType
    case loadCommandCount
    case loadCommandBounds
    case malformedHeader
    case malformedLoadCommand
    case missingUUID
    case duplicateUUID
    case zeroUUID
    case missingCodeSignature
    case duplicateCodeSignature
    case unsupportedDyldEnvironment
    case codeSignatureBounds
    case superBlobBounds
    case superBlobEntryCount
    case duplicateSuperBlobSlot
    case missingPrimaryCodeDirectory
    case codeDirectoryOrder
    case unsupportedCodeDirectorySlot
    case blobBounds
    case overlappingBlobRanges
    case codeDirectoryBounds
    case unsupportedCodeDirectoryVersion
    case unsupportedCodeDirectoryHashType
    case codeDirectoryHashSize
    case signingIdentifier
    case teamIdentifier
    case cmsBlob
    case cmsAdHocInconsistency
    case malformedDynamicLoaderLoadCommand(ordinal: UInt64)
    case unknownRequiredDynamicLoaderLoadCommand(
        ordinal: UInt64,
        command: UInt32
    )
    case unknownOptionalDynamicLoaderLoadCommand(
        ordinal: UInt64,
        command: UInt32
    )
    case forbiddenDynamicLoaderLoadCommand(
        ordinal: UInt64,
        kind: SyntheticDynamicLoaderForbiddenCommand
    )
    case missingDynamicLoaderIdentity
    case duplicateDynamicLoaderIdentity(ordinal: UInt64)
    case dynamicLoaderIdentityLayout(
        ordinal: UInt64,
        field: SyntheticDynamicLoaderIdentityField
    )
    case malformedFileImageLoadCommand(ordinal: UInt64)
    case unknownRequiredFileImageLoadCommand(
        ordinal: UInt64,
        command: UInt32
    )
    case unknownOptionalFileImageLoadCommand(
        ordinal: UInt64,
        command: UInt32
    )
    case forbiddenFileImageLoadCommand(
        ordinal: UInt64,
        kind: SyntheticFileImageForbiddenCommand
    )
    case missingFileImageIdentity
    case duplicateFileImageIdentity(ordinal: UInt64)
    case fileImageIdentityLayout(
        ordinal: UInt64,
        field: SyntheticFileImageIdentityField
    )
}

struct SyntheticCodeDirectoryComparison: Equatable {
    let slot: UInt32
    let blob: Data
    let blobSHA256: String
    let version: UInt32
    let flags: UInt32
    let hashType: UInt8
    let hashSize: UInt8
    let signingIdentifier: Data
    let teamIdentifier: Data
}

struct SyntheticMachOIdentityComparison: Equatable {
    let retainedFileBytes: Data
    let fileSHA256: String
    let machHeaderMagic: UInt32
    let cpuType: UInt32
    let cpuSubtype: UInt32
    let fileType: UInt32
    let headerFlags: UInt32
    let loadCommandCount: UInt32
    let loadCommandBytes: Data
    let loadCommandsSHA256: String
    let machOUUID: Data
    let codeSignatureRegion: Data
    let codeSignatureRegionSHA256: String
    let codeDirectories: [SyntheticCodeDirectoryComparison]
    let cmsBlob: Data?
    let cmsBlobSHA256: String
    let isAdHoc: Bool

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
}

enum SyntheticMachOIdentityParser {
    // These are intentionally conservative local limits for deterministic
    // synthetic comparison. They are not claims about platform-wide limits.
    static let maximumFileBytes = 1_048_576
    static let maximumLoadCommandCount: UInt32 = 256
    static let maximumLoadCommandBytes = 262_144
    static let maximumSuperBlobEntries: UInt32 = 64

    private static let maximumCodeSignatureBytes = 262_144
    private static let maximumCodeDirectoryBytes = 65_536
    private static let maximumIdentityBytes = 4_096

    private static let machHeader64Bytes = 32
    private static let machMagic64: UInt32 = 0xfeedfacf
    private static let cpuTypeARM64: UInt32 = 0x0100000c
    private static let machExecute: UInt32 = 0x2
    private static let lcUUID: UInt32 = 0x1b
    private static let lcCodeSignature: UInt32 = 0x1d
    private static let lcDyldEnvironment: UInt32 = 0x27

    private static let csMagicBlobWrapper: UInt32 = 0xfade0b01
    private static let csMagicCodeDirectory: UInt32 = 0xfade0c02
    private static let csMagicEmbeddedSignature: UInt32 = 0xfade0cc0
    private static let csAdHoc: UInt32 = 0x2
    private static let primaryCodeDirectorySlot: UInt32 = 0
    private static let firstAlternateCodeDirectorySlot: UInt32 = 0x1000
    private static let lastAlternateCodeDirectorySlot: UInt32 = 0x1004
    private static let cmsSlot: UInt32 = 0x10000

    static func parse(_ input: Data) throws
        -> SyntheticMachOIdentityComparison
    {
        // Normalize slices to a zero-based, value-semantic byte snapshot before
        // any integer-indexed parsing.
        let bytes = Data(input)
        guard bytes.count >= machHeader64Bytes,
              bytes.count <= maximumFileBytes
        else {
            throw SyntheticMachOIdentityFailure.fileBounds
        }
        guard readUInt32LE(bytes, at: 0) == machMagic64 else {
            throw SyntheticMachOIdentityFailure.unsupportedContainer
        }

        let cpuType = try requiredUInt32LE(bytes, at: 4)
        guard cpuType == cpuTypeARM64 else {
            throw SyntheticMachOIdentityFailure.unsupportedArchitecture
        }
        let cpuSubtype = try requiredUInt32LE(bytes, at: 8)
        let fileType = try requiredUInt32LE(bytes, at: 12)
        guard fileType == machExecute else {
            throw SyntheticMachOIdentityFailure.unsupportedFileType
        }
        let commandCount = try requiredUInt32LE(bytes, at: 16)
        guard commandCount > 0,
              commandCount <= maximumLoadCommandCount
        else {
            throw SyntheticMachOIdentityFailure.loadCommandCount
        }
        let commandBytes = Int(try requiredUInt32LE(bytes, at: 20))
        guard commandBytes > 0,
              commandBytes <= maximumLoadCommandBytes,
              let loadCommandRange = checkedRange(
                  offset: machHeader64Bytes,
                  length: commandBytes,
                  limit: bytes.count
              )
        else {
            throw SyntheticMachOIdentityFailure.loadCommandBounds
        }
        let headerFlags = try requiredUInt32LE(bytes, at: 24)
        guard try requiredUInt32LE(bytes, at: 28) == 0 else {
            throw SyntheticMachOIdentityFailure.malformedHeader
        }

        let commandFacts = try parseLoadCommands(
            bytes,
            range: loadCommandRange,
            expectedCount: commandCount
        )
        let signatureRegion = Data(
            bytes[commandFacts.codeSignatureRange]
        )
        let signatureFacts = try parseEmbeddedSignature(
            signatureRegion
        )
        let loadCommands = Data(bytes[loadCommandRange])

        return SyntheticMachOIdentityComparison(
            retainedFileBytes: bytes,
            fileSHA256: sha256Hex(bytes),
            machHeaderMagic: machMagic64,
            cpuType: cpuType,
            cpuSubtype: cpuSubtype,
            fileType: fileType,
            headerFlags: headerFlags,
            loadCommandCount: commandCount,
            loadCommandBytes: loadCommands,
            loadCommandsSHA256: sha256Hex(loadCommands),
            machOUUID: commandFacts.uuid,
            codeSignatureRegion: signatureRegion,
            codeSignatureRegionSHA256: sha256Hex(signatureRegion),
            codeDirectories: signatureFacts.codeDirectories,
            cmsBlob: signatureFacts.cmsBlob,
            cmsBlobSHA256: signatureFacts.cmsBlob.map(sha256Hex)
                ?? String(repeating: "0", count: 64),
            isAdHoc: signatureFacts.isAdHoc
        )
    }

    private struct LoadCommandFacts {
        let uuid: Data
        let codeSignatureRange: Range<Int>
    }

    private static func parseLoadCommands(
        _ bytes: Data,
        range: Range<Int>,
        expectedCount: UInt32
    ) throws -> LoadCommandFacts {
        var cursor = range.lowerBound
        var uuid: Data?
        var codeSignatureRange: Range<Int>?

        for _ in 0..<expectedCount {
            guard let headerRange = checkedRange(
                offset: cursor,
                length: 8,
                limit: range.upperBound
            ) else {
                throw SyntheticMachOIdentityFailure
                    .malformedLoadCommand
            }
            let command = try requiredUInt32LE(bytes, at: headerRange.lowerBound)
            let commandSize = Int(
                try requiredUInt32LE(
                    bytes,
                    at: headerRange.lowerBound + 4
                )
            )
            guard commandSize >= 8,
                  commandSize.isMultiple(of: 8),
                  let commandRange = checkedRange(
                      offset: cursor,
                      length: commandSize,
                      limit: range.upperBound
                  )
            else {
                throw SyntheticMachOIdentityFailure
                    .malformedLoadCommand
            }

            switch command {
            case lcUUID:
                guard commandSize == 24 else {
                    throw SyntheticMachOIdentityFailure
                        .malformedLoadCommand
                }
                guard uuid == nil else {
                    throw SyntheticMachOIdentityFailure.duplicateUUID
                }
                let value = Data(bytes[(cursor + 8)..<(cursor + 24)])
                guard value.contains(where: { $0 != 0 }) else {
                    throw SyntheticMachOIdentityFailure.zeroUUID
                }
                uuid = value

            case lcCodeSignature:
                guard commandSize == 16 else {
                    throw SyntheticMachOIdentityFailure
                        .malformedLoadCommand
                }
                guard codeSignatureRange == nil else {
                    throw SyntheticMachOIdentityFailure
                        .duplicateCodeSignature
                }
                let signatureOffset = Int(
                    try requiredUInt32LE(bytes, at: cursor + 8)
                )
                let signatureBytes = Int(
                    try requiredUInt32LE(bytes, at: cursor + 12)
                )
                guard signatureBytes > 0,
                      signatureBytes <= maximumCodeSignatureBytes,
                      signatureOffset >= range.upperBound,
                      let signatureRange = checkedRange(
                          offset: signatureOffset,
                          length: signatureBytes,
                          limit: bytes.count
                      )
                else {
                    throw SyntheticMachOIdentityFailure
                        .codeSignatureBounds
                }
                codeSignatureRange = signatureRange

            case lcDyldEnvironment:
                throw SyntheticMachOIdentityFailure
                    .unsupportedDyldEnvironment

            default:
                break
            }
            cursor = commandRange.upperBound
        }

        guard cursor == range.upperBound else {
            throw SyntheticMachOIdentityFailure.malformedLoadCommand
        }
        guard let uuid else {
            throw SyntheticMachOIdentityFailure.missingUUID
        }
        guard let codeSignatureRange else {
            throw SyntheticMachOIdentityFailure.missingCodeSignature
        }
        return LoadCommandFacts(
            uuid: uuid,
            codeSignatureRange: codeSignatureRange
        )
    }

    private struct EmbeddedSignatureFacts {
        let codeDirectories: [SyntheticCodeDirectoryComparison]
        let cmsBlob: Data?
        let isAdHoc: Bool
    }

    private struct BlobIndex {
        let tablePosition: Int
        let slot: UInt32
        let offset: Int
    }

    private struct BlobDescriptor {
        let tablePosition: Int
        let slot: UInt32
        let range: Range<Int>
        let magic: UInt32
        let bytes: Data
    }

    private static func parseEmbeddedSignature(
        _ signature: Data
    ) throws -> EmbeddedSignatureFacts {
        guard signature.count >= 12,
              readUInt32BE(signature, at: 0)
                == csMagicEmbeddedSignature,
              readUInt32BE(signature, at: 4)
                == UInt32(signature.count)
        else {
            throw SyntheticMachOIdentityFailure.superBlobBounds
        }
        let count = try requiredUInt32BE(signature, at: 8)
        guard count > 0, count <= maximumSuperBlobEntries else {
            throw SyntheticMachOIdentityFailure.superBlobEntryCount
        }
        guard let indexBytes = checkedMultiply(Int(count), 8),
              let indexRange = checkedRange(
                  offset: 12,
                  length: indexBytes,
                  limit: signature.count
              )
        else {
            throw SyntheticMachOIdentityFailure.superBlobBounds
        }

        var indexes: [BlobIndex] = []
        indexes.reserveCapacity(Int(count))
        var seenSlots = Set<UInt32>()
        for position in 0..<Int(count) {
            let entryOffset = 12 + position * 8
            let slot = try requiredUInt32BE(
                signature,
                at: entryOffset
            )
            guard seenSlots.insert(slot).inserted else {
                throw SyntheticMachOIdentityFailure
                    .duplicateSuperBlobSlot
            }
            indexes.append(
                BlobIndex(
                    tablePosition: position,
                    slot: slot,
                    offset: Int(
                        try requiredUInt32BE(
                            signature,
                            at: entryOffset + 4
                        )
                    )
                )
            )
        }

        let sortedIndexes = indexes.sorted {
            if $0.offset == $1.offset {
                return $0.tablePosition < $1.tablePosition
            }
            return $0.offset < $1.offset
        }
        var descriptors: [BlobDescriptor] = []
        descriptors.reserveCapacity(sortedIndexes.count)
        var previous: BlobDescriptor?

        for index in sortedIndexes {
            guard index.offset >= indexRange.upperBound else {
                throw SyntheticMachOIdentityFailure.blobBounds
            }
            if let previous {
                if index.offset < previous.range.upperBound {
                    throw SyntheticMachOIdentityFailure
                        .overlappingBlobRanges
                }
                if index.offset > previous.range.upperBound {
                    if isCodeDirectorySlot(previous.slot) {
                        throw SyntheticMachOIdentityFailure
                            .codeDirectoryBounds
                    }
                    throw SyntheticMachOIdentityFailure.blobBounds
                }
            } else if index.offset != indexRange.upperBound {
                throw SyntheticMachOIdentityFailure.blobBounds
            }
            guard checkedRange(
                offset: index.offset,
                length: 8,
                limit: signature.count
            ) != nil else {
                throw SyntheticMachOIdentityFailure.blobBounds
            }
            let magic = try requiredUInt32BE(
                signature,
                at: index.offset
            )
            let length = Int(
                try requiredUInt32BE(
                    signature,
                    at: index.offset + 4
                )
            )
            guard length >= 8,
                  let blobRange = checkedRange(
                      offset: index.offset,
                      length: length,
                      limit: signature.count
                  )
            else {
                throw SyntheticMachOIdentityFailure.blobBounds
            }
            let descriptor = BlobDescriptor(
                tablePosition: index.tablePosition,
                slot: index.slot,
                range: blobRange,
                magic: magic,
                bytes: Data(signature[blobRange])
            )
            descriptors.append(descriptor)
            previous = descriptor
        }
        guard descriptors.last?.range.upperBound == signature.count else {
            if let finalDescriptor = descriptors.last,
               isCodeDirectorySlot(finalDescriptor.slot)
            {
                throw SyntheticMachOIdentityFailure
                    .codeDirectoryBounds
            }
            throw SyntheticMachOIdentityFailure.blobBounds
        }

        let descriptorsByPosition = descriptors.sorted {
            $0.tablePosition < $1.tablePosition
        }
        let codeDirectoryDescriptors = descriptorsByPosition.filter {
            isCodeDirectorySlot($0.slot) ||
                $0.magic == csMagicCodeDirectory
        }
        guard codeDirectoryDescriptors.contains(where: {
            $0.slot == primaryCodeDirectorySlot
        }) else {
            throw SyntheticMachOIdentityFailure
                .missingPrimaryCodeDirectory
        }
        for descriptor in codeDirectoryDescriptors {
            guard isCodeDirectorySlot(descriptor.slot) else {
                throw SyntheticMachOIdentityFailure
                    .unsupportedCodeDirectorySlot
            }
        }
        let directorySlots = codeDirectoryDescriptors.map(\.slot)
        guard directorySlots.first == primaryCodeDirectorySlot,
              directorySlots.count <= 6,
              zip(
                  directorySlots.dropFirst(),
                  directorySlots.dropFirst().dropFirst()
              ).allSatisfy(<)
        else {
            throw SyntheticMachOIdentityFailure.codeDirectoryOrder
        }

        var codeDirectories: [SyntheticCodeDirectoryComparison] = []
        var cmsBlob: Data?
        for descriptor in descriptorsByPosition {
            if isCodeDirectorySlot(descriptor.slot) {
                guard descriptor.magic == csMagicCodeDirectory else {
                    throw SyntheticMachOIdentityFailure
                        .codeDirectoryBounds
                }
                codeDirectories.append(
                    try parseCodeDirectory(
                        descriptor.bytes,
                        slot: descriptor.slot
                    )
                )
            } else if descriptor.slot == cmsSlot {
                guard descriptor.magic == csMagicBlobWrapper else {
                    throw SyntheticMachOIdentityFailure.cmsBlob
                }
                cmsBlob = descriptor.bytes
            } else if descriptor.magic == csMagicBlobWrapper {
                throw SyntheticMachOIdentityFailure.cmsBlob
            }
        }

        guard let primary = codeDirectories.first else {
            throw SyntheticMachOIdentityFailure
                .missingPrimaryCodeDirectory
        }
        let isAdHoc = primary.flags & csAdHoc != 0
        guard codeDirectories.allSatisfy({
            ($0.flags & csAdHoc != 0) == isAdHoc
        }),
            isAdHoc == (cmsBlob == nil)
        else {
            throw SyntheticMachOIdentityFailure
                .cmsAdHocInconsistency
        }
        return EmbeddedSignatureFacts(
            codeDirectories: codeDirectories,
            cmsBlob: cmsBlob,
            isAdHoc: isAdHoc
        )
    }

    /// Reuses the exact committed embedded-signature formulation for the
    /// separately sealed file-type parsers. This does not select a file type,
    /// artifact role, path, or operational authority.
    static func parseEmbeddedSignatureForFileTypeIdentity(
        _ signature: Data
    ) throws -> (
        codeDirectories: [SyntheticCodeDirectoryComparison],
        cmsBlob: Data?,
        isAdHoc: Bool
    ) {
        let facts = try parseEmbeddedSignature(signature)
        return (
            codeDirectories: facts.codeDirectories,
            cmsBlob: facts.cmsBlob,
            isAdHoc: facts.isAdHoc
        )
    }

    private static func parseCodeDirectory(
        _ blob: Data,
        slot: UInt32
    ) throws -> SyntheticCodeDirectoryComparison {
        guard blob.count >= 44,
              blob.count <= maximumCodeDirectoryBytes,
              readUInt32BE(blob, at: 0) == csMagicCodeDirectory,
              readUInt32BE(blob, at: 4) == UInt32(blob.count)
        else {
            throw SyntheticMachOIdentityFailure.codeDirectoryBounds
        }
        let version = try requiredUInt32BE(blob, at: 8)
        guard version >= 0x20001, version <= 0x20600 else {
            throw SyntheticMachOIdentityFailure
                .unsupportedCodeDirectoryVersion
        }
        let fixedBytes = fixedCodeDirectoryBytes(version: version)
        guard blob.count >= fixedBytes else {
            throw SyntheticMachOIdentityFailure.codeDirectoryBounds
        }

        let flags = try requiredUInt32BE(blob, at: 12)
        let hashOffset = Int(try requiredUInt32BE(blob, at: 16))
        let identifierOffset = Int(
            try requiredUInt32BE(blob, at: 20)
        )
        let specialSlots = Int(try requiredUInt32BE(blob, at: 24))
        let codeSlots = Int(try requiredUInt32BE(blob, at: 28))
        let hashSize = blob[36]
        let hashType = blob[37]
        let expectedHashSize: UInt8
        switch hashType {
        case 1:
            expectedHashSize = 20
        case 2:
            expectedHashSize = 32
        case 3:
            expectedHashSize = 20
        case 4:
            expectedHashSize = 48
        default:
            throw SyntheticMachOIdentityFailure
                .unsupportedCodeDirectoryHashType
        }
        guard hashSize == expectedHashSize else {
            throw SyntheticMachOIdentityFailure.codeDirectoryHashSize
        }
        guard try requiredUInt32BE(blob, at: 40) == 0 else {
            throw SyntheticMachOIdentityFailure.codeDirectoryBounds
        }
        // Optional offset-bearing layouts are intentionally outside this
        // narrow parser. Refuse them rather than partially validating a newer
        // CodeDirectory structure.
        if version >= 0x20100 {
            guard try requiredUInt32BE(blob, at: 44) == 0 else {
                throw SyntheticMachOIdentityFailure.codeDirectoryBounds
            }
        }
        if version >= 0x20300 {
            guard try requiredUInt32BE(blob, at: 52) == 0 else {
                throw SyntheticMachOIdentityFailure.codeDirectoryBounds
            }
        }
        if version >= 0x20500 {
            guard try requiredUInt32BE(blob, at: 92) == 0 else {
                throw SyntheticMachOIdentityFailure.codeDirectoryBounds
            }
        }
        if version >= 0x20600 {
            guard blob[96] == 0,
                  blob[97] == 0,
                  blob[98] == 0,
                  blob[99] == 0,
                  try requiredUInt32BE(blob, at: 100) == 0,
                  try requiredUInt32BE(blob, at: 104) == 0
            else {
                throw SyntheticMachOIdentityFailure.codeDirectoryBounds
            }
        }

        guard let specialHashBytes = checkedMultiply(
            specialSlots,
            Int(hashSize)
        ),
            let codeHashBytes = checkedMultiply(
                codeSlots,
                Int(hashSize)
            ),
            hashOffset >= specialHashBytes,
            hashOffset <= blob.count,
            let codeHashRange = checkedRange(
                offset: hashOffset,
                length: codeHashBytes,
                limit: blob.count
            ),
            codeHashRange.upperBound == blob.count
        else {
            throw SyntheticMachOIdentityFailure.codeDirectoryBounds
        }
        let metadataLimit = hashOffset - specialHashBytes
        let signingIdentifier = try terminatedIdentity(
            blob,
            offset: identifierOffset,
            fixedBytes: fixedBytes,
            limit: metadataLimit,
            emptyAllowed: false,
            failure: .signingIdentifier
        )

        let teamOffset: Int
        if version >= 0x20200 {
            teamOffset = Int(try requiredUInt32BE(blob, at: 48))
        } else {
            teamOffset = 0
        }
        let teamIdentifier: Data
        if teamOffset == 0 {
            teamIdentifier = Data()
        } else {
            teamIdentifier = try terminatedIdentity(
                blob,
                offset: teamOffset,
                fixedBytes: fixedBytes,
                limit: metadataLimit,
                emptyAllowed: false,
                failure: .teamIdentifier
            )
            let signingRange = identityStorageRange(
                offset: identifierOffset,
                value: signingIdentifier
            )
            let teamRange = identityStorageRange(
                offset: teamOffset,
                value: teamIdentifier
            )
            guard !signingRange.overlaps(teamRange) else {
                throw SyntheticMachOIdentityFailure.teamIdentifier
            }
        }

        return SyntheticCodeDirectoryComparison(
            slot: slot,
            blob: blob,
            blobSHA256: sha256Hex(blob),
            version: version,
            flags: flags,
            hashType: hashType,
            hashSize: hashSize,
            signingIdentifier: signingIdentifier,
            teamIdentifier: teamIdentifier
        )
    }

    private static func terminatedIdentity(
        _ bytes: Data,
        offset: Int,
        fixedBytes: Int,
        limit: Int,
        emptyAllowed: Bool,
        failure: SyntheticMachOIdentityFailure
    ) throws -> Data {
        guard offset >= fixedBytes,
              offset < limit,
              let scanRange = checkedRange(
                  offset: offset,
                  length: limit - offset,
                  limit: bytes.count
              ),
              let terminator = bytes[scanRange].firstIndex(of: 0)
        else {
            throw failure
        }
        let length = terminator - offset
        guard length <= maximumIdentityBytes,
              emptyAllowed || length > 0
        else {
            throw failure
        }
        return Data(bytes[offset..<terminator])
    }

    private static func identityStorageRange(
        offset: Int,
        value: Data
    ) -> Range<Int> {
        offset..<(offset + value.count + 1)
    }

    private static func fixedCodeDirectoryBytes(version: UInt32) -> Int {
        switch version {
        case ..<0x20100:
            return 44
        case ..<0x20200:
            return 48
        case ..<0x20300:
            return 52
        case ..<0x20400:
            return 64
        case ..<0x20500:
            return 88
        case ..<0x20600:
            return 96
        default:
            return 108
        }
    }

    private static func isCodeDirectorySlot(_ slot: UInt32) -> Bool {
        slot == primaryCodeDirectorySlot ||
            (
                slot >= firstAlternateCodeDirectorySlot &&
                    slot <= lastAlternateCodeDirectorySlot
            )
    }

    private static func requiredUInt32LE(
        _ bytes: Data,
        at offset: Int
    ) throws -> UInt32 {
        guard let value = readUInt32LE(bytes, at: offset) else {
            throw SyntheticMachOIdentityFailure.fileBounds
        }
        return value
    }

    private static func requiredUInt32BE(
        _ bytes: Data,
        at offset: Int
    ) throws -> UInt32 {
        guard let value = readUInt32BE(bytes, at: offset) else {
            throw SyntheticMachOIdentityFailure.blobBounds
        }
        return value
    }

    private static func readUInt32LE(
        _ bytes: Data,
        at offset: Int
    ) -> UInt32? {
        guard let range = checkedRange(
            offset: offset,
            length: 4,
            limit: bytes.count
        ) else {
            return nil
        }
        return UInt32(bytes[range.lowerBound]) |
            UInt32(bytes[range.lowerBound + 1]) << 8 |
            UInt32(bytes[range.lowerBound + 2]) << 16 |
            UInt32(bytes[range.lowerBound + 3]) << 24
    }

    private static func readUInt32BE(
        _ bytes: Data,
        at offset: Int
    ) -> UInt32? {
        guard let range = checkedRange(
            offset: offset,
            length: 4,
            limit: bytes.count
        ) else {
            return nil
        }
        return UInt32(bytes[range.lowerBound]) << 24 |
            UInt32(bytes[range.lowerBound + 1]) << 16 |
            UInt32(bytes[range.lowerBound + 2]) << 8 |
            UInt32(bytes[range.lowerBound + 3])
    }

    private static func checkedRange(
        offset: Int,
        length: Int,
        limit: Int
    ) -> Range<Int>? {
        guard offset >= 0, length >= 0, offset <= limit else {
            return nil
        }
        let (end, overflow) = offset.addingReportingOverflow(length)
        guard !overflow, end <= limit else {
            return nil
        }
        return offset..<end
    }

    private static func checkedMultiply(
        _ lhs: Int,
        _ rhs: Int
    ) -> Int? {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        return overflow ? nil : value
    }

    private static func sha256Hex(_ bytes: Data) -> String {
        SHA256.hash(data: bytes)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}
