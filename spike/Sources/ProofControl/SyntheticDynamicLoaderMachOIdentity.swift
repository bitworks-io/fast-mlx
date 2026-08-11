import CryptoKit
import Foundation

enum SyntheticDynamicLoaderForbiddenCommand: Equatable, Sendable {
    case dyldEnvironment
    case main
    case loadDynamicLinker
    case encryptionInfo
    case encryptionInfo64
    case fileSetEntry
    case idDylib
    case loadDylib
    case loadWeakDylib
    case reexportDylib
    case lazyLoadDylib
    case loadUpwardDylib
    case rpath
}

enum SyntheticDynamicLoaderIdentityField: Equatable, Sendable {
    case commandSize
    case nameOffset
    case prefixPadding
    case terminator
    case length
    case trailingPadding
    case canonicalName
}

/// Bounded synthetic `MH_DYLINKER` facts only. This value is comparison
/// evidence, not a file locator, launch decision, or runtime admission.
struct SyntheticDynamicLoaderMachOIdentityComparison: Equatable {
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
    let dynamicLoaderIDName: Data
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

enum SyntheticDynamicLoaderMachOIdentityParser {
    static let maximumFileBytes =
        SyntheticMachOIdentityParser.maximumFileBytes
    static let maximumLoadCommandCount =
        SyntheticMachOIdentityParser.maximumLoadCommandCount
    static let maximumLoadCommandBytes =
        SyntheticMachOIdentityParser.maximumLoadCommandBytes

    static let recognizedLoadCommandValueCount =
        recognizedLoadCommands.count

    private static let maximumCodeSignatureBytes = 262_144
    private static let maximumIdentityBytes = 4_096

    private static let machHeader64Bytes = 32
    private static let machMagic64: UInt32 = 0xfeedfacf
    private static let cpuTypeARM64: UInt32 = 0x0100000c
    private static let machDynamicLinker: UInt32 = 0x7
    private static let lcIDDynamicLinker: UInt32 = 0x0f
    private static let lcUUID: UInt32 = 0x1b
    private static let lcCodeSignature: UInt32 = 0x1d
    private static let requiredByDynamicLinkerBit: UInt32 = 0x80000000

    private static let recognizedLoadCommands: Set<UInt32> = [
        0x00000001, 0x00000002, 0x00000003, 0x00000004,
        0x00000005, 0x00000006, 0x00000007, 0x00000008,
        0x00000009, 0x0000000a, 0x0000000b, 0x0000000c,
        0x0000000d, 0x0000000e, 0x0000000f, 0x00000010,
        0x00000011, 0x00000012, 0x00000013, 0x00000014,
        0x00000015, 0x00000016, 0x00000017, 0x80000018,
        0x00000019, 0x0000001a, 0x0000001b, 0x8000001c,
        0x0000001d, 0x0000001e, 0x8000001f, 0x00000020,
        0x00000021, 0x00000022, 0x80000022, 0x80000023,
        0x00000024, 0x00000025, 0x00000026, 0x00000027,
        0x80000028, 0x00000029, 0x0000002a, 0x0000002b,
        0x0000002c, 0x0000002d, 0x0000002e, 0x0000002f,
        0x00000030, 0x00000031, 0x00000032, 0x80000033,
        0x80000034, 0x80000035,
    ]

    static func parse(_ input: Data) throws
        -> SyntheticDynamicLoaderMachOIdentityComparison
    {
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
        guard fileType == machDynamicLinker else {
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
        guard commandFacts.signatureBytes > 0,
              commandFacts.signatureBytes <= maximumCodeSignatureBytes,
              commandFacts.signatureOffset >= loadCommandRange.upperBound,
              let codeSignatureRange = checkedRange(
                  offset: commandFacts.signatureOffset,
                  length: commandFacts.signatureBytes,
                  limit: bytes.count
              )
        else {
            throw SyntheticMachOIdentityFailure.codeSignatureBounds
        }

        let signatureRegion = Data(bytes[codeSignatureRange])
        let signatureFacts = try SyntheticMachOIdentityParser
            .parseEmbeddedSignatureForFileTypeIdentity(signatureRegion)
        let loadCommands = Data(bytes[loadCommandRange])

        return SyntheticDynamicLoaderMachOIdentityComparison(
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
            dynamicLoaderIDName: commandFacts.dynamicLoaderIDName,
            codeSignatureRegion: signatureRegion,
            codeSignatureRegionSHA256: sha256Hex(signatureRegion),
            codeDirectories: signatureFacts.codeDirectories,
            cmsBlob: signatureFacts.cmsBlob,
            cmsBlobSHA256: signatureFacts.cmsBlob.map(sha256Hex)
                ?? String(repeating: "0", count: 64),
            isAdHoc: signatureFacts.isAdHoc
        )
    }
}

private extension SyntheticDynamicLoaderMachOIdentityParser {
    struct LoadCommandFrame {
        let ordinal: UInt64
        let command: UInt32
        let range: Range<Int>
    }

    struct LoadCommandFacts {
        let uuid: Data
        let dynamicLoaderIDName: Data
        let signatureOffset: Int
        let signatureBytes: Int
    }

    static func parseLoadCommands(
        _ bytes: Data,
        range: Range<Int>,
        expectedCount: UInt32
    ) throws -> LoadCommandFacts {
        let frames = try scanLoadCommandFrames(
            bytes,
            range: range,
            expectedCount: expectedCount
        )

        for frame in frames {
            guard recognizedLoadCommands.contains(frame.command) else {
                if frame.command & requiredByDynamicLinkerBit != 0 {
                    throw SyntheticMachOIdentityFailure
                        .unknownRequiredDynamicLoaderLoadCommand(
                            ordinal: frame.ordinal,
                            command: frame.command
                        )
                }
                throw SyntheticMachOIdentityFailure
                    .unknownOptionalDynamicLoaderLoadCommand(
                        ordinal: frame.ordinal,
                        command: frame.command
                    )
            }
        }
        for frame in frames {
            if let forbidden = forbiddenKind(frame.command) {
                throw SyntheticMachOIdentityFailure
                    .forbiddenDynamicLoaderLoadCommand(
                        ordinal: frame.ordinal,
                        kind: forbidden
                    )
            }
        }

        var uuid: Data?
        var dynamicLoaderIDName: Data?
        var signatureOffset: Int?
        var signatureBytes: Int?

        for frame in frames {
            let cursor = frame.range.lowerBound
            let commandSize = frame.range.count
            switch frame.command {
            case lcUUID:
                guard commandSize == 24 else {
                    throw SyntheticMachOIdentityFailure
                        .malformedDynamicLoaderLoadCommand(
                            ordinal: frame.ordinal
                        )
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
                        .malformedDynamicLoaderLoadCommand(
                            ordinal: frame.ordinal
                        )
                }
                guard signatureOffset == nil else {
                    throw SyntheticMachOIdentityFailure
                        .duplicateCodeSignature
                }
                signatureOffset = Int(
                    try requiredUInt32LE(bytes, at: cursor + 8)
                )
                signatureBytes = Int(
                    try requiredUInt32LE(bytes, at: cursor + 12)
                )

            case lcIDDynamicLinker:
                guard dynamicLoaderIDName == nil else {
                    throw SyntheticMachOIdentityFailure
                        .duplicateDynamicLoaderIdentity(
                            ordinal: frame.ordinal
                        )
                }
                dynamicLoaderIDName = try parseDynamicLoaderIdentity(
                    bytes,
                    range: frame.range,
                    ordinal: frame.ordinal
                )

            default:
                break
            }
        }

        guard let uuid else {
            throw SyntheticMachOIdentityFailure.missingUUID
        }
        guard let signatureOffset, let signatureBytes else {
            throw SyntheticMachOIdentityFailure.missingCodeSignature
        }
        guard let dynamicLoaderIDName else {
            throw SyntheticMachOIdentityFailure
                .missingDynamicLoaderIdentity
        }
        return LoadCommandFacts(
            uuid: uuid,
            dynamicLoaderIDName: dynamicLoaderIDName,
            signatureOffset: signatureOffset,
            signatureBytes: signatureBytes
        )
    }

    static func scanLoadCommandFrames(
        _ bytes: Data,
        range: Range<Int>,
        expectedCount: UInt32
    ) throws -> [LoadCommandFrame] {
        var cursor = range.lowerBound
        var frames: [LoadCommandFrame] = []
        frames.reserveCapacity(Int(expectedCount))

        for ordinal in 0..<expectedCount {
            let boundedOrdinal = UInt64(ordinal)
            guard checkedRange(
                offset: cursor,
                length: 8,
                limit: range.upperBound
            ) != nil,
                let command = readUInt32LE(bytes, at: cursor),
                let rawCommandSize = readUInt32LE(bytes, at: cursor + 4)
            else {
                throw SyntheticMachOIdentityFailure
                    .malformedDynamicLoaderLoadCommand(
                        ordinal: boundedOrdinal
                    )
            }
            let commandSize = Int(rawCommandSize)
            guard commandSize >= 8,
                  commandSize.isMultiple(of: 8),
                  let commandRange = checkedRange(
                      offset: cursor,
                      length: commandSize,
                      limit: range.upperBound
                  )
            else {
                throw SyntheticMachOIdentityFailure
                    .malformedDynamicLoaderLoadCommand(
                        ordinal: boundedOrdinal
                    )
            }
            frames.append(
                LoadCommandFrame(
                    ordinal: boundedOrdinal,
                    command: command,
                    range: commandRange
                )
            )
            cursor = commandRange.upperBound
        }

        guard cursor == range.upperBound else {
            throw SyntheticMachOIdentityFailure
                .malformedDynamicLoaderLoadCommand(
                    ordinal: UInt64(expectedCount)
                )
        }
        return frames
    }

    static func parseDynamicLoaderIdentity(
        _ bytes: Data,
        range: Range<Int>,
        ordinal: UInt64
    ) throws -> Data {
        guard range.count >= 16 else {
            throw identityFailure(ordinal, .commandSize)
        }
        let nameOffset = Int(
            try requiredUInt32LE(bytes, at: range.lowerBound + 8)
        )
        guard nameOffset >= 12, nameOffset < range.count else {
            throw identityFailure(ordinal, .nameOffset)
        }
        let nameStart = range.lowerBound + nameOffset
        if bytes[(range.lowerBound + 12)..<nameStart]
            .contains(where: { $0 != 0 })
        {
            throw identityFailure(ordinal, .prefixPadding)
        }
        guard let terminator = bytes[nameStart..<range.upperBound]
            .firstIndex(of: 0)
        else {
            throw identityFailure(ordinal, .terminator)
        }
        let nameLength = terminator - nameStart
        guard nameLength >= 2, nameLength <= maximumIdentityBytes else {
            throw identityFailure(ordinal, .length)
        }
        let trailingStart = terminator + 1
        if bytes[trailingStart..<range.upperBound]
            .contains(where: { $0 != 0 })
        {
            throw identityFailure(ordinal, .trailingPadding)
        }
        let name = Data(bytes[nameStart..<terminator])
        guard isCanonicalAbsoluteInstallName(name) else {
            throw identityFailure(ordinal, .canonicalName)
        }
        return name
    }

    static func identityFailure(
        _ ordinal: UInt64,
        _ field: SyntheticDynamicLoaderIdentityField
    ) -> SyntheticMachOIdentityFailure {
        .dynamicLoaderIdentityLayout(
            ordinal: ordinal,
            field: field
        )
    }

    static func isCanonicalAbsoluteInstallName(_ name: Data) -> Bool {
        guard name.count >= 2, name.first == 0x2f else {
            return false
        }
        var componentStart = 1
        for index in 1..<name.count {
            let byte = name[index]
            guard byte >= 0x21, byte <= 0x7e,
                  byte != 0x40,
                  byte != 0x5c
            else {
                return false
            }
            if byte == 0x2f {
                guard validComponent(
                    name[componentStart..<index]
                ) else {
                    return false
                }
                componentStart = index + 1
            }
        }
        return validComponent(name[componentStart..<name.count])
    }

    static func validComponent(
        _ component: Data.SubSequence
    ) -> Bool {
        guard !component.isEmpty else {
            return false
        }
        if component.count == 1, component.first == 0x2e {
            return false
        }
        if component.count == 2,
           component[component.startIndex] == 0x2e,
           component[component.index(after: component.startIndex)] == 0x2e
        {
            return false
        }
        return true
    }

    static func forbiddenKind(
        _ command: UInt32
    ) -> SyntheticDynamicLoaderForbiddenCommand? {
        switch command {
        case 0x00000027:
            return .dyldEnvironment
        case 0x80000028:
            return .main
        case 0x0000000e:
            return .loadDynamicLinker
        case 0x00000021:
            return .encryptionInfo
        case 0x0000002c:
            return .encryptionInfo64
        case 0x80000035:
            return .fileSetEntry
        case 0x0000000d:
            return .idDylib
        case 0x0000000c:
            return .loadDylib
        case 0x80000018:
            return .loadWeakDylib
        case 0x8000001f:
            return .reexportDylib
        case 0x00000020:
            return .lazyLoadDylib
        case 0x80000023:
            return .loadUpwardDylib
        case 0x8000001c:
            return .rpath
        default:
            return nil
        }
    }

    static func requiredUInt32LE(
        _ bytes: Data,
        at offset: Int
    ) throws -> UInt32 {
        guard let value = readUInt32LE(bytes, at: offset) else {
            throw SyntheticMachOIdentityFailure.fileBounds
        }
        return value
    }

    static func readUInt32LE(
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

    static func checkedRange(
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

    static func sha256Hex(_ bytes: Data) -> String {
        SHA256.hash(data: bytes)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}
