import Foundation

/// Closed, comparison-only source profile for the reviewed dyld-1378 layout.
/// It carries no locator, host-build assertion, or derived identity.
enum SyntheticSharedCacheSourceProfile: Equatable, Sendable {
    struct Layout: Equatable, Sendable {
        let headerBytes = 552
        let mappingOffsetOffset = 16
        let mappingCountOffset = 20
        let codeSignatureOffsetOffset = 40
        let uuidOffset = 88
        let cacheTypeOffset = 104
        let platformOffset = 216
        let formatFlagsOffset = 220
        let mappingWithSlideOffsetOffset = 312
        let subCacheArrayOffsetOffset = 392
        let subCacheArrayCountOffset = 396
        let symbolFileUUIDOffset = 400
        let imagesOffsetOffset = 448
        let imagesCountOffset = 452
        let cacheSubTypeOffset = 456
        let legacySubcacheEntryBytes = 24
        let modernSubcacheEntryBytes = 56
        let modernSubcacheUUIDOffset = 0
        let modernSubcacheVMOffsetOffset = 16
        let modernSubcacheSuffixOffset = 24
        let mappingInfoBytes = 32
        let mappingAndSlideInfoBytes = 56
        let imageInfoBytes = 32
        let imageTextInfoBytes = 32
    }

    struct Policy: Equatable, Sendable {
        let maximumFileCount = 64
        let maximumSubcacheCount = 63
        let maximumDiscoveryBytes = 1_048_576
        let maximumFileBytes: UInt64 = 17_179_869_184
        let maximumAggregateBytes: UInt64 = 68_719_476_736
        let maximumMappingCount: UInt32 = 8
    }

    case dyld1378ModernArm64MacOS

    static let reviewed = Self.dyld1378ModernArm64MacOS
    static let dyld1378Arm64MacOS = Self.reviewed

    var tag: String { "dyld-1378" }
    var tagObject: String { "79fdd08288695d6dbbdac41f022a2851d20fa647" }
    var commit: String { "fd8d0c4d52320ebf64db34f3cb280310d905c5ae" }
    var layout: Layout { Layout() }
    var policy: Policy { Policy() }

    var dyldCacheFormatHeaderSHA256: String {
        "dd6f7d9ffc5cb318988c16dbecf958d04b0c65cd9ee1892a838e374f76fd182c"
    }
    var dyldSharedCacheImplementationSHA256: String {
        "2edb158b41203e595b1d95937acad429888ce1464f25b7b620cb9b19f38e6475"
    }
    var dyldSharedCacheHeaderSHA256: String {
        "881a5f2e174f458f5c05c2be04bd4f7eee943d4a3950e3ecd58e26e94b8b743f"
    }
    var sharedCacheRuntimeSHA256: String {
        "5edf7835d47f6b78110da10d556cd45e2ac04c04a9d1627294522dc4aca058be"
    }
    var subCacheBuilderSHA256: String {
        "d4f06686b932360c7d93f5afda8b2e8eb50d6bdcf742dc2a431f0b91c0967b8a"
    }
    var newSharedCacheBuilderSHA256: String {
        "58d1f19d122c16a04b04f27fb7336cb41afc9dfa5ed5cb28c22a93fc9d6d48c5"
    }
}

/// Value-only synthetic metadata and discovery-byte input. The verifier takes
/// its own bounded byte snapshot before parsing or retaining any facts.
struct SyntheticSharedCacheDiscoveryFile: Equatable, Sendable {
    let metadata: SyntheticCaptureFileMetadata
    let discoveryBytes: Data

    init(
        metadata: SyntheticCaptureFileMetadata,
        discoveryBytes: Data
    ) {
        self.metadata = metadata
        // Avoid allocating another potentially unbounded buffer before the
        // verifier has enforced the discovery-byte ceiling.
        self.discoveryBytes = discoveryBytes
    }
}

/// Immutable, all-files comparison facts. This value has no ID, locator,
/// descriptor, reopen surface, parser conversion, or operational authority.
struct SyntheticSharedCacheSetPlanComparison: Equatable, Sendable {
    struct MappingFact: Equatable, Sendable {
        let row: Int
        let address: UInt64
        let size: UInt64
        let fileOffset: UInt64
        let maximumProtection: UInt32
        let initialProtection: UInt32
        let slideInfoOffset: UInt64
        let slideInfoSize: UInt64
        let flags: UInt64
    }

    struct HeaderFacts: Equatable, Sendable {
        let magic: Data
        let mappingOffset: UInt32
        let mappingCount: UInt32
        let mappingWithSlideOffset: UInt32
        let mappingWithSlideCount: UInt32
        let mappingTablesEnd: UInt64
        let codeSignatureOffset: UInt64
        let codeSignatureSize: UInt64
        let sharedRegionStart: UInt64
        let sharedRegionSize: UInt64
        let imagesOffset: UInt64
        let imagesCount: UInt32
        let imagesTextOffset: UInt64
        let imagesTextCount: UInt64
        let subCacheArrayOffset: UInt32
        let subCacheArrayCount: UInt32
        let subcacheTableEnd: UInt64
        let symbolFileUUID: String
        let headerUUID: String
        let platform: UInt32
        let formatVersion: UInt32
        let simulator: Bool
        let locallyBuiltCache: Bool
        let newFormatTLVs: Bool
        let padding: UInt32
        let cacheType: UInt64
        let cacheSubType: UInt32
        let mappings: [MappingFact]
    }

    struct HeaderOrderFile: Equatable, Sendable {
        let ordinal: Int
        let metadata: SyntheticCaptureFileMetadata
        let fileBytes: UInt64
        let decodedSuffix: Data
        let suffixByteCount: UInt64
        let suffixBase64URL: String
        let headerUUID: String
        let cacheVMOffset: UInt64
        let header: HeaderFacts
        let discoveryBytes: Data

        var suffixBytes: Data { Data(decodedSuffix) }
    }

    fileprivate enum ConstructionSeal: Equatable, Sendable {
        case verified
    }

    let sourceProfile: SyntheticSharedCacheSourceProfile
    let mainMetadata: SyntheticCaptureFileMetadata
    let mainHeader: HeaderFacts
    let headerOrderFiles: [HeaderOrderFile]
    let aggregateFileBytes: UInt64
    let runtimeDecision = RuntimeClosureExpectationRuntimeDecision.noGo
    fileprivate let constructionSeal: ConstructionSeal

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

    var commonHeader: HeaderFacts { mainHeader }
    var fileFacts: [HeaderOrderFile] { headerOrderFiles }
    var mainDiscoveryBytes: Data {
        Data(headerOrderFiles[0].discoveryBytes)
    }
    var subcacheDiscoveryBytes: [Data] {
        headerOrderFiles.dropFirst().map { Data($0.discoveryBytes) }
    }

    fileprivate init(
        sourceProfile: SyntheticSharedCacheSourceProfile,
        mainMetadata: SyntheticCaptureFileMetadata,
        mainHeader: HeaderFacts,
        headerOrderFiles: [HeaderOrderFile],
        aggregateFileBytes: UInt64,
        seal: ConstructionSeal
    ) {
        self.sourceProfile = sourceProfile
        self.mainMetadata = mainMetadata
        self.mainHeader = Self.copy(header: mainHeader)
        self.headerOrderFiles = headerOrderFiles.map(Self.copy(file:))
        self.aggregateFileBytes = aggregateFileBytes
        self.constructionSeal = seal
    }

    private static func copy(header: HeaderFacts) -> HeaderFacts {
        HeaderFacts(
            magic: Data(header.magic),
            mappingOffset: header.mappingOffset,
            mappingCount: header.mappingCount,
            mappingWithSlideOffset: header.mappingWithSlideOffset,
            mappingWithSlideCount: header.mappingWithSlideCount,
            mappingTablesEnd: header.mappingTablesEnd,
            codeSignatureOffset: header.codeSignatureOffset,
            codeSignatureSize: header.codeSignatureSize,
            sharedRegionStart: header.sharedRegionStart,
            sharedRegionSize: header.sharedRegionSize,
            imagesOffset: header.imagesOffset,
            imagesCount: header.imagesCount,
            imagesTextOffset: header.imagesTextOffset,
            imagesTextCount: header.imagesTextCount,
            subCacheArrayOffset: header.subCacheArrayOffset,
            subCacheArrayCount: header.subCacheArrayCount,
            subcacheTableEnd: header.subcacheTableEnd,
            symbolFileUUID: header.symbolFileUUID,
            headerUUID: header.headerUUID,
            platform: header.platform,
            formatVersion: header.formatVersion,
            simulator: header.simulator,
            locallyBuiltCache: header.locallyBuiltCache,
            newFormatTLVs: header.newFormatTLVs,
            padding: header.padding,
            cacheType: header.cacheType,
            cacheSubType: header.cacheSubType,
            mappings: Array(header.mappings)
        )
    }

    private static func copy(file: HeaderOrderFile) -> HeaderOrderFile {
        HeaderOrderFile(
            ordinal: file.ordinal,
            metadata: file.metadata,
            fileBytes: file.fileBytes,
            decodedSuffix: Data(file.decodedSuffix),
            suffixByteCount: file.suffixByteCount,
            suffixBase64URL: file.suffixBase64URL,
            headerUUID: file.headerUUID,
            cacheVMOffset: file.cacheVMOffset,
            header: copy(header: file.header),
            discoveryBytes: Data(file.discoveryBytes)
        )
    }
}

enum SyntheticSharedCacheSetPlanFailure: Error, Equatable, Sendable {
    enum ResourceEnvelopeReason: Equatable, Sendable {
        case constantMismatch
        case arithmeticOverflow
        case integerConversion
    }

    enum DiscoveryBytesReason: Equatable, Sendable {
        case minimumHeader(actual: Int, minimum: Int)
        case ceiling(actual: Int, maximum: Int)
        case exactLength(expected: UInt64, actual: Int)
        case tableBounds
    }

    enum CacheFormatField: Equatable, Sendable {
        case platform
        case formatVersion
        case simulator
        case locallyBuiltCache
        case newFormatTLVs
        case padding
        case cacheType
        case cacheSubType
    }

    enum CacheMappingReason: Equatable, Sendable {
        case mappingOffset
        case mappingCount
        case mappingWithSlideOffset
        case mappingWithSlideCount
        case tableBounds
        case rowMismatch(row: Int)
        case zeroSize(row: Int)
        case firstFileOffset(UInt64)
        case vmOrder(row: Int)
        case fileContiguity(row: Int)
        case fileRange(row: Int)
        case slideInfoRange(row: Int)
        case unknownFlags(row: Int, flags: UInt64)
    }

    enum CacheHeaderReason: Equatable, Sendable {
        case imagesOffset
        case imagesTextOffset
        case imagesTextCount
        case subcacheArrayOffset
        case subcacheArrayCount
        case tableArithmetic
        case codeSignatureSize
        case codeSignatureEnd(expected: UInt64, actual: UInt64)
        case sharedRegion
        case nestedImageTable
        case nestedSubcacheTable
    }

    enum CacheSetCountReason: Equatable, Sendable {
        case fileCount(actual: Int, minimum: Int, maximum: Int)
        case suppliedSubcaches(expected: Int, actual: Int)
    }

    enum CacheEntryReason: Equatable, Sendable {
        case tableBounds(entryIndex: Int)
    }

    enum CacheSuffixReason: Equatable, Sendable {
        case missingTerminator
        case decodedLength(Int)
        case nonzeroPadding(index: Int)
        case nonPrintable(index: Int)
        case firstByte
        case forbiddenByte(index: Int)
        case dotComponent
        case duplicate(previousEntryIndex: Int)
    }

    enum CacheUUIDReason: Equatable, Sendable {
        case zero
        case duplicate(previousOrdinal: Int)
        case entryHeaderMismatch(entryOrdinal: Int)
    }

    enum CacheVMOffsetReason: Equatable, Sendable {
        case zero
        case nonIncreasing(previous: UInt64, actual: UInt64)
        case sharedRegionOverflow
        case sharedRegionStartMismatch(expected: UInt64, actual: UInt64)
        case firstMappingMismatch(expected: UInt64, actual: UInt64)
    }

    enum AggregateSizeReason: Equatable, Sendable {
        case invalidFileSize(fileOrdinal: Int, actual: Int64)
        case arithmeticOverflow
        case setLimit(actual: UInt64, maximum: UInt64)
    }

    enum CacheSubcacheReason: Equatable, Sendable {
        case globalMappingOrder(fileOrdinal: Int, row: Int)
        case globalMappingOverlap(fileOrdinal: Int, row: Int)
        case finalMappingBeyondMainSharedRegion(fileOrdinal: Int, row: Int)
    }

    case sourceProfile
    case resourceEnvelope(ResourceEnvelopeReason)
    case metadata(fileOrdinal: Int, field: SyntheticSmallArtifactCaptureFailure.MetadataField)
    case fileType(fileOrdinal: Int, mode: UInt32)
    case linkCount(fileOrdinal: Int, actual: UInt64)
    case fileSize(fileOrdinal: Int, actual: Int64)
    case dataless(fileOrdinal: Int)
    case sparseStateUnavailable(fileOrdinal: Int)
    case sparseFile(fileOrdinal: Int)
    case discoveryBytes(fileOrdinal: Int, reason: DiscoveryBytesReason)
    case cacheMagic(fileOrdinal: Int)
    case cacheFormat(fileOrdinal: Int, field: CacheFormatField)
    case cacheMapping(fileOrdinal: Int, reason: CacheMappingReason)
    case cacheHeader(fileOrdinal: Int, reason: CacheHeaderReason)
    case cacheSetCount(CacheSetCountReason)
    case cacheEntry(entryIndex: Int, reason: CacheEntryReason)
    case cacheSuffix(entryIndex: Int, reason: CacheSuffixReason)
    case cacheUUID(fileOrdinal: Int, reason: CacheUUIDReason)
    case cacheVMOffset(fileOrdinal: Int, reason: CacheVMOffsetReason)
    case aggregateSize(AggregateSizeReason)
    case cacheSubcache(fileOrdinal: Int, reason: CacheSubcacheReason)
}

enum SyntheticSharedCacheSetPlanVerifier {
    static let sourceTag = "dyld-1378"
    static let sourceTagObject = "79fdd08288695d6dbbdac41f022a2851d20fa647"
    static let sourceCommit = "fd8d0c4d52320ebf64db34f3cb280310d905c5ae"

    static let maximumCacheFileCount = 64
    static let maximumSubcacheFileCount = 63
    static let maximumDiscoveryBytes = 1_048_576
    static let maximumCacheFileBytes: UInt64 = 17_179_869_184
    static let maximumCacheSetBytes: UInt64 = 68_719_476_736

    static let dyldCacheHeaderBytes = 552
    static let mappingInfoBytes = 32
    static let mappingAndSlideInfoBytes = 56
    static let modernSubcacheEntryBytes = 56
    static let imageInfoBytes = 32
    static let imageTextInfoBytes = 32
    static let minimumMappingCount: UInt32 = 1
    static let maximumMappingCount: UInt32 = 8

    static let mappingOffsetFieldOffset = 16
    static let mappingCountFieldOffset = 20
    static let codeSignatureOffsetFieldOffset = 40
    static let uuidFieldOffset = 88
    static let cacheTypeFieldOffset = 104
    static let imagesTextOffsetFieldOffset = 136
    static let imagesTextCountFieldOffset = 144
    static let platformFieldOffset = 216
    static let formatFlagsFieldOffset = 220
    static let sharedRegionStartFieldOffset = 224
    static let sharedRegionSizeFieldOffset = 232
    static let mappingWithSlideOffsetFieldOffset = 312
    static let mappingWithSlideCountFieldOffset = 316
    static let subCacheArrayOffsetFieldOffset = 392
    static let subCacheArrayCountFieldOffset = 396
    static let symbolFileUUIDFieldOffset = 400
    static let imagesOffsetFieldOffset = 448
    static let imagesCountFieldOffset = 452
    static let cacheSubTypeFieldOffset = 456

    static let modernSubcacheUUIDOffset = 0
    static let modernSubcacheVMOffset = 16
    static let modernSubcacheSuffixOffset = 24

    static let currentMagic = Data([
        0x64, 0x79, 0x6c, 0x64, 0x5f, 0x76, 0x31, 0x20,
        0x20, 0x20, 0x61, 0x72, 0x6d, 0x36, 0x34, 0x00,
    ])

    static let expectedPlatform: UInt32 = 1
    static let expectedCacheType: UInt64 = 2
    static let expectedCacheSubType: UInt32 = 1
    static let expectedFormatVersion: UInt32 = 0
    static let simulatorFlag: UInt32 = 1 << 9
    static let locallyBuiltFlag: UInt32 = 1 << 10
    static let newFormatTLVsFlag: UInt32 = 1 << 12
    static let knownFormatFlags: UInt32 = newFormatTLVsFlag
    static let allowedMappingFlags: UInt64 = 0x7F

    static func compare(
        sourceProfile: SyntheticSharedCacheSourceProfile,
        main: SyntheticSharedCacheDiscoveryFile,
        subcaches inputSubcaches: [SyntheticSharedCacheDiscoveryFile]
    ) throws -> SyntheticSharedCacheSetPlanComparison {
        guard sourceProfile == .reviewed,
              sourceProfile.tag == sourceTag,
              sourceProfile.tagObject == sourceTagObject,
              sourceProfile.commit == sourceCommit else {
            throw SyntheticSharedCacheSetPlanFailure.sourceProfile
        }
        try validateFixedConstants()

        let (fileCount, fileCountOverflow) = inputSubcaches.count
            .addingReportingOverflow(1)
        guard !fileCountOverflow,
              (1...maximumCacheFileCount).contains(fileCount) else {
            throw SyntheticSharedCacheSetPlanFailure.cacheSetCount(
                .fileCount(
                    actual: fileCountOverflow ? Int.max : fileCount,
                    minimum: 1,
                    maximum: maximumCacheFileCount
                )
            )
        }

        try validateMetadata(main.metadata, fileOrdinal: 0)
        let mainDiscoveryByteCount = try validateDiscoveryEnvelope(
            main.discoveryBytes,
            fileOrdinal: 0
        )
        let mainDiscoveryBytes = try ownedCopy(
            main.discoveryBytes,
            expectedCount: mainDiscoveryByteCount,
            fileOrdinal: 0
        )
        let mainRaw = try parseRawHeader(mainDiscoveryBytes, fileOrdinal: 0)
        try validateProfile(mainRaw, fileOrdinal: 0)
        let mainMappings = try validateMappings(
            mainRaw,
            bytes: mainDiscoveryBytes,
            fileOrdinal: 0
        )
        let mainTableEnd = try validateMainTables(
            mainRaw,
            bytes: mainDiscoveryBytes,
            metadata: main.metadata,
            fileOrdinal: 0
        )
        try validateIdentityAndRegion(
            mainRaw,
            mappings: mainMappings,
            metadata: main.metadata,
            fileOrdinal: 0
        )

        guard inputSubcaches.count == Int(mainRaw.subCacheArrayCount) else {
            throw SyntheticSharedCacheSetPlanFailure.cacheSetCount(
                .suppliedSubcaches(
                    expected: Int(mainRaw.subCacheArrayCount),
                    actual: inputSubcaches.count
                )
            )
        }

        let entries = try parseEntries(
            bytes: mainDiscoveryBytes,
            tableOffset: mainRaw.subCacheArrayOffset,
            count: mainRaw.subCacheArrayCount,
            mainUUID: mainRaw.uuid
        )
        let aggregateFileBytes = try checkedAggregateBytes(
            main: main.metadata,
            subcaches: inputSubcaches
        )

        for (index, file) in inputSubcaches.enumerated() {
            try validateMetadata(file.metadata, fileOrdinal: index + 1)
        }

        var parsedSubcaches: [(RawHeader, [ParsedMapping])] = []
        var subcacheDiscoveryBytes: [Data] = []
        parsedSubcaches.reserveCapacity(inputSubcaches.count)
        subcacheDiscoveryBytes.reserveCapacity(inputSubcaches.count)
        for (index, file) in inputSubcaches.enumerated() {
            let ordinal = index + 1
            let discoveryByteCount = try validateDiscoveryEnvelope(
                file.discoveryBytes,
                fileOrdinal: ordinal
            )
            let discoveryBytes = try ownedCopy(
                file.discoveryBytes,
                expectedCount: discoveryByteCount,
                fileOrdinal: ordinal
            )
            let raw = try parseRawHeader(
                discoveryBytes,
                fileOrdinal: ordinal
            )
            try validateProfile(raw, fileOrdinal: ordinal)
            let mappings = try validateMappings(
                raw,
                bytes: discoveryBytes,
                fileOrdinal: ordinal
            )
            guard UInt64(discoveryBytes.count) == raw.mappingTablesEnd else {
                throw SyntheticSharedCacheSetPlanFailure.discoveryBytes(
                    fileOrdinal: ordinal,
                    reason: .exactLength(
                        expected: raw.mappingTablesEnd,
                        actual: discoveryBytes.count
                    )
                )
            }
            try validateCodeSignature(
                raw,
                metadata: file.metadata,
                fileOrdinal: ordinal
            )
            parsedSubcaches.append((raw, mappings))
            subcacheDiscoveryBytes.append(discoveryBytes)
        }

        for index in entries.indices {
            let ordinal = index + 1
            let entry = entries[index]
            let raw = parsedSubcaches[index].0
            let mappings = parsedSubcaches[index].1
            guard raw.uuid == entry.uuid else {
                throw SyntheticSharedCacheSetPlanFailure.cacheUUID(
                    fileOrdinal: ordinal,
                    reason: .entryHeaderMismatch(entryOrdinal: ordinal)
                )
            }
            let (expectedStart, overflow) = mainRaw.sharedRegionStart
                .addingReportingOverflow(entry.cacheVMOffset)
            guard !overflow else {
                throw SyntheticSharedCacheSetPlanFailure.cacheVMOffset(
                    fileOrdinal: ordinal,
                    reason: .sharedRegionOverflow
                )
            }
            guard raw.sharedRegionStart == expectedStart else {
                throw SyntheticSharedCacheSetPlanFailure.cacheVMOffset(
                    fileOrdinal: ordinal,
                    reason: .sharedRegionStartMismatch(
                        expected: expectedStart,
                        actual: raw.sharedRegionStart
                    )
                )
            }
            guard let first = mappings.first,
                  first.address == expectedStart else {
                throw SyntheticSharedCacheSetPlanFailure.cacheVMOffset(
                    fileOrdinal: ordinal,
                    reason: .firstMappingMismatch(
                        expected: expectedStart,
                        actual: mappings.first?.address ?? 0
                    )
                )
            }
        }

        for (index, parsed) in parsedSubcaches.enumerated() {
            let raw = parsed.0
            let ordinal = index + 1
            guard raw.subCacheArrayOffset == 0,
                  raw.subCacheArrayCount == 0 else {
                throw SyntheticSharedCacheSetPlanFailure.cacheHeader(
                    fileOrdinal: ordinal,
                    reason: .nestedSubcacheTable
                )
            }
            guard raw.imagesOffset == 0,
                  raw.imagesCount == 0,
                  raw.imagesTextOffset == 0,
                  raw.imagesTextCount == 0 else {
                throw SyntheticSharedCacheSetPlanFailure.cacheHeader(
                    fileOrdinal: ordinal,
                    reason: .nestedImageTable
                )
            }
        }

        try validateGlobalMappings(
            main: mainRaw,
            mainMappings: mainMappings,
            subcaches: parsedSubcaches
        )

        let mainHeader = headerFacts(
            raw: mainRaw,
            mappings: mainMappings,
            subcacheTableEnd: mainTableEnd
        )
        var files: [SyntheticSharedCacheSetPlanComparison.HeaderOrderFile] = [
            fileFact(
                ordinal: 0,
                metadata: main.metadata,
                suffix: Data(),
                uuid: mainRaw.uuid,
                cacheVMOffset: 0,
                raw: mainRaw,
                mappings: mainMappings,
                tableEnd: mainTableEnd,
                discoveryBytes: mainDiscoveryBytes
            ),
        ]
        files.reserveCapacity(fileCount)
        for index in entries.indices {
            let raw = parsedSubcaches[index].0
            let mappings = parsedSubcaches[index].1
            files.append(
                fileFact(
                    ordinal: index + 1,
                    metadata: inputSubcaches[index].metadata,
                    suffix: entries[index].suffix,
                    uuid: raw.uuid,
                    cacheVMOffset: entries[index].cacheVMOffset,
                    raw: raw,
                    mappings: mappings,
                    tableEnd: raw.mappingTablesEnd,
                    discoveryBytes: subcacheDiscoveryBytes[index]
                )
            )
        }

        return SyntheticSharedCacheSetPlanComparison(
            sourceProfile: sourceProfile,
            mainMetadata: main.metadata,
            mainHeader: mainHeader,
            headerOrderFiles: files,
            aggregateFileBytes: aggregateFileBytes,
            seal: .verified
        )
    }
}

private extension SyntheticSharedCacheSetPlanVerifier {
    struct RawHeader {
        let magic: Data
        let mappingOffset: UInt32
        let mappingCount: UInt32
        let codeSignatureOffset: UInt64
        let codeSignatureSize: UInt64
        let uuid: Data
        let cacheType: UInt64
        let imagesTextOffset: UInt64
        let imagesTextCount: UInt64
        let platform: UInt32
        let formatFlags: UInt32
        let sharedRegionStart: UInt64
        let sharedRegionSize: UInt64
        let mappingWithSlideOffset: UInt32
        let mappingWithSlideCount: UInt32
        let subCacheArrayOffset: UInt32
        let subCacheArrayCount: UInt32
        let symbolFileUUID: Data
        let imagesOffset: UInt64
        let imagesCount: UInt32
        let cacheSubType: UInt32
        let mappingTablesEnd: UInt64
    }

    struct ParsedMapping {
        let row: Int
        let address: UInt64
        let size: UInt64
        let fileOffset: UInt64
        let maximumProtection: UInt32
        let initialProtection: UInt32
        let slideInfoOffset: UInt64
        let slideInfoSize: UInt64
        let flags: UInt64
    }

    struct ParsedEntry {
        let uuid: Data
        let cacheVMOffset: UInt64
        let suffix: Data
    }

    static func validateFixedConstants() throws {
        guard dyldCacheHeaderBytes == 552,
              mappingInfoBytes == 32,
              mappingAndSlideInfoBytes == 56,
              modernSubcacheEntryBytes == 56,
              imageInfoBytes == 32,
              imageTextInfoBytes == 32,
              maximumCacheFileCount == 64,
              maximumSubcacheFileCount == 63,
              maximumDiscoveryBytes == 1_048_576,
              maximumCacheFileBytes == 17_179_869_184,
              maximumCacheSetBytes == 68_719_476_736 else {
            throw SyntheticSharedCacheSetPlanFailure.resourceEnvelope(
                .constantMismatch
            )
        }
    }

    static func validateMetadata(
        _ metadata: SyntheticCaptureFileMetadata,
        fileOrdinal: Int
    ) throws {
        guard metadata.mode & SyntheticSmallArtifactCaptureVerifier.regularFileTypeMask ==
                SyntheticSmallArtifactCaptureVerifier.regularFileType else {
            throw SyntheticSharedCacheSetPlanFailure.fileType(
                fileOrdinal: fileOrdinal,
                mode: metadata.mode
            )
        }
        guard metadata.linkCount == 1 else {
            throw SyntheticSharedCacheSetPlanFailure.linkCount(
                fileOrdinal: fileOrdinal,
                actual: metadata.linkCount
            )
        }
        guard metadata.size > 0,
              let fileBytes = UInt64(exactly: metadata.size),
              fileBytes <= maximumCacheFileBytes else {
            throw SyntheticSharedCacheSetPlanFailure.fileSize(
                fileOrdinal: fileOrdinal,
                actual: metadata.size
            )
        }
        guard metadata.flags & SyntheticSmallArtifactCaptureVerifier.datalessFlag == 0 else {
            throw SyntheticSharedCacheSetPlanFailure.dataless(
                fileOrdinal: fileOrdinal
            )
        }
        guard metadata.extendedAttributeSupportMask &
                SyntheticSmallArtifactCaptureVerifier.requiredSparseStateSupportBit != 0 else {
            throw SyntheticSharedCacheSetPlanFailure.sparseStateUnavailable(
                fileOrdinal: fileOrdinal
            )
        }
        guard metadata.extendedFlags &
                SyntheticSmallArtifactCaptureVerifier.sparseFlag == 0 else {
            throw SyntheticSharedCacheSetPlanFailure.sparseFile(
                fileOrdinal: fileOrdinal
            )
        }
    }

    static func validateDiscoveryEnvelope(
        _ bytes: Data,
        fileOrdinal: Int
    ) throws -> Int {
        let checkedCount = bytes.count
        guard checkedCount <= maximumDiscoveryBytes else {
            throw SyntheticSharedCacheSetPlanFailure.discoveryBytes(
                fileOrdinal: fileOrdinal,
                reason: .ceiling(
                    actual: checkedCount,
                    maximum: maximumDiscoveryBytes
                )
            )
        }
        guard checkedCount >= dyldCacheHeaderBytes else {
            throw SyntheticSharedCacheSetPlanFailure.discoveryBytes(
                fileOrdinal: fileOrdinal,
                reason: .minimumHeader(
                    actual: checkedCount,
                    minimum: dyldCacheHeaderBytes
                )
            )
        }
        return checkedCount
    }

    static func parseRawHeader(
        _ bytes: Data,
        fileOrdinal: Int
    ) throws -> RawHeader {
        guard let magic = readData(bytes, at: 0, count: 16),
              let mappingOffset = readUInt32(bytes, at: mappingOffsetFieldOffset),
              let mappingCount = readUInt32(bytes, at: mappingCountFieldOffset),
              let codeSignatureOffset = readUInt64(bytes, at: codeSignatureOffsetFieldOffset),
              let codeSignatureSize = readUInt64(bytes, at: codeSignatureOffsetFieldOffset + 8),
              let uuid = readData(bytes, at: uuidFieldOffset, count: 16),
              let cacheType = readUInt64(bytes, at: cacheTypeFieldOffset),
              let imagesTextOffset = readUInt64(bytes, at: imagesTextOffsetFieldOffset),
              let imagesTextCount = readUInt64(bytes, at: imagesTextCountFieldOffset),
              let platform = readUInt32(bytes, at: platformFieldOffset),
              let formatFlags = readUInt32(bytes, at: formatFlagsFieldOffset),
              let sharedRegionStart = readUInt64(bytes, at: sharedRegionStartFieldOffset),
              let sharedRegionSize = readUInt64(bytes, at: sharedRegionSizeFieldOffset),
              let mappingWithSlideOffset = readUInt32(bytes, at: mappingWithSlideOffsetFieldOffset),
              let mappingWithSlideCount = readUInt32(bytes, at: mappingWithSlideCountFieldOffset),
              let subCacheArrayOffset = readUInt32(bytes, at: subCacheArrayOffsetFieldOffset),
              let subCacheArrayCount = readUInt32(bytes, at: subCacheArrayCountFieldOffset),
              let symbolFileUUID = readData(bytes, at: symbolFileUUIDFieldOffset, count: 16),
              let imagesOffset = readUInt32(bytes, at: imagesOffsetFieldOffset),
              let imagesCount = readUInt32(bytes, at: imagesCountFieldOffset),
              let cacheSubType = readUInt32(bytes, at: cacheSubTypeFieldOffset) else {
            throw SyntheticSharedCacheSetPlanFailure.discoveryBytes(
                fileOrdinal: fileOrdinal,
                reason: .tableBounds
            )
        }
        let mappingBytes = try checkedMultiply(
            UInt64(mappingCount),
            UInt64(mappingAndSlideInfoBytes)
        )
        let expectedSlideOffset = try checkedAdd(
            UInt64(dyldCacheHeaderBytes),
            mappingBytes
        )
        let mappingTablesEnd = try checkedAdd(expectedSlideOffset, mappingBytes)
        return RawHeader(
            magic: magic,
            mappingOffset: mappingOffset,
            mappingCount: mappingCount,
            codeSignatureOffset: codeSignatureOffset,
            codeSignatureSize: codeSignatureSize,
            uuid: uuid,
            cacheType: cacheType,
            imagesTextOffset: imagesTextOffset,
            imagesTextCount: imagesTextCount,
            platform: platform,
            formatFlags: formatFlags,
            sharedRegionStart: sharedRegionStart,
            sharedRegionSize: sharedRegionSize,
            mappingWithSlideOffset: mappingWithSlideOffset,
            mappingWithSlideCount: mappingWithSlideCount,
            subCacheArrayOffset: subCacheArrayOffset,
            subCacheArrayCount: subCacheArrayCount,
            symbolFileUUID: symbolFileUUID,
            imagesOffset: UInt64(imagesOffset),
            imagesCount: imagesCount,
            cacheSubType: cacheSubType,
            mappingTablesEnd: mappingTablesEnd
        )
    }

    static func validateProfile(
        _ raw: RawHeader,
        fileOrdinal: Int
    ) throws {
        guard raw.magic == currentMagic else {
            throw SyntheticSharedCacheSetPlanFailure.cacheMagic(
                fileOrdinal: fileOrdinal
            )
        }
        guard raw.platform == expectedPlatform else {
            throw SyntheticSharedCacheSetPlanFailure.cacheFormat(
                fileOrdinal: fileOrdinal,
                field: .platform
            )
        }
        let formatVersion = raw.formatFlags & 0xFF
        guard formatVersion == expectedFormatVersion else {
            throw SyntheticSharedCacheSetPlanFailure.cacheFormat(
                fileOrdinal: fileOrdinal,
                field: .formatVersion
            )
        }
        guard raw.formatFlags & simulatorFlag == 0 else {
            throw SyntheticSharedCacheSetPlanFailure.cacheFormat(
                fileOrdinal: fileOrdinal,
                field: .simulator
            )
        }
        guard raw.formatFlags & locallyBuiltFlag == 0 else {
            throw SyntheticSharedCacheSetPlanFailure.cacheFormat(
                fileOrdinal: fileOrdinal,
                field: .locallyBuiltCache
            )
        }
        guard raw.formatFlags & newFormatTLVsFlag != 0 else {
            throw SyntheticSharedCacheSetPlanFailure.cacheFormat(
                fileOrdinal: fileOrdinal,
                field: .newFormatTLVs
            )
        }
        guard raw.formatFlags & ~knownFormatFlags == 0 else {
            throw SyntheticSharedCacheSetPlanFailure.cacheFormat(
                fileOrdinal: fileOrdinal,
                field: .padding
            )
        }
        guard raw.cacheType == expectedCacheType else {
            throw SyntheticSharedCacheSetPlanFailure.cacheFormat(
                fileOrdinal: fileOrdinal,
                field: .cacheType
            )
        }
        guard raw.cacheSubType == expectedCacheSubType else {
            throw SyntheticSharedCacheSetPlanFailure.cacheFormat(
                fileOrdinal: fileOrdinal,
                field: .cacheSubType
            )
        }
    }

    static func validateMappings(
        _ raw: RawHeader,
        bytes: Data,
        fileOrdinal: Int
    ) throws -> [ParsedMapping] {
        guard raw.mappingOffset == UInt32(dyldCacheHeaderBytes) else {
            throw SyntheticSharedCacheSetPlanFailure.cacheMapping(
                fileOrdinal: fileOrdinal,
                reason: .mappingOffset
            )
        }
        guard (minimumMappingCount...maximumMappingCount)
            .contains(raw.mappingCount) else {
            throw SyntheticSharedCacheSetPlanFailure.cacheMapping(
                fileOrdinal: fileOrdinal,
                reason: .mappingCount
            )
        }
        let mappingBytes = try checkedMultiply(
            UInt64(raw.mappingCount),
            UInt64(mappingAndSlideInfoBytes)
        )
        let expectedSlideOffset = try checkedAdd(
            UInt64(dyldCacheHeaderBytes),
            mappingBytes
        )
        guard UInt64(raw.mappingWithSlideOffset) == expectedSlideOffset else {
            throw SyntheticSharedCacheSetPlanFailure.cacheMapping(
                fileOrdinal: fileOrdinal,
                reason: .mappingWithSlideOffset
            )
        }
        guard raw.mappingWithSlideCount == raw.mappingCount else {
            throw SyntheticSharedCacheSetPlanFailure.cacheMapping(
                fileOrdinal: fileOrdinal,
                reason: .mappingWithSlideCount
            )
        }
        guard raw.mappingTablesEnd <= UInt64(bytes.count),
              let legacyStart = Int(exactly: raw.mappingOffset),
              let slideStart = Int(exactly: raw.mappingWithSlideOffset) else {
            throw SyntheticSharedCacheSetPlanFailure.cacheMapping(
                fileOrdinal: fileOrdinal,
                reason: .tableBounds
            )
        }

        var mappings: [ParsedMapping] = []
        mappings.reserveCapacity(Int(raw.mappingCount))
        var previousVMEnd: UInt64?
        var previousFileEnd: UInt64?
        for row in 0..<Int(raw.mappingCount) {
            let legacy = legacyStart + row * mappingInfoBytes
            let slide = slideStart + row * mappingAndSlideInfoBytes
            guard let legacyAddress = readUInt64(bytes, at: legacy),
                  let legacySize = readUInt64(bytes, at: legacy + 8),
                  let legacyFileOffset = readUInt64(bytes, at: legacy + 16),
                  let legacyMaximumProtection = readUInt32(bytes, at: legacy + 24),
                  let legacyInitialProtection = readUInt32(bytes, at: legacy + 28),
                  let address = readUInt64(bytes, at: slide),
                  let size = readUInt64(bytes, at: slide + 8),
                  let fileOffset = readUInt64(bytes, at: slide + 16),
                  let slideInfoOffset = readUInt64(bytes, at: slide + 24),
                  let slideInfoSize = readUInt64(bytes, at: slide + 32),
                  let flags = readUInt64(bytes, at: slide + 40),
                  let maximumProtection = readUInt32(bytes, at: slide + 48),
                  let initialProtection = readUInt32(bytes, at: slide + 52) else {
                throw SyntheticSharedCacheSetPlanFailure.cacheMapping(
                    fileOrdinal: fileOrdinal,
                    reason: .tableBounds
                )
            }
            guard legacyAddress == address,
                  legacySize == size,
                  legacyFileOffset == fileOffset,
                  legacyMaximumProtection == maximumProtection,
                  legacyInitialProtection == initialProtection else {
                throw SyntheticSharedCacheSetPlanFailure.cacheMapping(
                    fileOrdinal: fileOrdinal,
                    reason: .rowMismatch(row: row)
                )
            }
            guard size > 0 else {
                throw SyntheticSharedCacheSetPlanFailure.cacheMapping(
                    fileOrdinal: fileOrdinal,
                    reason: .zeroSize(row: row)
                )
            }
            if row == 0, fileOffset != 0 {
                throw SyntheticSharedCacheSetPlanFailure.cacheMapping(
                    fileOrdinal: fileOrdinal,
                    reason: .firstFileOffset(fileOffset)
                )
            }
            let vmEnd = try checkedAdd(address, size)
            let fileEnd = try checkedAdd(fileOffset, size)
            if let previousVMEnd, address < previousVMEnd {
                throw SyntheticSharedCacheSetPlanFailure.cacheMapping(
                    fileOrdinal: fileOrdinal,
                    reason: .vmOrder(row: row)
                )
            }
            if let previousFileEnd, fileOffset != previousFileEnd {
                throw SyntheticSharedCacheSetPlanFailure.cacheMapping(
                    fileOrdinal: fileOrdinal,
                    reason: .fileContiguity(row: row)
                )
            }
            guard fileEnd <= raw.codeSignatureOffset else {
                throw SyntheticSharedCacheSetPlanFailure.cacheMapping(
                    fileOrdinal: fileOrdinal,
                    reason: .fileRange(row: row)
                )
            }
            if slideInfoOffset == 0 || slideInfoSize == 0 {
                guard slideInfoOffset == 0, slideInfoSize == 0 else {
                    throw SyntheticSharedCacheSetPlanFailure.cacheMapping(
                        fileOrdinal: fileOrdinal,
                        reason: .slideInfoRange(row: row)
                    )
                }
            } else {
                let slideEnd = try checkedAdd(slideInfoOffset, slideInfoSize)
                guard slideEnd <= raw.codeSignatureOffset else {
                    throw SyntheticSharedCacheSetPlanFailure.cacheMapping(
                        fileOrdinal: fileOrdinal,
                        reason: .slideInfoRange(row: row)
                    )
                }
            }
            guard flags & ~allowedMappingFlags == 0 else {
                throw SyntheticSharedCacheSetPlanFailure.cacheMapping(
                    fileOrdinal: fileOrdinal,
                    reason: .unknownFlags(row: row, flags: flags)
                )
            }
            mappings.append(
                ParsedMapping(
                    row: row,
                    address: address,
                    size: size,
                    fileOffset: fileOffset,
                    maximumProtection: maximumProtection,
                    initialProtection: initialProtection,
                    slideInfoOffset: slideInfoOffset,
                    slideInfoSize: slideInfoSize,
                    flags: flags
                )
            )
            previousVMEnd = vmEnd
            previousFileEnd = fileEnd
        }
        return mappings
    }

    static func validateMainTables(
        _ raw: RawHeader,
        bytes: Data,
        metadata: SyntheticCaptureFileMetadata,
        fileOrdinal: Int
    ) throws -> UInt64 {
        guard raw.imagesOffset == raw.mappingTablesEnd else {
            throw SyntheticSharedCacheSetPlanFailure.cacheHeader(
                fileOrdinal: fileOrdinal,
                reason: .imagesOffset
            )
        }
        let imageBytes = try checkedMultiply(
            UInt64(raw.imagesCount),
            UInt64(imageInfoBytes)
        )
        let expectedImagesTextOffset = try checkedAdd(
            raw.imagesOffset,
            imageBytes
        )
        guard raw.imagesTextOffset == expectedImagesTextOffset else {
            throw SyntheticSharedCacheSetPlanFailure.cacheHeader(
                fileOrdinal: fileOrdinal,
                reason: .imagesTextOffset
            )
        }
        guard raw.imagesTextCount == UInt64(raw.imagesCount) else {
            throw SyntheticSharedCacheSetPlanFailure.cacheHeader(
                fileOrdinal: fileOrdinal,
                reason: .imagesTextCount
            )
        }
        let imageTextBytes = try checkedMultiply(
            raw.imagesTextCount,
            UInt64(imageTextInfoBytes)
        )
        let expectedSubcacheOffset = try checkedAdd(
            raw.imagesTextOffset,
            imageTextBytes
        )
        guard UInt64(raw.subCacheArrayOffset) == expectedSubcacheOffset else {
            throw SyntheticSharedCacheSetPlanFailure.cacheHeader(
                fileOrdinal: fileOrdinal,
                reason: .subcacheArrayOffset
            )
        }
        guard raw.subCacheArrayCount <= UInt32(maximumSubcacheFileCount) else {
            throw SyntheticSharedCacheSetPlanFailure.cacheHeader(
                fileOrdinal: fileOrdinal,
                reason: .subcacheArrayCount
            )
        }
        let entryBytes = try checkedMultiply(
            UInt64(raw.subCacheArrayCount),
            UInt64(modernSubcacheEntryBytes)
        )
        let tableEnd = try checkedAdd(expectedSubcacheOffset, entryBytes)
        guard tableEnd <= UInt64(maximumDiscoveryBytes),
              let metadataBytes = UInt64(exactly: metadata.size),
              tableEnd <= metadataBytes else {
            throw SyntheticSharedCacheSetPlanFailure.cacheHeader(
                fileOrdinal: fileOrdinal,
                reason: .tableArithmetic
            )
        }
        guard UInt64(bytes.count) == tableEnd else {
            throw SyntheticSharedCacheSetPlanFailure.discoveryBytes(
                fileOrdinal: fileOrdinal,
                reason: .exactLength(
                    expected: tableEnd,
                    actual: bytes.count
                )
            )
        }
        return tableEnd
    }

    static func validateIdentityAndRegion(
        _ raw: RawHeader,
        mappings: [ParsedMapping],
        metadata: SyntheticCaptureFileMetadata,
        fileOrdinal: Int
    ) throws {
        guard raw.uuid.contains(where: { $0 != 0 }) else {
            throw SyntheticSharedCacheSetPlanFailure.cacheUUID(
                fileOrdinal: fileOrdinal,
                reason: .zero
            )
        }
        try validateCodeSignature(
            raw,
            metadata: metadata,
            fileOrdinal: fileOrdinal
        )
        guard raw.sharedRegionSize > 0,
              let first = mappings.first,
              first.address == raw.sharedRegionStart else {
            throw SyntheticSharedCacheSetPlanFailure.cacheHeader(
                fileOrdinal: fileOrdinal,
                reason: .sharedRegion
            )
        }
        let regionEnd = try checkedAdd(
            raw.sharedRegionStart,
            raw.sharedRegionSize
        )
        for mapping in mappings {
            let mappingEnd = try checkedAdd(mapping.address, mapping.size)
            guard mappingEnd <= regionEnd else {
                throw SyntheticSharedCacheSetPlanFailure.cacheHeader(
                    fileOrdinal: fileOrdinal,
                    reason: .sharedRegion
                )
            }
        }
    }

    static func validateCodeSignature(
        _ raw: RawHeader,
        metadata: SyntheticCaptureFileMetadata,
        fileOrdinal: Int
    ) throws {
        guard raw.codeSignatureSize > 0 else {
            throw SyntheticSharedCacheSetPlanFailure.cacheHeader(
                fileOrdinal: fileOrdinal,
                reason: .codeSignatureSize
            )
        }
        let end = try checkedAdd(
            raw.codeSignatureOffset,
            raw.codeSignatureSize
        )
        guard let metadataBytes = UInt64(exactly: metadata.size),
              end == metadataBytes else {
            throw SyntheticSharedCacheSetPlanFailure.cacheHeader(
                fileOrdinal: fileOrdinal,
                reason: .codeSignatureEnd(
                    expected: UInt64(exactly: metadata.size) ?? 0,
                    actual: end
                )
            )
        }
    }

    static func parseEntries(
        bytes: Data,
        tableOffset: UInt32,
        count: UInt32,
        mainUUID: Data
    ) throws -> [ParsedEntry] {
        guard mainUUID.contains(where: { $0 != 0 }) else {
            throw SyntheticSharedCacheSetPlanFailure.cacheUUID(
                fileOrdinal: 0,
                reason: .zero
            )
        }
        var seenUUIDs: [Data: Int] = [mainUUID: 0]
        var seenSuffixes: [Data: Int] = [:]
        var entries: [ParsedEntry] = []
        entries.reserveCapacity(Int(count))
        var previousVMOffset: UInt64 = 0
        guard let start = Int(exactly: tableOffset) else {
            throw SyntheticSharedCacheSetPlanFailure.resourceEnvelope(
                .integerConversion
            )
        }
        for index in 0..<Int(count) {
            let offset = start + index * modernSubcacheEntryBytes
            guard let uuid = readData(bytes, at: offset, count: 16),
                  let vmOffset = readUInt64(bytes, at: offset + 16),
                  let suffixField = readData(bytes, at: offset + 24, count: 32) else {
                throw SyntheticSharedCacheSetPlanFailure.cacheEntry(
                    entryIndex: index,
                    reason: .tableBounds(entryIndex: index)
                )
            }
            guard uuid.contains(where: { $0 != 0 }) else {
                throw SyntheticSharedCacheSetPlanFailure.cacheUUID(
                    fileOrdinal: index + 1,
                    reason: .zero
                )
            }
            if let prior = seenUUIDs[uuid] {
                throw SyntheticSharedCacheSetPlanFailure.cacheUUID(
                    fileOrdinal: index + 1,
                    reason: .duplicate(previousOrdinal: prior)
                )
            }
            guard vmOffset > 0 else {
                throw SyntheticSharedCacheSetPlanFailure.cacheVMOffset(
                    fileOrdinal: index + 1,
                    reason: .zero
                )
            }
            guard vmOffset > previousVMOffset else {
                throw SyntheticSharedCacheSetPlanFailure.cacheVMOffset(
                    fileOrdinal: index + 1,
                    reason: .nonIncreasing(
                        previous: previousVMOffset,
                        actual: vmOffset
                    )
                )
            }
            let suffix = try validateSuffix(suffixField, entryIndex: index)
            if let prior = seenSuffixes[suffix] {
                throw SyntheticSharedCacheSetPlanFailure.cacheSuffix(
                    entryIndex: index,
                    reason: .duplicate(previousEntryIndex: prior)
                )
            }
            seenUUIDs[uuid] = index + 1
            seenSuffixes[suffix] = index
            previousVMOffset = vmOffset
            entries.append(
                ParsedEntry(
                    uuid: uuid,
                    cacheVMOffset: vmOffset,
                    suffix: suffix
                )
            )
        }
        return entries
    }

    static func validateSuffix(
        _ field: Data,
        entryIndex: Int
    ) throws -> Data {
        guard let terminator = field.firstIndex(of: 0) else {
            throw SyntheticSharedCacheSetPlanFailure.cacheSuffix(
                entryIndex: entryIndex,
                reason: .missingTerminator
            )
        }
        let suffix = Data(field[..<terminator])
        guard (1...31).contains(suffix.count) else {
            throw SyntheticSharedCacheSetPlanFailure.cacheSuffix(
                entryIndex: entryIndex,
                reason: .decodedLength(suffix.count)
            )
        }
        for index in field.indices where index > terminator {
            guard field[index] == 0 else {
                throw SyntheticSharedCacheSetPlanFailure.cacheSuffix(
                    entryIndex: entryIndex,
                    reason: .nonzeroPadding(index: index)
                )
            }
        }
        guard suffix.first == UInt8(ascii: ".") else {
            throw SyntheticSharedCacheSetPlanFailure.cacheSuffix(
                entryIndex: entryIndex,
                reason: .firstByte
            )
        }
        for (index, byte) in suffix.enumerated() {
            guard (0x21...0x7E).contains(byte) else {
                throw SyntheticSharedCacheSetPlanFailure.cacheSuffix(
                    entryIndex: entryIndex,
                    reason: .nonPrintable(index: index)
                )
            }
            guard byte != UInt8(ascii: "/"),
                  byte != UInt8(ascii: "\\"),
                  byte != UInt8(ascii: "@") else {
                throw SyntheticSharedCacheSetPlanFailure.cacheSuffix(
                    entryIndex: entryIndex,
                    reason: .forbiddenByte(index: index)
                )
            }
        }
        if suffix == Data(".".utf8) || suffix == Data("..".utf8) {
            throw SyntheticSharedCacheSetPlanFailure.cacheSuffix(
                entryIndex: entryIndex,
                reason: .dotComponent
            )
        }
        return suffix
    }

    static func checkedAggregateBytes(
        main: SyntheticCaptureFileMetadata,
        subcaches: [SyntheticSharedCacheDiscoveryFile]
    ) throws -> UInt64 {
        let metadata = [main] + subcaches.map(\.metadata)
        var total: UInt64 = 0
        for (ordinal, item) in metadata.enumerated() {
            guard let fileBytes = UInt64(exactly: item.size) else {
                throw SyntheticSharedCacheSetPlanFailure.aggregateSize(
                    .invalidFileSize(
                        fileOrdinal: ordinal,
                        actual: item.size
                    )
                )
            }
            let (next, overflow) = total.addingReportingOverflow(fileBytes)
            guard !overflow else {
                throw SyntheticSharedCacheSetPlanFailure.aggregateSize(
                    .arithmeticOverflow
                )
            }
            total = next
        }
        guard total <= maximumCacheSetBytes else {
            throw SyntheticSharedCacheSetPlanFailure.aggregateSize(
                .setLimit(actual: total, maximum: maximumCacheSetBytes)
            )
        }
        return total
    }

    static func validateGlobalMappings(
        main: RawHeader,
        mainMappings: [ParsedMapping],
        subcaches: [(RawHeader, [ParsedMapping])]
    ) throws {
        let mainEnd = try checkedAdd(
            main.sharedRegionStart,
            main.sharedRegionSize
        )
        var previousAddress: UInt64?
        var previousEnd: UInt64?
        let all = [(main, mainMappings)] + subcaches
        for (fileOrdinal, item) in all.enumerated() {
            for mapping in item.1 {
                let end = try checkedAdd(mapping.address, mapping.size)
                if let previousAddress, mapping.address < previousAddress {
                    throw SyntheticSharedCacheSetPlanFailure.cacheSubcache(
                        fileOrdinal: fileOrdinal,
                        reason: .globalMappingOrder(
                            fileOrdinal: fileOrdinal,
                            row: mapping.row
                        )
                    )
                }
                if let previousEnd, mapping.address < previousEnd {
                    throw SyntheticSharedCacheSetPlanFailure.cacheSubcache(
                        fileOrdinal: fileOrdinal,
                        reason: .globalMappingOverlap(
                            fileOrdinal: fileOrdinal,
                            row: mapping.row
                        )
                    )
                }
                guard end <= mainEnd else {
                    throw SyntheticSharedCacheSetPlanFailure.cacheSubcache(
                        fileOrdinal: fileOrdinal,
                        reason: .finalMappingBeyondMainSharedRegion(
                            fileOrdinal: fileOrdinal,
                            row: mapping.row
                        )
                    )
                }
                previousAddress = mapping.address
                previousEnd = end
            }
        }
    }

    static func headerFacts(
        raw: RawHeader,
        mappings: [ParsedMapping],
        subcacheTableEnd: UInt64
    ) -> SyntheticSharedCacheSetPlanComparison.HeaderFacts {
        SyntheticSharedCacheSetPlanComparison.HeaderFacts(
            magic: Data(raw.magic),
            mappingOffset: raw.mappingOffset,
            mappingCount: raw.mappingCount,
            mappingWithSlideOffset: raw.mappingWithSlideOffset,
            mappingWithSlideCount: raw.mappingWithSlideCount,
            mappingTablesEnd: raw.mappingTablesEnd,
            codeSignatureOffset: raw.codeSignatureOffset,
            codeSignatureSize: raw.codeSignatureSize,
            sharedRegionStart: raw.sharedRegionStart,
            sharedRegionSize: raw.sharedRegionSize,
            imagesOffset: raw.imagesOffset,
            imagesCount: raw.imagesCount,
            imagesTextOffset: raw.imagesTextOffset,
            imagesTextCount: raw.imagesTextCount,
            subCacheArrayOffset: raw.subCacheArrayOffset,
            subCacheArrayCount: raw.subCacheArrayCount,
            subcacheTableEnd: subcacheTableEnd,
            symbolFileUUID: hex(raw.symbolFileUUID),
            headerUUID: hex(raw.uuid),
            platform: raw.platform,
            formatVersion: raw.formatFlags & 0xFF,
            simulator: raw.formatFlags & simulatorFlag != 0,
            locallyBuiltCache: raw.formatFlags & locallyBuiltFlag != 0,
            newFormatTLVs: raw.formatFlags & newFormatTLVsFlag != 0,
            padding: raw.formatFlags >> 13,
            cacheType: raw.cacheType,
            cacheSubType: raw.cacheSubType,
            mappings: mappings.map { mapping in
                SyntheticSharedCacheSetPlanComparison.MappingFact(
                    row: mapping.row,
                    address: mapping.address,
                    size: mapping.size,
                    fileOffset: mapping.fileOffset,
                    maximumProtection: mapping.maximumProtection,
                    initialProtection: mapping.initialProtection,
                    slideInfoOffset: mapping.slideInfoOffset,
                    slideInfoSize: mapping.slideInfoSize,
                    flags: mapping.flags
                )
            }
        )
    }

    static func fileFact(
        ordinal: Int,
        metadata: SyntheticCaptureFileMetadata,
        suffix: Data,
        uuid: Data,
        cacheVMOffset: UInt64,
        raw: RawHeader,
        mappings: [ParsedMapping],
        tableEnd: UInt64,
        discoveryBytes: Data
    ) -> SyntheticSharedCacheSetPlanComparison.HeaderOrderFile {
        SyntheticSharedCacheSetPlanComparison.HeaderOrderFile(
            ordinal: ordinal,
            metadata: metadata,
            fileBytes: UInt64(metadata.size),
            decodedSuffix: Data(suffix),
            suffixByteCount: UInt64(suffix.count),
            suffixBase64URL: base64URL(suffix),
            headerUUID: hex(uuid),
            cacheVMOffset: cacheVMOffset,
            header: headerFacts(
                raw: raw,
                mappings: mappings,
                subcacheTableEnd: tableEnd
            ),
            discoveryBytes: Data(discoveryBytes)
        )
    }

    static func checkedAdd(_ lhs: UInt64, _ rhs: UInt64) throws -> UInt64 {
        let (result, overflow) = lhs.addingReportingOverflow(rhs)
        guard !overflow else {
            throw SyntheticSharedCacheSetPlanFailure.resourceEnvelope(
                .arithmeticOverflow
            )
        }
        return result
    }

    static func checkedMultiply(_ lhs: UInt64, _ rhs: UInt64) throws -> UInt64 {
        let (result, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else {
            throw SyntheticSharedCacheSetPlanFailure.resourceEnvelope(
                .arithmeticOverflow
            )
        }
        return result
    }

    static func readUInt32(_ data: Data, at offset: Int) -> UInt32? {
        guard let bytes = readData(data, at: offset, count: 4) else {
            return nil
        }
        return bytes.enumerated().reduce(UInt32(0)) { value, item in
            value | (UInt32(item.element) << UInt32(item.offset * 8))
        }
    }

    static func readUInt64(_ data: Data, at offset: Int) -> UInt64? {
        guard let bytes = readData(data, at: offset, count: 8) else {
            return nil
        }
        return bytes.enumerated().reduce(UInt64(0)) { value, item in
            value | (UInt64(item.element) << UInt64(item.offset * 8))
        }
    }

    static func readData(
        _ data: Data,
        at offset: Int,
        count: Int
    ) -> Data? {
        let (end, overflow) = offset.addingReportingOverflow(count)
        guard offset >= 0,
              count >= 0,
              !overflow,
              end <= data.count else {
            return nil
        }
        return Data(data[offset..<end])
    }

    static func ownedCopy(
        _ data: Data,
        expectedCount: Int,
        fileOrdinal: Int
    ) throws -> Data {
        guard data.count == expectedCount else {
            throw SyntheticSharedCacheSetPlanFailure.discoveryBytes(
                fileOrdinal: fileOrdinal,
                reason: .exactLength(
                    expected: UInt64(expectedCount),
                    actual: data.count
                )
            )
        }
        var copy = Data(count: expectedCount)
        let copied = copy.withUnsafeMutableBytes { destination in
            data.copyBytes(to: destination)
        }
        guard copied == expectedCount,
              data.count == expectedCount else {
            throw SyntheticSharedCacheSetPlanFailure.discoveryBytes(
                fileOrdinal: fileOrdinal,
                reason: .exactLength(
                    expected: UInt64(expectedCount),
                    actual: data.count
                )
            )
        }
        return copy
    }

    static func hex(_ data: Data) -> String {
        data.map { String(format: "%02x", $0) }.joined()
    }

    static func base64URL(_ data: Data) -> String {
        data.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }
}
