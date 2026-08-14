import CryptoKit
import Foundation

/// Synthetic descriptor metadata used only to prove the E1 transcript state
/// machine. It is not produced by a filesystem API and carries no locator.
struct SyntheticCaptureFileMetadata: Equatable, Sendable {
    var device: UInt64
    var inode: UInt64
    var mode: UInt32
    var linkCount: UInt64
    var userID: UInt32
    var groupID: UInt32
    var size: Int64
    var blockCount: Int64
    var blockSize: Int64
    var flags: UInt32
    var generation: UInt32?
    var modificationTimeSeconds: Int64
    var modificationTimeNanoseconds: Int64
    var statusChangeTimeSeconds: Int64
    var statusChangeTimeNanoseconds: Int64
    var birthTimeSeconds: Int64
    var birthTimeNanoseconds: Int64
    var extendedAttributeSupportMask: UInt64
    var extendedFlags: UInt64
    var cloneID: UInt64?
    var cloneReferenceCount: UInt64?
}

/// One synthetic result from the future explicit-offset read loop.
enum SyntheticCaptureReadEvent: Equatable, Sendable {
    case interrupted(offset: UInt64)
    case bytes(offset: UInt64, data: Data)
    case endOfFile(offset: UInt64)
    case error(offset: UInt64, code: Int32)

    var offset: UInt64 {
        switch self {
        case .interrupted(let offset),
             .bytes(let offset, _),
             .endOfFile(let offset),
             .error(let offset, _):
            offset
        }
    }
}

enum SyntheticSmallArtifactCaptureFailure: Error, Equatable, Sendable {
    enum MetadataField: CaseIterable, Equatable, Sendable {
        case device
        case inode
        case mode
        case linkCount
        case userID
        case groupID
        case size
        case blockCount
        case blockSize
        case flags
        case generation
        case modificationTimeSeconds
        case modificationTimeNanoseconds
        case statusChangeTimeSeconds
        case statusChangeTimeNanoseconds
        case birthTimeSeconds
        case birthTimeNanoseconds
        case extendedAttributeSupportMask
        case extendedFlags
        case cloneID
        case cloneReferenceCount
    }

    enum ResourceEnvelopeReason: Equatable, Sendable {
        case arithmeticOverflow
        case readAttemptLimit(actual: UInt64, maximum: UInt64)
    }

    enum ReadErrorReason: Equatable, Sendable {
        case offset(
            index: Int,
            expected: UInt64,
            actual: UInt64
        )
        case fragmentSize(index: Int, bytes: Int)
        case fragmentBeyondChunk(index: Int, chunkEnd: UInt64)
        case system(offset: UInt64, code: Int32)
    }

    enum ShortReadReason: Equatable, Sendable {
        case endOfFile(expected: UInt64, actual: UInt64)
        case transcriptEnded(expected: UInt64, actual: UInt64)
        case missingEndOfFile(offset: UInt64)
    }

    enum ExpectedSHA256Reason: Equatable, Sendable {
        case format
        case mismatch(expected: String, actual: String)
    }

    case resourceEnvelope(ResourceEnvelopeReason)
    case fileType
    case linkCount(UInt64)
    case dataless
    case sparseStateUnavailable
    case sparseFile
    case fileSize(Int64)
    case expectedByteCount(expected: UInt64, actual: UInt64)
    case readInterruptedLimit(offset: UInt64)
    case readFragmentLimit(chunkOffset: UInt64)
    case readError(ReadErrorReason)
    case shortRead(ShortReadReason)
    case unexpectedTrailingByte(index: Int)
    case metadataDrift(MetadataField)
    case expectedSHA256WhenExplicit(ExpectedSHA256Reason)
}

/// Independently owned synthetic comparison facts only. This value has no ID,
/// locator, descriptor, reopen surface, parser conversion, or authority.
struct SyntheticSmallArtifactCaptureComparison: Equatable, Sendable {
    enum Role: Equatable, Sendable {
        case gitRoot
        case selfGuardRoot
        case dynamicLoader
        case fileImage
    }

    fileprivate enum ConstructionSeal: Equatable, Sendable {
        case verified
    }

    let role: Role
    let metadata: SyntheticCaptureFileMetadata
    let retainedBytes: Data
    let fileSHA256: String
    let fileBytes: UInt64
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

    fileprivate init(
        role: Role,
        metadata: SyntheticCaptureFileMetadata,
        retainedBytes: Data,
        fileSHA256: String,
        fileBytes: UInt64,
        seal: ConstructionSeal
    ) {
        self.role = role
        self.metadata = metadata
        self.retainedBytes = Data(retainedBytes)
        self.fileSHA256 = fileSHA256
        self.fileBytes = fileBytes
        self.constructionSeal = seal
    }
}

enum SyntheticSmallArtifactCaptureVerifier {
    static let maximumFileBytes = 1_048_576
    static let maximumReadChunkBytes = 65_536
    static let maximumPositiveFragmentsPerChunk = 16
    static let maximumInterruptedRetriesPerOffset = 8

    static let regularFileTypeMask: UInt32 = 0o170000
    static let regularFileType: UInt32 = 0o100000
    static let datalessFlag: UInt32 = 0x40000000

    // E1 models normalized synthetic facts. A future Darwin adapter must map
    // its source-grounded attributes in a separately reviewed gate.
    static let requiredSparseStateSupportBit: UInt64 = 1 << 0
    static let sparseFlag: UInt64 = 1 << 0

    static func compare(
        role: SyntheticSmallArtifactCaptureComparison.Role,
        before: SyntheticCaptureFileMetadata,
        after: SyntheticCaptureFileMetadata,
        expectedFileBytes: UInt64,
        expectedSHA256: String,
        transcript input: [SyntheticCaptureReadEvent]
    ) throws -> SyntheticSmallArtifactCaptureComparison {
        try validatePreReadMetadata(before)

        guard let metadataBytes = UInt64(exactly: before.size) else {
            throw SyntheticSmallArtifactCaptureFailure.fileSize(before.size)
        }
        guard let retainedCapacity = Int(exactly: metadataBytes) else {
            throw SyntheticSmallArtifactCaptureFailure.resourceEnvelope(
                .arithmeticOverflow
            )
        }

        let maximumReadAttempts = try checkedReadAttemptLimit(
            forFileBytes: metadataBytes
        )
        guard let readAttemptCount = UInt64(exactly: input.count) else {
            throw SyntheticSmallArtifactCaptureFailure.resourceEnvelope(
                .arithmeticOverflow
            )
        }
        guard readAttemptCount <= maximumReadAttempts else {
            throw SyntheticSmallArtifactCaptureFailure.resourceEnvelope(
                .readAttemptLimit(
                    actual: readAttemptCount,
                    maximum: maximumReadAttempts
                )
            )
        }

        let transcript = Array(input)
        var retainedBytes = Data()
        retainedBytes.reserveCapacity(retainedCapacity)
        var hasher = SHA256()
        var currentOffset: UInt64 = 0
        var currentChunkOffset: UInt64 = 0
        var currentChunkEnd = min(
            UInt64(maximumReadChunkBytes),
            metadataBytes
        )
        var positiveFragmentCount = 0
        var interruptedRetryCount = 0
        var sawEndOfFile = false

        for (index, event) in transcript.enumerated() {
            if sawEndOfFile {
                throw SyntheticSmallArtifactCaptureFailure
                    .unexpectedTrailingByte(index: index)
            }
            guard event.offset == currentOffset else {
                throw SyntheticSmallArtifactCaptureFailure.readError(
                    .offset(
                        index: index,
                        expected: currentOffset,
                        actual: event.offset
                    )
                )
            }
            if currentOffset == metadataBytes,
               case .interrupted = event
            {
                throw SyntheticSmallArtifactCaptureFailure
                    .unexpectedTrailingByte(index: index)
            }

            switch event {
            case .interrupted:
                let (next, overflow) = interruptedRetryCount
                    .addingReportingOverflow(1)
                guard !overflow else {
                    throw SyntheticSmallArtifactCaptureFailure
                        .resourceEnvelope(.arithmeticOverflow)
                }
                interruptedRetryCount = next
                guard interruptedRetryCount <=
                    maximumInterruptedRetriesPerOffset
                else {
                    throw SyntheticSmallArtifactCaptureFailure
                        .readInterruptedLimit(offset: currentOffset)
                }

            case .error(let offset, let code):
                throw SyntheticSmallArtifactCaptureFailure.readError(
                    .system(offset: offset, code: code)
                )

            case .endOfFile:
                guard currentOffset == metadataBytes else {
                    throw SyntheticSmallArtifactCaptureFailure.shortRead(
                        .endOfFile(
                            expected: metadataBytes,
                            actual: currentOffset
                        )
                    )
                }
                sawEndOfFile = true

            case .bytes(_, let data):
                guard data.count > 0,
                      data.count <= maximumReadChunkBytes
                else {
                    throw SyntheticSmallArtifactCaptureFailure.readError(
                        .fragmentSize(index: index, bytes: data.count)
                    )
                }

                let (nextFragmentCount, fragmentCountOverflow) =
                    positiveFragmentCount.addingReportingOverflow(1)
                guard !fragmentCountOverflow else {
                    throw SyntheticSmallArtifactCaptureFailure
                        .resourceEnvelope(.arithmeticOverflow)
                }
                positiveFragmentCount = nextFragmentCount
                guard positiveFragmentCount <=
                    maximumPositiveFragmentsPerChunk
                else {
                    throw SyntheticSmallArtifactCaptureFailure
                        .readFragmentLimit(
                            chunkOffset: currentChunkOffset
                        )
                }

                guard let fragmentBytes = UInt64(exactly: data.count) else {
                    throw SyntheticSmallArtifactCaptureFailure
                        .resourceEnvelope(.arithmeticOverflow)
                }
                let (nextOffset, offsetOverflow) = currentOffset
                    .addingReportingOverflow(fragmentBytes)
                guard !offsetOverflow else {
                    throw SyntheticSmallArtifactCaptureFailure
                        .resourceEnvelope(.arithmeticOverflow)
                }
                guard nextOffset <= metadataBytes else {
                    throw SyntheticSmallArtifactCaptureFailure
                        .unexpectedTrailingByte(index: index)
                }
                guard nextOffset <= currentChunkEnd else {
                    throw SyntheticSmallArtifactCaptureFailure.readError(
                        .fragmentBeyondChunk(
                            index: index,
                            chunkEnd: currentChunkEnd
                        )
                    )
                }

                retainedBytes.append(data)
                hasher.update(data: data)
                currentOffset = nextOffset
                interruptedRetryCount = 0

                if currentOffset == currentChunkEnd,
                   currentOffset < metadataBytes
                {
                    currentChunkOffset = currentOffset
                    let remaining = metadataBytes - currentOffset
                    let planned = min(
                        UInt64(maximumReadChunkBytes),
                        remaining
                    )
                    let (nextChunkEnd, chunkOverflow) = currentOffset
                        .addingReportingOverflow(planned)
                    guard !chunkOverflow else {
                        throw SyntheticSmallArtifactCaptureFailure
                            .resourceEnvelope(.arithmeticOverflow)
                    }
                    currentChunkEnd = nextChunkEnd
                    positiveFragmentCount = 0
                }
            }
        }

        guard currentOffset == metadataBytes else {
            throw SyntheticSmallArtifactCaptureFailure.shortRead(
                .transcriptEnded(
                    expected: metadataBytes,
                    actual: currentOffset
                )
            )
        }
        guard sawEndOfFile else {
            throw SyntheticSmallArtifactCaptureFailure.shortRead(
                .missingEndOfFile(offset: currentOffset)
            )
        }

        if let drift = firstMetadataDrift(before: before, after: after) {
            throw SyntheticSmallArtifactCaptureFailure
                .metadataDrift(drift)
        }

        guard let retainedCount = UInt64(exactly: retainedBytes.count) else {
            throw SyntheticSmallArtifactCaptureFailure.resourceEnvelope(
                .arithmeticOverflow
            )
        }
        guard retainedCount == expectedFileBytes else {
            throw SyntheticSmallArtifactCaptureFailure.expectedByteCount(
                expected: expectedFileBytes,
                actual: retainedCount
            )
        }
        guard isLowercaseSHA256(expectedSHA256) else {
            throw SyntheticSmallArtifactCaptureFailure
                .expectedSHA256WhenExplicit(.format)
        }

        let actualSHA256 = hasher.finalize()
            .map { String(format: "%02x", $0) }
            .joined()
        guard actualSHA256 == expectedSHA256 else {
            throw SyntheticSmallArtifactCaptureFailure
                .expectedSHA256WhenExplicit(
                    .mismatch(
                        expected: expectedSHA256,
                        actual: actualSHA256
                    )
                )
        }

        return SyntheticSmallArtifactCaptureComparison(
            role: role,
            metadata: before,
            retainedBytes: retainedBytes,
            fileSHA256: actualSHA256,
            fileBytes: retainedCount,
            seal: .verified
        )
    }

    static func checkedReadAttemptLimit(
        forFileBytes fileBytes: UInt64
    ) throws -> UInt64 {
        let chunkBytes = UInt64(maximumReadChunkBytes)
        let completeChunks = fileBytes / chunkBytes
        let partialChunk = fileBytes % chunkBytes == 0 ? UInt64(0) : 1
        let (chunkCount, chunkCountOverflow) = completeChunks
            .addingReportingOverflow(partialChunk)
        guard !chunkCountOverflow else {
            throw SyntheticSmallArtifactCaptureFailure.resourceEnvelope(
                .arithmeticOverflow
            )
        }
        return try checkedReadAttemptLimit(forChunkCount: chunkCount)
    }

    static func checkedReadAttemptLimit(
        forChunkCount chunkCount: UInt64
    ) throws -> UInt64 {
        let (fragmentAttempts, fragmentOverflow) = chunkCount
            .multipliedReportingOverflow(
                by: UInt64(maximumPositiveFragmentsPerChunk)
            )
        guard !fragmentOverflow else {
            throw SyntheticSmallArtifactCaptureFailure.resourceEnvelope(
                .arithmeticOverflow
            )
        }
        let retriesAndPositive = UInt64(
            maximumInterruptedRetriesPerOffset + 1
        )
        let (streamAttempts, streamOverflow) = fragmentAttempts
            .multipliedReportingOverflow(by: retriesAndPositive)
        guard !streamOverflow else {
            throw SyntheticSmallArtifactCaptureFailure.resourceEnvelope(
                .arithmeticOverflow
            )
        }
        let (withEOF, eofOverflow) = streamAttempts
            .addingReportingOverflow(1)
        guard !eofOverflow else {
            throw SyntheticSmallArtifactCaptureFailure.resourceEnvelope(
                .arithmeticOverflow
            )
        }
        return withEOF
    }
}

private extension SyntheticSmallArtifactCaptureVerifier {
    static func validatePreReadMetadata(
        _ metadata: SyntheticCaptureFileMetadata
    ) throws {
        guard metadata.mode & regularFileTypeMask == regularFileType else {
            throw SyntheticSmallArtifactCaptureFailure.fileType
        }
        guard metadata.linkCount == 1 else {
            throw SyntheticSmallArtifactCaptureFailure
                .linkCount(metadata.linkCount)
        }
        guard metadata.size > 0,
              metadata.size <= Int64(maximumFileBytes)
        else {
            throw SyntheticSmallArtifactCaptureFailure
                .fileSize(metadata.size)
        }
        guard metadata.flags & datalessFlag == 0 else {
            throw SyntheticSmallArtifactCaptureFailure.dataless
        }
        guard metadata.extendedAttributeSupportMask &
            requiredSparseStateSupportBit != 0
        else {
            throw SyntheticSmallArtifactCaptureFailure
                .sparseStateUnavailable
        }
        guard metadata.extendedFlags & sparseFlag == 0 else {
            throw SyntheticSmallArtifactCaptureFailure.sparseFile
        }
    }

    static func isLowercaseSHA256(_ value: String) -> Bool {
        let bytes = Array(value.utf8)
        return bytes.count == 64 && bytes.allSatisfy {
            (0x30...0x39).contains($0) ||
                (0x61...0x66).contains($0)
        }
    }

    static func firstMetadataDrift(
        before: SyntheticCaptureFileMetadata,
        after: SyntheticCaptureFileMetadata
    ) -> SyntheticSmallArtifactCaptureFailure.MetadataField? {
        if before.device != after.device { return .device }
        if before.inode != after.inode { return .inode }
        if before.mode != after.mode { return .mode }
        if before.linkCount != after.linkCount { return .linkCount }
        if before.userID != after.userID { return .userID }
        if before.groupID != after.groupID { return .groupID }
        if before.size != after.size { return .size }
        if before.blockCount != after.blockCount { return .blockCount }
        if before.blockSize != after.blockSize { return .blockSize }
        if before.flags != after.flags { return .flags }
        if before.generation != after.generation { return .generation }
        if before.modificationTimeSeconds !=
            after.modificationTimeSeconds
        {
            return .modificationTimeSeconds
        }
        if before.modificationTimeNanoseconds !=
            after.modificationTimeNanoseconds
        {
            return .modificationTimeNanoseconds
        }
        if before.statusChangeTimeSeconds !=
            after.statusChangeTimeSeconds
        {
            return .statusChangeTimeSeconds
        }
        if before.statusChangeTimeNanoseconds !=
            after.statusChangeTimeNanoseconds
        {
            return .statusChangeTimeNanoseconds
        }
        if before.birthTimeSeconds != after.birthTimeSeconds {
            return .birthTimeSeconds
        }
        if before.birthTimeNanoseconds != after.birthTimeNanoseconds {
            return .birthTimeNanoseconds
        }
        if before.extendedAttributeSupportMask !=
            after.extendedAttributeSupportMask
        {
            return .extendedAttributeSupportMask
        }
        if before.extendedFlags != after.extendedFlags {
            return .extendedFlags
        }
        if before.cloneID != after.cloneID { return .cloneID }
        if before.cloneReferenceCount != after.cloneReferenceCount {
            return .cloneReferenceCount
        }
        return nil
    }
}
