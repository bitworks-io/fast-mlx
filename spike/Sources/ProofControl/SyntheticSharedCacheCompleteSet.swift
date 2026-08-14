import CryptoKit
import Foundation

struct SyntheticSharedCacheCompleteFileTranscript:
    Equatable,
    Sendable
{
    let fileOrdinal: Int
    let beforeMetadata: SyntheticCaptureFileMetadata
    let afterMetadata: SyntheticCaptureFileMetadata
    let events: [SyntheticCaptureReadEvent]
}

enum SyntheticSharedCacheCompleteSetFailure:
    Error,
    Equatable,
    Sendable
{
    enum MetadataPosition: Equatable, Sendable {
        case before
        case after
    }

    enum CompleteStreamReason: Equatable, Sendable {
        case transcriptCount(expected: Int, actual: Int)
        case fileOrdinal(
            transcriptOrdinal: Int,
            expected: Int,
            actual: Int
        )
        case role(
            expected: RuntimeClosureExpectationArtifactRole,
            actual: RuntimeClosureExpectationArtifactRole
        )
        case platform
        case cacheRecords
        case sourceProfile
        case resourceArithmetic
        case aggregateBytes(actual: UInt64, maximum: UInt64)
        case chunkLimit(actual: UInt64, maximum: UInt64)
        case attemptLimit(actual: UInt64, maximum: UInt64)
        case eventCountDrift(fileOrdinal: Int)
        case payloadCountDrift(fileOrdinal: Int, eventOrdinal: Int)
        case offsetGap(
            fileOrdinal: Int,
            eventOrdinal: Int,
            expected: UInt64,
            actual: UInt64
        )
        case offsetOverlapOrReorder(
            fileOrdinal: Int,
            eventOrdinal: Int,
            expected: UInt64,
            actual: UInt64
        )
        case fragmentSize(
            fileOrdinal: Int,
            eventOrdinal: Int,
            bytes: Int
        )
        case fragmentBeyondChunk(
            fileOrdinal: Int,
            eventOrdinal: Int,
            chunkEnd: UInt64
        )
        case fragmentPastMetadata(
            fileOrdinal: Int,
            eventOrdinal: Int,
            fileBytes: UInt64
        )
        case earlyEOF(
            fileOrdinal: Int,
            eventOrdinal: Int,
            expected: UInt64,
            actual: UInt64
        )
        case missingEOF(fileOrdinal: Int, offset: UInt64)
        case interruptionAfterFinalBytes(
            fileOrdinal: Int,
            eventOrdinal: Int
        )
        case retainedStoreShape
        case retainedByteMismatch(
            fileOrdinal: Int,
            offset: UInt64
        )
        case retainedCoverage(fileOrdinal: Int)
        case evidenceByteLimit(actual: UInt64, maximum: UInt64)
    }

    enum ExpectedSHA256Reason: Equatable, Sendable {
        case format
        case mismatch(expected: String, actual: String)
    }

    enum CacheSetIdentityReason: Equatable, Sendable {
        case derivation
        case mismatch
    }

    enum CacheImageFormatReason: Equatable, Sendable {
        case binding
        case installName
        case machOHeader
        case loadCommandFrame
        case unknownLoadCommand
        case forbiddenLoadCommand
        case uuid
        case idDylibCommand
        case codeSignature
        case linkedit
        case signatureParse
        case expectedMember
        case rederivation
    }

    enum CacheImageIdentityReason: Equatable, Sendable {
        case derivation
        case mismatch
    }

    case predecessor(SyntheticSharedCacheDiscoveryRangePlanFailure)
    case completeStream(CompleteStreamReason)
    case readInterruptedLimit(fileOrdinal: Int, eventOrdinal: Int)
    case readFragmentLimit(fileOrdinal: Int, eventOrdinal: Int)
    case readError(fileOrdinal: Int, eventOrdinal: Int, code: Int32)
    case unexpectedTrailingByte(fileOrdinal: Int, eventOrdinal: Int)
    case metadataDrift(
        fileOrdinal: Int,
        position: MetadataPosition,
        field: SyntheticSmallArtifactCaptureFailure.MetadataField
    )
    case expectedByteCount(
        fileOrdinal: Int,
        expected: UInt64,
        actual: UInt64
    )
    case expectedSHA256(fileOrdinal: Int, ExpectedSHA256Reason)
    case cacheSetIdentity(CacheSetIdentityReason)
    case cacheImageFormat(memberOrdinal: Int, CacheImageFormatReason)
    case cacheImageIdentity(memberOrdinal: Int, CacheImageIdentityReason)
}

struct SyntheticSharedCacheCompleteSetComparison: Equatable {
    struct ResourceCounts: Equatable {
        let fileCount: Int
        let aggregateFileBytes: UInt64
        let streamChunkCount: UInt64
        let streamAttemptCount: UInt64
        let matchedRetainedBytes: UInt64
        let memberCount: Int
        let compactEvidenceSemanticBytes: UInt64
    }

    fileprivate enum ConstructionSeal: Equatable {
        case verified
    }

    let cacheSetEvidence: SyntheticSharedCacheSetIdentityEvidence
    let sharedCacheImageEvidence:
        [SyntheticSharedCacheImageContentIdentityEvidence]
    let resourceCounts: ResourceCounts
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

    fileprivate init(
        cacheSetEvidence: SyntheticSharedCacheSetIdentityEvidence,
        sharedCacheImageEvidence:
            [SyntheticSharedCacheImageContentIdentityEvidence],
        resourceCounts: ResourceCounts,
        seal: ConstructionSeal
    ) {
        self.cacheSetEvidence = cacheSetEvidence
        self.sharedCacheImageEvidence = Array(sharedCacheImageEvidence)
        self.resourceCounts = resourceCounts
        self.constructionSeal = seal
    }
}

enum SyntheticSharedCacheCompleteSetVerifier {
    static let maximumFileCount = 64
    static let maximumSubcacheCount = 63
    static let maximumFileBytes: UInt64 = 17_179_869_184
    static let maximumSetBytes: UInt64 = 68_719_476_736
    static let logicalChunkBytes: UInt64 = 1_048_576
    static let maximumPositiveFragmentsPerChunk = 16
    static let maximumInterruptedRetriesPerOffset = 8
    static let maximumPerFileChunkCount: UInt64 = 16_384
    static let maximumPerFileAttempts: UInt64 = 2_359_297
    static let maximumSetChunkCount: UInt64 = 65_599
    static let maximumSetAttempts: UInt64 = 9_446_320
    static let maximumFragmentSnapshotBytes = 1_048_576
    static let maximumCompactEvidenceSemanticBytes: UInt64 = 16_777_216
    static let maximumVerifierOwnedContentBytes: UInt64 = 537_919_488

    static func compare(
        setPlan: SyntheticSharedCacheSetPlanComparison,
        gitExpectation: AnchoredRuntimeClosureExpectationDocument,
        selfGuardExpectation: AnchoredRuntimeClosureExpectationDocument,
        discoveryTranscripts: [SyntheticSharedCacheBoundedReadTranscript],
        planProbeTranscripts: [SyntheticSharedCacheBoundedReadTranscript],
        completeFileTranscripts:
            [SyntheticSharedCacheCompleteFileTranscript]
    ) throws -> SyntheticSharedCacheCompleteSetComparison {
        let rangePlan: SyntheticSharedCacheDiscoveryRangePlanComparison
        do {
            rangePlan =
                try SyntheticSharedCacheDiscoveryRangePlanVerifier.compare(
                    setPlan: setPlan,
                    gitExpectation: gitExpectation,
                    selfGuardExpectation: selfGuardExpectation,
                    discoveryTranscripts: discoveryTranscripts,
                    planProbeTranscripts: planProbeTranscripts
                )
        } catch let failure as SyntheticSharedCacheDiscoveryRangePlanFailure {
            throw Failure.predecessor(failure)
        }

        guard setPlan.sourceProfile == .reviewed,
              rangePlan.sourceProfile == setPlan.sourceProfile
        else {
            throw Failure.completeStream(.sourceProfile)
        }

        let git = try reanchor(gitExpectation, expected: .git)
        let selfGuard = try reanchor(
            selfGuardExpectation,
            expected: .selfGuard
        )
        let members = try sharedCacheMembers(git: git, selfGuard: selfGuard)
        let expectedRecords = try expectedHeaderOrderRecords(
            files: setPlan.headerOrderFiles,
            git: git,
            selfGuard: selfGuard
        )
        let envelope = try validateEnvelope(
            files: setPlan.headerOrderFiles,
            records: expectedRecords,
            expectedAggregateFileBytes: setPlan.aggregateFileBytes,
            completeFileTranscripts: completeFileTranscripts
        )
        let retained = try validateRetainedStores(
            rangePlan: rangePlan,
            files: setPlan.headerOrderFiles,
            members: members
        )

        var facts: [HeaderStreamFact] = []
        facts.reserveCapacity(setPlan.headerOrderFiles.count)
        var matchedRetainedBytes: UInt64 = 0
        for index in setPlan.headerOrderFiles.indices {
            let fact = try replayCompleteFile(
                transcript: completeFileTranscripts[index],
                file: setPlan.headerOrderFiles[index],
                expectedRecord: expectedRecords[index],
                retainedRanges: retained.rangesByFile[index]
            )
            matchedRetainedBytes = try checkedAdd(
                matchedRetainedBytes,
                fact.matchedRetainedBytes
            )
            facts.append(fact)
        }
        guard matchedRetainedBytes ==
            rangePlan.resourceCounts.combinedRetainedBytes
        else {
            throw Failure.completeStream(.retainedCoverage(fileOrdinal: 0))
        }

        let cacheSetEvidence = try deriveCacheSetEvidence(
            facts: facts,
            git: git,
            selfGuard: selfGuard
        )
        let imageEvidence = try deriveImageEvidence(
            cacheSetEvidence: cacheSetEvidence,
            rangePlan: rangePlan,
            files: setPlan.headerOrderFiles,
            members: members
        )
        let compactBytes = try compactEvidenceSemanticBytes(
            cacheSetEvidence: cacheSetEvidence,
            imageEvidence: imageEvidence
        )
        guard compactBytes <= maximumCompactEvidenceSemanticBytes else {
            throw Failure.completeStream(.evidenceByteLimit(
                actual: compactBytes,
                maximum: maximumCompactEvidenceSemanticBytes
            ))
        }

        let counts = SyntheticSharedCacheCompleteSetComparison
            .ResourceCounts(
                fileCount: facts.count,
                aggregateFileBytes: envelope.aggregateFileBytes,
                streamChunkCount: envelope.streamChunkCount,
                streamAttemptCount: envelope.streamAttemptCount,
                matchedRetainedBytes: matchedRetainedBytes,
                memberCount: members.count,
                compactEvidenceSemanticBytes: compactBytes
            )
        return SyntheticSharedCacheCompleteSetComparison(
            cacheSetEvidence: cacheSetEvidence,
            sharedCacheImageEvidence: imageEvidence,
            resourceCounts: counts,
            seal: .verified
        )
    }
}

private extension SyntheticSharedCacheCompleteSetVerifier {
    typealias Failure = SyntheticSharedCacheCompleteSetFailure
    typealias CompleteReason = Failure.CompleteStreamReason
    typealias MetadataPosition = Failure.MetadataPosition
    typealias HeaderOrderFile =
        SyntheticSharedCacheSetPlanComparison.HeaderOrderFile
    typealias RangeComparison =
        SyntheticSharedCacheDiscoveryRangePlanComparison

    struct Envelope {
        let aggregateFileBytes: UInt64
        let streamChunkCount: UInt64
        let streamAttemptCount: UInt64
    }

    struct SharedMember {
        let ordinal: Int
        let field: RuntimeClosureExpectationMemberFields
    }

    struct HeaderStreamFact {
        let headerOrderOrdinal: Int
        let decodedSuffix: Data
        let suffixBytes: UInt64
        let suffixBase64URL: String
        let headerUUID: String
        let fileSHA256: String
        let fileBytes: UInt64
        let matchedRetainedBytes: UInt64
    }

    struct RetainedRange {
        let fileOrdinal: Int
        let start: UInt64
        let length: UInt64
        let bytes: Data
        var comparedUntil: UInt64

        var end: UInt64 { start + length }
    }

    struct RetainedStores {
        var rangesByFile: [[RetainedRange]]
    }

    struct Mapping {
        let fileOrdinal: Int
        let vmStart: UInt64
        let vmEnd: UInt64
        let fileStart: UInt64
        let fileEnd: UInt64
    }

    struct ImageRow {
        let address: UInt64
        let pathFileOffset: UInt64
    }

    struct Translation {
        let fileOrdinal: Int
        let fileStart: UInt64
    }

    struct MachHeaderFacts {
        let commandCount: UInt32
        let sizeOfCommands: UInt32
    }

    struct SignatureRequest {
        let fileOrdinal: Int
        let fileStart: UInt64
        let length: UInt64
    }

    struct LinkeditFacts {
        let vmAddress: UInt64
        let vmSize: UInt64
        let fileOffset: UInt64
        let fileSize: UInt64
    }

    struct LoadCommandFacts {
        let uuid: Data
        let loadCommandsSHA256: String
        let signature: SignatureRequest?
    }

    static let recognizedLoadCommands: Set<UInt32> = [
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

    static let forbiddenLoadCommands: Set<UInt32> = [
        0x00000027,
        0x80000028,
        0x0000000e,
        0x00000021,
        0x0000002c,
        0x80000035,
        0x0000000f,
    ]

    static func reanchor(
        _ input: AnchoredRuntimeClosureExpectationDocument,
        expected role: RuntimeClosureExpectationArtifactRole
    ) throws -> AnchoredRuntimeClosureExpectationDocument {
        let current: AnchoredRuntimeClosureExpectationDocument
        do {
            current = try RuntimeClosureExpectationVerifier.anchor(
                expectationFile: input.expectationFile,
                trustAnchor: input.trustAnchor
            )
        } catch {
            throw Failure.completeStream(.role(
                expected: role,
                actual: input.fields.artifactRole
            ))
        }
        guard current == input else {
            throw Failure.completeStream(.cacheRecords)
        }
        guard current.fields.artifactRole == role else {
            throw Failure.completeStream(.role(
                expected: role,
                actual: current.fields.artifactRole
            ))
        }
        return current
    }

    static func sharedCacheMembers(
        git: AnchoredRuntimeClosureExpectationDocument,
        selfGuard: AnchoredRuntimeClosureExpectationDocument
    ) throws -> [SharedMember] {
        guard platformTuple(git.fields) == platformTuple(selfGuard.fields)
        else {
            throw Failure.completeStream(.platform)
        }
        guard git.sharedCacheSetEvidence == selfGuard.sharedCacheSetEvidence
        else {
            throw Failure.completeStream(.cacheRecords)
        }

        var union: [Data: RuntimeClosureExpectationMemberFields] = [:]
        for fields in [git.fields, selfGuard.fields] {
            for member in fields.members where member.storage == .sharedCache {
                if let existing = union[member.decodedInstallName] {
                    guard existing == member else {
                        throw Failure.completeStream(.cacheRecords)
                    }
                } else {
                    union[Data(member.decodedInstallName)] = member
                }
            }
        }
        guard !union.isEmpty, union.count <= 512 else {
            throw Failure.completeStream(.cacheRecords)
        }
        return union.values.sorted {
            $0.decodedInstallName.lexicographicallyPrecedes(
                $1.decodedInstallName
            )
        }.enumerated().map { ordinal, member in
            SharedMember(ordinal: ordinal, field: member)
        }
    }

    static func expectedHeaderOrderRecords(
        files: [HeaderOrderFile],
        git: AnchoredRuntimeClosureExpectationDocument,
        selfGuard: AnchoredRuntimeClosureExpectationDocument
    ) throws -> [SyntheticSharedCacheFileRecord] {
        let gitCache = git.sharedCacheSetEvidence
        let selfCache = selfGuard.sharedCacheSetEvidence
        guard gitCache == selfCache, gitCache.records.count == files.count
        else {
            throw Failure.completeStream(.cacheRecords)
        }

        var result: [SyntheticSharedCacheFileRecord] = []
        result.reserveCapacity(files.count)
        for file in files {
            let matches = gitCache.decodedSuffixes.indices.filter {
                gitCache.decodedSuffixes[$0] == file.decodedSuffix
            }
            guard matches.count == 1 else {
                throw Failure.completeStream(.cacheRecords)
            }
            let record = gitCache.records[matches[0]]
            guard record.suffixBytes == file.suffixByteCount,
                  record.suffixBase64URL == file.suffixBase64URL,
                  record.fileBytes == file.fileBytes,
                  record.headerUUID == file.headerUUID
            else {
                throw Failure.completeStream(.cacheRecords)
            }
            result.append(record)
        }
        return result
    }

    static func validateEnvelope(
        files: [HeaderOrderFile],
        records: [SyntheticSharedCacheFileRecord],
        expectedAggregateFileBytes: UInt64,
        completeFileTranscripts:
            [SyntheticSharedCacheCompleteFileTranscript]
    ) throws -> Envelope {
        guard (1...maximumFileCount).contains(files.count),
              records.count == files.count,
              completeFileTranscripts.count == files.count
        else {
            throw Failure.completeStream(.transcriptCount(
                expected: files.count,
                actual: completeFileTranscripts.count
            ))
        }
        guard files.count - 1 <= maximumSubcacheCount else {
            throw Failure.completeStream(.transcriptCount(
                expected: maximumFileCount,
                actual: files.count
            ))
        }

        var aggregateBytes: UInt64 = 0
        var aggregateChunks: UInt64 = 0
        var aggregateAttempts: UInt64 = 0
        for index in files.indices {
            let file = files[index]
            let transcript = completeFileTranscripts[index]
            guard transcript.fileOrdinal == index,
                  file.ordinal == index
            else {
                throw Failure.completeStream(.fileOrdinal(
                    transcriptOrdinal: index,
                    expected: index,
                    actual: transcript.fileOrdinal
                ))
            }
            if let field = firstMetadataDrift(
                file.metadata,
                transcript.beforeMetadata
            ) {
                throw Failure.metadataDrift(
                    fileOrdinal: index,
                    position: .before,
                    field: field
                )
            }
            guard let metadataBytes = UInt64(exactly: file.metadata.size),
                  metadataBytes > 0,
                  metadataBytes == file.fileBytes,
                  metadataBytes == records[index].fileBytes
            else {
                throw Failure.expectedByteCount(
                    fileOrdinal: index,
                    expected: file.fileBytes,
                    actual: UInt64(max(file.metadata.size, 0))
                )
            }
            guard metadataBytes <= maximumFileBytes else {
                throw Failure.completeStream(.aggregateBytes(
                    actual: metadataBytes,
                    maximum: maximumFileBytes
                ))
            }
            guard records[index].headerUUID == file.headerUUID else {
                throw Failure.completeStream(.cacheRecords)
            }
            guard isLowercaseHex(records[index].fileSHA256, count: 64) else {
                throw Failure.expectedSHA256(
                    fileOrdinal: index,
                    .format
                )
            }

            let chunks = chunkCount(metadataBytes)
            guard chunks <= maximumPerFileChunkCount else {
                throw Failure.completeStream(.chunkLimit(
                    actual: chunks,
                    maximum: maximumPerFileChunkCount
                ))
            }
            let attempts = try maximumAttempts(forChunkCount: chunks)
            guard UInt64(transcript.events.count) <= attempts,
                  attempts <= maximumPerFileAttempts
            else {
                throw Failure.completeStream(.attemptLimit(
                    actual: UInt64(transcript.events.count),
                    maximum: maximumPerFileAttempts
                ))
            }
            aggregateBytes = try checkedAdd(aggregateBytes, metadataBytes)
            aggregateChunks = try checkedAdd(aggregateChunks, chunks)
            aggregateAttempts = try checkedAdd(
                aggregateAttempts,
                UInt64(transcript.events.count)
            )
        }
        guard aggregateBytes == expectedAggregateFileBytes,
              aggregateBytes <= maximumSetBytes
        else {
            throw Failure.completeStream(.aggregateBytes(
                actual: aggregateBytes,
                maximum: maximumSetBytes
            ))
        }
        guard aggregateChunks <= maximumSetChunkCount else {
            throw Failure.completeStream(.chunkLimit(
                actual: aggregateChunks,
                maximum: maximumSetChunkCount
            ))
        }
        guard aggregateAttempts <= maximumSetAttempts else {
            throw Failure.completeStream(.attemptLimit(
                actual: aggregateAttempts,
                maximum: maximumSetAttempts
            ))
        }
        return Envelope(
            aggregateFileBytes: aggregateBytes,
            streamChunkCount: aggregateChunks,
            streamAttemptCount: aggregateAttempts
        )
    }

    static func validateRetainedStores(
        rangePlan: RangeComparison,
        files: [HeaderOrderFile],
        members: [SharedMember]
    ) throws -> RetainedStores {
        guard rangePlan.discoveryStores.count == files.count else {
            throw Failure.completeStream(.retainedStoreShape)
        }
        var rangesByFile = Array(
            repeating: [RetainedRange](),
            count: files.count
        )
        for index in files.indices {
            let store = rangePlan.discoveryStores[index]
            guard store.fileOrdinal == index,
                  store.bytes == files[index].discoveryBytes,
                  UInt64(store.bytes.count) <= files[index].fileBytes
            else {
                throw Failure.completeStream(.retainedStoreShape)
            }
            if !store.bytes.isEmpty {
                rangesByFile[index].append(RetainedRange(
                    fileOrdinal: index,
                    start: 0,
                    length: UInt64(store.bytes.count),
                    bytes: store.bytes,
                    comparedUntil: 0
                ))
            }
        }

        var previousAdditional: (fileOrdinal: Int, start: UInt64, end: UInt64)?
        for index in rangePlan.additionalPhysicalRanges.indices {
            let range = rangePlan.additionalPhysicalRanges[index]
            guard range.ordinal == index,
                  range.fileOrdinal >= 0,
                  range.fileOrdinal < files.count,
                  range.length > 0,
                  UInt64(range.bytes.count) == range.length,
                  let end = optionalCheckedAdd(range.start, range.length),
                  end <= files[range.fileOrdinal].fileBytes
            else {
                throw Failure.completeStream(.retainedStoreShape)
            }
            if let previous = previousAdditional,
               range.fileOrdinal < previous.fileOrdinal ||
                (range.fileOrdinal == previous.fileOrdinal &&
                 range.start < previous.end)
            {
                throw Failure.completeStream(.retainedStoreShape)
            }
            let discoveryEnd = UInt64(
                rangePlan.discoveryStores[range.fileOrdinal].bytes.count
            )
            guard range.start >= discoveryEnd else {
                throw Failure.completeStream(.retainedStoreShape)
            }
            previousAdditional = (range.fileOrdinal, range.start, end)
            rangesByFile[range.fileOrdinal].append(RetainedRange(
                fileOrdinal: range.fileOrdinal,
                start: range.start,
                length: range.length,
                bytes: range.bytes,
                comparedUntil: range.start
            ))
        }

        try validateConsumerBindings(
            rangePlan.consumerBindings,
            files: files,
            discoveryStores: rangePlan.discoveryStores,
            additionalRanges: rangePlan.additionalPhysicalRanges,
            members: members
        )
        return RetainedStores(rangesByFile: rangesByFile)
    }

    static func validateConsumerBindings(
        _ bindings: [RangeComparison.ConsumerBinding],
        files: [HeaderOrderFile],
        discoveryStores: [RangeComparison.DiscoveryStore],
        additionalRanges: [RangeComparison.AdditionalPhysicalRange],
        members: [SharedMember]
    ) throws {
        var seenConsumers = Set<String>()
        var signatureCount = 0
        for binding in bindings {
            guard binding.fileOrdinal >= 0,
                  binding.fileOrdinal < files.count,
                  binding.length > 0,
                  let end = optionalCheckedAdd(binding.start, binding.length),
                  end <= files[binding.fileOrdinal].fileBytes
            else {
                throw Failure.completeStream(.retainedStoreShape)
            }
            switch binding.consumerKind {
            case .setHeader, .mappingTable:
                guard binding.consumerOrdinal == binding.fileOrdinal else {
                    throw Failure.completeStream(.retainedStoreShape)
                }
            case .imageTable, .imageNameWindow:
                guard binding.consumerOrdinal == 0 else {
                    throw Failure.completeStream(.retainedStoreShape)
                }
            case .installName, .machOHeader, .loadCommands, .codeSignature:
                guard binding.consumerOrdinal >= 0,
                      binding.consumerOrdinal < members.count
                else {
                    throw Failure.completeStream(.retainedStoreShape)
                }
            }
            let consumerKey = "\(binding.consumerKind.rawValue):" +
                "\(binding.consumerOrdinal)"
            guard seenConsumers.insert(consumerKey).inserted else {
                throw Failure.completeStream(.retainedStoreShape)
            }
            if binding.consumerKind == .codeSignature {
                guard let nextSignatureCount = optionalCheckedAdd(
                    signatureCount,
                    1
                ) else {
                    throw Failure.completeStream(.retainedStoreShape)
                }
                signatureCount = nextSignatureCount
            }
            guard bindingPiecesAreInRetainedRanges(
                binding,
                discoveryStores: discoveryStores,
                additionalRanges: additionalRanges
            ) else {
                throw Failure.completeStream(.retainedStoreShape)
            }
        }

        for file in files {
            let mappingStart = UInt64(file.header.mappingOffset)
            guard file.header.mappingTablesEnd >= mappingStart else {
                throw Failure.completeStream(.retainedStoreShape)
            }
            guard hasExactBinding(
                .setHeader,
                ordinal: file.ordinal,
                fileOrdinal: file.ordinal,
                start: 0,
                length: 552,
                in: bindings
            ), hasExactBinding(
                .mappingTable,
                ordinal: file.ordinal,
                fileOrdinal: file.ordinal,
                start: mappingStart,
                length: file.header.mappingTablesEnd - mappingStart,
                in: bindings
            ) else {
                throw Failure.completeStream(.retainedStoreShape)
            }
        }
        guard let main = files.first,
              let imageTableLength = optionalCheckedMultiply(
                Int(main.header.imagesCount),
                32
              ),
              hasExactBinding(
                .imageTable,
                ordinal: 0,
                fileOrdinal: 0,
                start: main.header.imagesOffset,
                length: UInt64(imageTableLength),
                in: bindings
              ),
              bindings.contains(where: {
                $0.consumerKind == .imageNameWindow &&
                    $0.consumerOrdinal == 0
              })
        else {
            throw Failure.completeStream(.retainedStoreShape)
        }
        for member in members {
            for kind in [
                SyntheticSharedCacheRangeConsumerKind.installName,
                .machOHeader,
                .loadCommands,
            ] where !bindings.contains(where: {
                $0.consumerKind == kind &&
                    $0.consumerOrdinal == member.ordinal
            }) {
                throw Failure.completeStream(.retainedStoreShape)
            }
        }
        guard let twiceFiles = optionalCheckedMultiply(files.count, 2),
              let fixedCount = optionalCheckedAdd(twiceFiles, 2),
              let thriceMembers = optionalCheckedMultiply(members.count, 3),
              let withoutSignatures = optionalCheckedAdd(
                fixedCount,
                thriceMembers
              ),
              let expectedCount = optionalCheckedAdd(
                withoutSignatures,
                signatureCount
              ),
              bindings.count == expectedCount
        else {
            throw Failure.completeStream(.retainedStoreShape)
        }
    }

    static func hasExactBinding(
        _ kind: SyntheticSharedCacheRangeConsumerKind,
        ordinal: Int,
        fileOrdinal: Int,
        start: UInt64,
        length: UInt64,
        in bindings: [RangeComparison.ConsumerBinding]
    ) -> Bool {
        bindings.contains {
            $0.consumerKind == kind &&
                $0.consumerOrdinal == ordinal &&
                $0.fileOrdinal == fileOrdinal &&
                $0.start == start &&
                $0.length == length
        }
    }

    static func bindingPiecesAreInRetainedRanges(
        _ binding: RangeComparison.ConsumerBinding,
        discoveryStores: [RangeComparison.DiscoveryStore],
        additionalRanges: [RangeComparison.AdditionalPhysicalRange]
    ) -> Bool {
        guard let bindingEnd = optionalCheckedAdd(
            binding.start,
            binding.length
        ) else {
            return false
        }
        var cursor = binding.start
        for piece in binding.pieces {
            let pieceStart: UInt64
            let length: UInt64
            switch piece {
            case let .discovery(fileOrdinal, relativeStart, pieceLength):
                guard fileOrdinal == binding.fileOrdinal,
                      fileOrdinal >= 0,
                      fileOrdinal < discoveryStores.count,
                      let relativeEnd = optionalCheckedAdd(
                        relativeStart,
                        pieceLength
                      ),
                      relativeEnd <= UInt64(
                        discoveryStores[fileOrdinal].bytes.count
                      )
                else {
                    return false
                }
                pieceStart = relativeStart
                length = pieceLength
            case let .additional(rangeOrdinal, relativeStart, pieceLength):
                guard rangeOrdinal >= 0,
                      rangeOrdinal < additionalRanges.count,
                      additionalRanges[rangeOrdinal].fileOrdinal ==
                        binding.fileOrdinal,
                      let relativeEnd = optionalCheckedAdd(
                        relativeStart,
                        pieceLength
                      ),
                      relativeEnd <= additionalRanges[rangeOrdinal].length,
                      let start = optionalCheckedAdd(
                        additionalRanges[rangeOrdinal].start,
                        relativeStart
                      )
                else {
                    return false
                }
                pieceStart = start
                length = pieceLength
            }
            guard length > 0,
                  pieceStart == cursor,
                  let next = optionalCheckedAdd(cursor, length),
                  next <= bindingEnd
            else {
                return false
            }
            cursor = next
        }
        return cursor == bindingEnd
    }

    static func replayCompleteFile(
        transcript: SyntheticSharedCacheCompleteFileTranscript,
        file: HeaderOrderFile,
        expectedRecord: SyntheticSharedCacheFileRecord,
        retainedRanges inputRetainedRanges: [RetainedRange]
    ) throws -> HeaderStreamFact {
        var retainedRanges = inputRetainedRanges
        let eventCount = transcript.events.count
        var hasher = SHA256()
        var expectedOffset: UInt64 = 0
        var chunkEnd = min(logicalChunkBytes, file.fileBytes)
        var fragmentsInChunk = 0
        var interruptionsAtOffset = 0
        var acceptedEOF = false
        var matchedRetainedBytes: UInt64 = 0

        for eventOrdinal in 0..<eventCount {
            guard transcript.events.count == eventCount else {
                throw Failure.completeStream(.eventCountDrift(
                    fileOrdinal: file.ordinal
                ))
            }
            if acceptedEOF {
                throw Failure.unexpectedTrailingByte(
                    fileOrdinal: file.ordinal,
                    eventOrdinal: eventOrdinal
                )
            }
            let event = transcript.events[eventOrdinal]
            if event.offset > expectedOffset {
                throw Failure.completeStream(.offsetGap(
                    fileOrdinal: file.ordinal,
                    eventOrdinal: eventOrdinal,
                    expected: expectedOffset,
                    actual: event.offset
                ))
            }
            if event.offset < expectedOffset {
                throw Failure.completeStream(.offsetOverlapOrReorder(
                    fileOrdinal: file.ordinal,
                    eventOrdinal: eventOrdinal,
                    expected: expectedOffset,
                    actual: event.offset
                ))
            }

            if case .interrupted = event, expectedOffset == file.fileBytes {
                throw Failure.completeStream(.interruptionAfterFinalBytes(
                    fileOrdinal: file.ordinal,
                    eventOrdinal: eventOrdinal
                ))
            }

            switch event {
            case .interrupted:
                interruptionsAtOffset += 1
                guard interruptionsAtOffset <=
                    maximumInterruptedRetriesPerOffset
                else {
                    throw Failure.readInterruptedLimit(
                        fileOrdinal: file.ordinal,
                        eventOrdinal: eventOrdinal
                    )
                }

            case let .error(_, code):
                throw Failure.readError(
                    fileOrdinal: file.ordinal,
                    eventOrdinal: eventOrdinal,
                    code: code
                )

            case .endOfFile:
                guard expectedOffset == file.fileBytes else {
                    throw Failure.completeStream(.earlyEOF(
                        fileOrdinal: file.ordinal,
                        eventOrdinal: eventOrdinal,
                        expected: file.fileBytes,
                        actual: expectedOffset
                    ))
                }
                acceptedEOF = true

            case let .bytes(_, data):
                let payloadCount = data.count
                guard payloadCount > 0,
                      payloadCount <= maximumFragmentSnapshotBytes
                else {
                    throw Failure.completeStream(.fragmentSize(
                        fileOrdinal: file.ordinal,
                        eventOrdinal: eventOrdinal,
                        bytes: payloadCount
                    ))
                }
                fragmentsInChunk += 1
                guard fragmentsInChunk <=
                    maximumPositiveFragmentsPerChunk
                else {
                    throw Failure.readFragmentLimit(
                        fileOrdinal: file.ordinal,
                        eventOrdinal: eventOrdinal
                    )
                }
                guard let payloadBytes = UInt64(exactly: payloadCount),
                      let next = optionalCheckedAdd(
                        expectedOffset,
                        payloadBytes
                      )
                else {
                    throw Failure.completeStream(.resourceArithmetic)
                }
                guard next <= file.fileBytes else {
                    throw Failure.completeStream(.fragmentPastMetadata(
                        fileOrdinal: file.ordinal,
                        eventOrdinal: eventOrdinal,
                        fileBytes: file.fileBytes
                    ))
                }
                guard next <= chunkEnd else {
                    throw Failure.completeStream(.fragmentBeyondChunk(
                        fileOrdinal: file.ordinal,
                        eventOrdinal: eventOrdinal,
                        chunkEnd: chunkEnd
                    ))
                }

                var snapshot = Data()
                snapshot.reserveCapacity(payloadCount)
                snapshot.append(data)
                guard transcript.events.count == eventCount else {
                    throw Failure.completeStream(.eventCountDrift(
                        fileOrdinal: file.ordinal
                    ))
                }
                guard data.count == payloadCount else {
                    throw Failure.completeStream(.payloadCountDrift(
                        fileOrdinal: file.ordinal,
                        eventOrdinal: eventOrdinal
                    ))
                }
                hasher.update(data: snapshot)
                matchedRetainedBytes = try checkedAdd(
                    matchedRetainedBytes,
                    compareRetainedIntersections(
                        snapshot: snapshot,
                        fragmentStart: expectedOffset,
                        fragmentEnd: next,
                        fileOrdinal: file.ordinal,
                        retainedRanges: &retainedRanges
                    )
                )
                expectedOffset = next
                interruptionsAtOffset = 0
                if expectedOffset == chunkEnd, expectedOffset < file.fileBytes {
                    chunkEnd = min(
                        try checkedAdd(expectedOffset, logicalChunkBytes),
                        file.fileBytes
                    )
                    fragmentsInChunk = 0
                }
            }
        }
        guard acceptedEOF else {
            throw Failure.completeStream(.missingEOF(
                fileOrdinal: file.ordinal,
                offset: expectedOffset
            ))
        }
        for range in retainedRanges where range.comparedUntil != range.end {
            throw Failure.completeStream(.retainedCoverage(
                fileOrdinal: file.ordinal
            ))
        }
        if let field = firstMetadataDrift(file.metadata, transcript.afterMetadata) {
            throw Failure.metadataDrift(
                fileOrdinal: file.ordinal,
                position: .after,
                field: field
            )
        }
        guard expectedOffset == expectedRecord.fileBytes else {
            throw Failure.expectedByteCount(
                fileOrdinal: file.ordinal,
                expected: expectedRecord.fileBytes,
                actual: expectedOffset
            )
        }
        let actualSHA256 = hasher.finalize()
            .map { String(format: "%02x", $0) }
            .joined()
        guard isLowercaseHex(expectedRecord.fileSHA256, count: 64) else {
            throw Failure.expectedSHA256(fileOrdinal: file.ordinal, .format)
        }
        guard actualSHA256 == expectedRecord.fileSHA256 else {
            throw Failure.expectedSHA256(
                fileOrdinal: file.ordinal,
                .mismatch(
                    expected: expectedRecord.fileSHA256,
                    actual: actualSHA256
                )
            )
        }
        return HeaderStreamFact(
            headerOrderOrdinal: file.ordinal,
            decodedSuffix: Data(file.decodedSuffix),
            suffixBytes: file.suffixByteCount,
            suffixBase64URL: file.suffixBase64URL,
            headerUUID: file.headerUUID,
            fileSHA256: actualSHA256,
            fileBytes: expectedOffset,
            matchedRetainedBytes: matchedRetainedBytes
        )
    }

    static func compareRetainedIntersections(
        snapshot: Data,
        fragmentStart: UInt64,
        fragmentEnd: UInt64,
        fileOrdinal: Int,
        retainedRanges: inout [RetainedRange]
    ) throws -> UInt64 {
        var matched: UInt64 = 0
        for index in retainedRanges.indices {
            if retainedRanges[index].end <= fragmentStart {
                guard retainedRanges[index].comparedUntil ==
                    retainedRanges[index].end
                else {
                    throw Failure.completeStream(.retainedCoverage(
                        fileOrdinal: fileOrdinal
                    ))
                }
                continue
            }
            if retainedRanges[index].start >= fragmentEnd { break }
            let lower = max(fragmentStart, retainedRanges[index].start)
            let upper = min(fragmentEnd, retainedRanges[index].end)
            guard lower < upper,
                  lower == retainedRanges[index].comparedUntil,
                  let snapshotLower = Int(exactly: lower - fragmentStart),
                  let snapshotUpper = Int(exactly: upper - fragmentStart),
                  let rangeLower = Int(exactly:
                    lower - retainedRanges[index].start
                  ),
                  let rangeUpper = Int(exactly:
                    upper - retainedRanges[index].start
                  )
            else {
                throw Failure.completeStream(.retainedCoverage(
                    fileOrdinal: fileOrdinal
                ))
            }
            guard snapshot[snapshotLower..<snapshotUpper].elementsEqual(
                retainedRanges[index].bytes[rangeLower..<rangeUpper]
            ) else {
                throw Failure.completeStream(.retainedByteMismatch(
                    fileOrdinal: fileOrdinal,
                    offset: lower
                ))
            }
            retainedRanges[index].comparedUntil = upper
            matched = try checkedAdd(matched, upper - lower)
        }
        return matched
    }

    static func deriveCacheSetEvidence(
        facts: [HeaderStreamFact],
        git: AnchoredRuntimeClosureExpectationDocument,
        selfGuard: AnchoredRuntimeClosureExpectationDocument
    ) throws -> SyntheticSharedCacheSetIdentityEvidence {
        guard let main = facts.first, main.decodedSuffix.isEmpty else {
            throw Failure.cacheSetIdentity(.mismatch)
        }
        var records: [SyntheticSharedCacheFileRecord] = [
            record(from: main),
        ]
        let sortedSubcaches = facts.dropFirst().sorted {
            $0.decodedSuffix.lexicographicallyPrecedes($1.decodedSuffix)
        }
        records.append(contentsOf: sortedSubcaches.map(record(from:)))
        let evidence: SyntheticSharedCacheSetIdentityEvidence
        do {
            evidence = try SyntheticSharedCacheSetIdentityVerifier.derive(
                records: records
            )
        } catch {
            throw Failure.cacheSetIdentity(.derivation)
        }
        guard evidence == git.sharedCacheSetEvidence,
              evidence == selfGuard.sharedCacheSetEvidence
        else {
            throw Failure.cacheSetIdentity(.mismatch)
        }
        return evidence
    }

    static func record(from fact: HeaderStreamFact)
        -> SyntheticSharedCacheFileRecord
    {
        SyntheticSharedCacheFileRecord(
            suffixBytes: fact.suffixBytes,
            suffixBase64URL: fact.suffixBase64URL,
            fileSHA256: fact.fileSHA256,
            fileBytes: fact.fileBytes,
            headerUUID: fact.headerUUID
        )
    }

    static func deriveImageEvidence(
        cacheSetEvidence: SyntheticSharedCacheSetIdentityEvidence,
        rangePlan: RangeComparison,
        files: [HeaderOrderFile],
        members: [SharedMember]
    ) throws -> [SyntheticSharedCacheImageContentIdentityEvidence] {
        let mappings = try cacheMappings(files: files)
        try validateExactImageBindings(
            rangePlan: rangePlan,
            files: files,
            members: members,
            mappings: mappings
        )
        var result: [SyntheticSharedCacheImageContentIdentityEvidence] = []
        result.reserveCapacity(members.count)
        for member in members {
            let facts = try rederiveImageFacts(
                member: member,
                rangePlan: rangePlan,
                mappings: mappings
            )
            let evidence: SyntheticSharedCacheImageContentIdentityEvidence
            do {
                evidence =
                    try SyntheticSharedCacheImageContentIdentityVerifier
                    .derive(
                        cacheSetEvidence: cacheSetEvidence,
                        facts: facts
                    )
            } catch {
                throw Failure.cacheImageIdentity(
                    memberOrdinal: member.ordinal,
                    .derivation
                )
            }
            guard evidence.contentEvidenceID.sha256 ==
                    member.field.contentEvidenceID,
                  evidence.decodedInstallName ==
                    member.field.decodedInstallName,
                  evidence.primaryCodeDirectoryBlobSHA256 ==
                    member.field.primaryCodeDirectoryBlobSHA256,
                  evidence.facts.machOUUID == member.field.machOUUID,
                  evidence.facts.loadCommandsSHA256 ==
                    member.field.loadCommandsSHA256
            else {
                throw Failure.cacheImageIdentity(
                    memberOrdinal: member.ordinal,
                    .mismatch
                )
            }
            result.append(evidence)
        }
        return result
    }

    static func validateExactImageBindings(
        rangePlan: RangeComparison,
        files: [HeaderOrderFile],
        members: [SharedMember],
        mappings: [Mapping]
    ) throws {
        guard let main = files.first,
              let imageTableLength = optionalCheckedMultiply(
                Int(main.header.imagesCount),
                32
              )
        else {
            throw Failure.cacheImageFormat(memberOrdinal: 0, .binding)
        }
        let imageTable = try exactBinding(
            .imageTable,
            ordinal: 0,
            fileOrdinal: 0,
            start: main.header.imagesOffset,
            length: UInt64(imageTableLength),
            rangePlan: rangePlan,
            memberOrdinal: 0
        )
        var rows: [ImageRow] = []
        rows.reserveCapacity(Int(main.header.imagesCount))
        for row in 0..<Int(main.header.imagesCount) {
            guard let rowOffset = optionalCheckedMultiply(row, 32),
                  let pathOffset = optionalCheckedAdd(rowOffset, 24),
                  let paddingOffset = optionalCheckedAdd(rowOffset, 28),
                  let address = readUInt64LE(
                    imageTable,
                    at: rowOffset,
                    rangePlan: rangePlan
                  ),
                  let pathFileOffset = readUInt32LE(
                    imageTable,
                    at: pathOffset,
                    rangePlan: rangePlan
                  ),
                  readUInt32LE(
                    imageTable,
                    at: paddingOffset,
                    rangePlan: rangePlan
                  ) == 0
            else {
                throw Failure.cacheImageFormat(memberOrdinal: 0, .binding)
            }
            rows.append(ImageRow(
                address: address,
                pathFileOffset: UInt64(pathFileOffset)
            ))
        }
        guard let windowStart = rows.map(\.pathFileOffset).min() else {
            throw Failure.cacheImageFormat(memberOrdinal: 0, .binding)
        }
        var candidateWindowEnd: UInt64 = 0
        for row in rows {
            guard row.pathFileOffset < main.fileBytes,
                  let rowMaximumEnd = optionalCheckedAdd(
                    row.pathFileOffset,
                    4_097
                  )
            else {
                throw Failure.cacheImageFormat(memberOrdinal: 0, .binding)
            }
            candidateWindowEnd = max(candidateWindowEnd, rowMaximumEnd)
        }
        let windowEnd = min(candidateWindowEnd, main.fileBytes)
        guard windowStart < windowEnd else {
            throw Failure.cacheImageFormat(memberOrdinal: 0, .binding)
        }
        let nameWindow = try exactBinding(
            .imageNameWindow,
            ordinal: 0,
            fileOrdinal: 0,
            start: windowStart,
            length: windowEnd - windowStart,
            rangePlan: rangePlan,
            memberOrdinal: 0
        )

        var usedAddresses = Set<UInt64>()
        for member in members {
            let matches = rows.filter { row in
                guard row.pathFileOffset >= windowStart,
                      let relativeStart = Int(exactly:
                        row.pathFileOffset - windowStart
                      )
                else {
                    return false
                }
                return bindingBytesEqual(
                    nameWindow,
                    at: relativeStart,
                    expected: member.field.decodedInstallName,
                    followedByNUL: true,
                    rangePlan: rangePlan
                )
            }
            guard matches.count == 1,
                  usedAddresses.insert(matches[0].address).inserted
            else {
                throw Failure.cacheImageFormat(
                    memberOrdinal: member.ordinal,
                    .installName
                )
            }
            let selected = matches[0]
            _ = try exactBinding(
                .installName,
                ordinal: member.ordinal,
                fileOrdinal: 0,
                start: selected.pathFileOffset,
                length: UInt64(member.field.decodedInstallName.count),
                rangePlan: rangePlan,
                memberOrdinal: member.ordinal
            )
            let headerTranslation = try translate(
                vmStart: selected.address,
                length: 32,
                mappings: mappings,
                memberOrdinal: member.ordinal
            )
            let headerBinding = try exactBinding(
                .machOHeader,
                ordinal: member.ordinal,
                fileOrdinal: headerTranslation.fileOrdinal,
                start: headerTranslation.fileStart,
                length: 32,
                rangePlan: rangePlan,
                memberOrdinal: member.ordinal
            )
            let header = try parseMachHeader(
                ownedBytes(
                    headerBinding,
                    rangePlan: rangePlan,
                    memberOrdinal: member.ordinal
                ),
                memberOrdinal: member.ordinal
            )
            guard let commandAddress = optionalCheckedAdd(
                selected.address,
                32
            ) else {
                throw Failure.cacheImageFormat(
                    memberOrdinal: member.ordinal,
                    .binding
                )
            }
            let commandTranslation = try translate(
                vmStart: commandAddress,
                length: UInt64(header.sizeOfCommands),
                mappings: mappings,
                memberOrdinal: member.ordinal
            )
            _ = try exactBinding(
                .loadCommands,
                ordinal: member.ordinal,
                fileOrdinal: commandTranslation.fileOrdinal,
                start: commandTranslation.fileStart,
                length: UInt64(header.sizeOfCommands),
                rangePlan: rangePlan,
                memberOrdinal: member.ordinal
            )
        }
    }

    static func exactBinding(
        _ kind: SyntheticSharedCacheRangeConsumerKind,
        ordinal: Int,
        fileOrdinal: Int,
        start: UInt64,
        length: UInt64,
        rangePlan: RangeComparison,
        memberOrdinal: Int
    ) throws -> RangeComparison.ConsumerBinding {
        let matches = rangePlan.consumerBindings.filter {
            $0.consumerKind == kind && $0.consumerOrdinal == ordinal
        }
        guard matches.count == 1,
              matches[0].fileOrdinal == fileOrdinal,
              matches[0].start == start,
              matches[0].length == length
        else {
            throw Failure.cacheImageFormat(
                memberOrdinal: memberOrdinal,
                .binding
            )
        }
        return matches[0]
    }

    static func bindingBytesEqual(
        _ binding: RangeComparison.ConsumerBinding,
        at relativeStart: Int,
        expected: Data,
        followedByNUL: Bool,
        rangePlan: RangeComparison
    ) -> Bool {
        guard relativeStart >= 0,
              let expectedEnd = optionalCheckedAdd(
                relativeStart,
                expected.count
              ),
              let requiredEnd = optionalCheckedAdd(
                expectedEnd,
                followedByNUL ? 1 : 0
              ),
              UInt64(requiredEnd) <= binding.length
        else {
            return false
        }
        for (offset, byte) in expected.enumerated() {
            guard let sourceOffset = optionalCheckedAdd(
                relativeStart,
                offset
            ) else {
                return false
            }
            guard bindingByte(
                binding,
                at: sourceOffset,
                rangePlan: rangePlan
            ) == byte else {
                return false
            }
        }
        return !followedByNUL || bindingByte(
            binding,
            at: expectedEnd,
            rangePlan: rangePlan
        ) == 0
    }

    static func bindingByte(
        _ binding: RangeComparison.ConsumerBinding,
        at relativeOffset: Int,
        rangePlan: RangeComparison
    ) -> UInt8? {
        guard relativeOffset >= 0,
              UInt64(relativeOffset) < binding.length
        else {
            return nil
        }
        var pieceBase: UInt64 = 0
        let target = UInt64(relativeOffset)
        for piece in binding.pieces {
            let pieceLength: UInt64
            switch piece {
            case let .discovery(_, _, length),
                 let .additional(_, _, length):
                pieceLength = length
            }
            guard let pieceEnd = optionalCheckedAdd(
                pieceBase,
                pieceLength
            ) else {
                return nil
            }
            defer { pieceBase = pieceEnd }
            guard target >= pieceBase, target < pieceEnd else { continue }
            let withinPiece = target - pieceBase
            switch piece {
            case let .discovery(fileOrdinal, relativeStart, _):
                guard fileOrdinal >= 0,
                      fileOrdinal < rangePlan.discoveryStores.count,
                      let sourceOffset = optionalCheckedAdd(
                        relativeStart,
                        withinPiece
                      ),
                      let index = Int(exactly: sourceOffset),
                      index < rangePlan.discoveryStores[fileOrdinal]
                        .bytes.count
                else {
                    return nil
                }
                return rangePlan.discoveryStores[fileOrdinal].bytes[index]
            case let .additional(rangeOrdinal, relativeStart, _):
                guard rangeOrdinal >= 0,
                      rangeOrdinal < rangePlan.additionalPhysicalRanges.count,
                      let sourceOffset = optionalCheckedAdd(
                        relativeStart,
                        withinPiece
                      ),
                      let index = Int(exactly: sourceOffset),
                      index < rangePlan.additionalPhysicalRanges[rangeOrdinal]
                        .bytes.count
                else {
                    return nil
                }
                return rangePlan.additionalPhysicalRanges[rangeOrdinal]
                    .bytes[index]
            }
        }
        return nil
    }

    static func readUInt32LE(
        _ binding: RangeComparison.ConsumerBinding,
        at offset: Int,
        rangePlan: RangeComparison
    ) -> UInt32? {
        var value: UInt32 = 0
        for index in 0..<4 {
            guard let sourceOffset = optionalCheckedAdd(offset, index) else {
                return nil
            }
            guard let byte = bindingByte(
                binding,
                at: sourceOffset,
                rangePlan: rangePlan
            ) else {
                return nil
            }
            value |= UInt32(byte) << UInt32(index * 8)
        }
        return value
    }

    static func readUInt64LE(
        _ binding: RangeComparison.ConsumerBinding,
        at offset: Int,
        rangePlan: RangeComparison
    ) -> UInt64? {
        var value: UInt64 = 0
        for index in 0..<8 {
            guard let sourceOffset = optionalCheckedAdd(offset, index) else {
                return nil
            }
            guard let byte = bindingByte(
                binding,
                at: sourceOffset,
                rangePlan: rangePlan
            ) else {
                return nil
            }
            value |= UInt64(byte) << UInt64(index * 8)
        }
        return value
    }

    static func rederiveImageFacts(
        member: SharedMember,
        rangePlan: RangeComparison,
        mappings: [Mapping]
    ) throws -> SyntheticSharedCacheImageContentFacts {
        let installName = try bytesForRequiredBinding(
            .installName,
            ordinal: member.ordinal,
            rangePlan: rangePlan,
            memberOrdinal: member.ordinal
        )
        guard installName == member.field.decodedInstallName else {
            throw Failure.cacheImageFormat(
                memberOrdinal: member.ordinal,
                .installName
            )
        }
        let headerBytes = try bytesForRequiredBinding(
            .machOHeader,
            ordinal: member.ordinal,
            rangePlan: rangePlan,
            memberOrdinal: member.ordinal
        )
        let header = try parseMachHeader(
            headerBytes,
            memberOrdinal: member.ordinal
        )
        let loadCommands = try bytesForRequiredBinding(
            .loadCommands,
            ordinal: member.ordinal,
            rangePlan: rangePlan,
            memberOrdinal: member.ordinal
        )
        let loadFacts = try parseLoadCommands(
            loadCommands,
            header: header,
            member: member,
            mappings: mappings
        )
        let signatureBindings = rangePlan.consumerBindings.filter {
            $0.consumerKind == .codeSignature &&
                $0.consumerOrdinal == member.ordinal
        }
        let primaryCodeDirectory:
            SyntheticSharedCacheImagePrimaryCodeDirectory
        if let request = loadFacts.signature {
            guard signatureBindings.count == 1,
                  signatureBindings[0].fileOrdinal == request.fileOrdinal,
                  signatureBindings[0].start == request.fileStart,
                  signatureBindings[0].length == request.length
            else {
                throw Failure.cacheImageFormat(
                    memberOrdinal: member.ordinal,
                    .codeSignature
                )
            }
            let signatureBytes = try ownedBytes(
                signatureBindings[0],
                rangePlan: rangePlan,
                memberOrdinal: member.ordinal
            )
            let primarySHA256 = try validateSignature(
                signatureBytes,
                member: member
            )
            primaryCodeDirectory = .present(blobSHA256: primarySHA256)
        } else {
            guard signatureBindings.isEmpty,
                  member.field.primaryCodeDirectoryBlobSHA256 ==
                    String(repeating: "0", count: 64)
            else {
                throw Failure.cacheImageFormat(
                    memberOrdinal: member.ordinal,
                    .expectedMember
                )
            }
            primaryCodeDirectory = .absent
        }
        guard hex(loadFacts.uuid) == member.field.machOUUID,
              loadFacts.loadCommandsSHA256 ==
                member.field.loadCommandsSHA256
        else {
            throw Failure.cacheImageFormat(
                memberOrdinal: member.ordinal,
                .expectedMember
            )
        }
        return SyntheticSharedCacheImageContentFacts(
            installNameBytes: UInt64(installName.count),
            installNameBase64URL: base64URL(installName),
            machOUUID: hex(loadFacts.uuid),
            primaryCodeDirectory: primaryCodeDirectory,
            loadCommandsSHA256: loadFacts.loadCommandsSHA256
        )
    }

    static func parseMachHeader(
        _ bytes: Data,
        memberOrdinal: Int
    ) throws -> MachHeaderFacts {
        guard bytes.count == 32,
              readUInt32LE(bytes, at: 0) == 0xfeedfacf,
              readUInt32LE(bytes, at: 4) == 0x0100000c,
              readUInt32LE(bytes, at: 12) == 0x6,
              let commandCount = readUInt32LE(bytes, at: 16),
              commandCount > 0,
              commandCount <= 256,
              let commandBytes = readUInt32LE(bytes, at: 20),
              commandBytes > 0,
              commandBytes <= 262_144,
              readUInt32LE(bytes, at: 28) == 0
        else {
            throw Failure.cacheImageFormat(
                memberOrdinal: memberOrdinal,
                .machOHeader
            )
        }
        return MachHeaderFacts(
            commandCount: commandCount,
            sizeOfCommands: commandBytes
        )
    }

    static func parseLoadCommands(
        _ bytes: Data,
        header: MachHeaderFacts,
        member: SharedMember,
        mappings: [Mapping]
    ) throws -> LoadCommandFacts {
        let memberOrdinal = member.ordinal
        guard bytes.count == Int(header.sizeOfCommands) else {
            throw Failure.cacheImageFormat(
                memberOrdinal: memberOrdinal,
                .loadCommandFrame
            )
        }
        var cursor = 0
        var uuid: Data?
        var sawDylibID = false
        var signature: (offset: UInt64, size: UInt64)?
        var linkedits: [LinkeditFacts] = []
        for ordinal in 0..<Int(header.commandCount) {
            guard let commandEnd = optionalCheckedAdd(cursor, 8),
                  commandEnd <= bytes.count,
                  let command = readUInt32LE(bytes, at: cursor),
                  let rawSize = readUInt32LE(bytes, at: cursor + 4),
                  let size = Int(exactly: rawSize),
                  size >= 8,
                  size.isMultiple(of: 8),
                  let frameEnd = optionalCheckedAdd(cursor, size),
                  frameEnd <= bytes.count
            else {
                throw Failure.cacheImageFormat(
                    memberOrdinal: memberOrdinal,
                    .loadCommandFrame
                )
            }
            guard recognizedLoadCommands.contains(command) else {
                throw Failure.cacheImageFormat(
                    memberOrdinal: memberOrdinal,
                    .unknownLoadCommand
                )
            }
            guard !forbiddenLoadCommands.contains(command) else {
                throw Failure.cacheImageFormat(
                    memberOrdinal: memberOrdinal,
                    .forbiddenLoadCommand
                )
            }
            switch command {
            case 0x1b:
                guard size == 24, uuid == nil else {
                    throw Failure.cacheImageFormat(
                        memberOrdinal: memberOrdinal,
                        .uuid
                    )
                }
                let value = Data(bytes[(cursor + 8)..<(cursor + 24)])
                guard value.contains(where: { $0 != 0 }) else {
                    throw Failure.cacheImageFormat(
                        memberOrdinal: memberOrdinal,
                        .uuid
                    )
                }
                uuid = value
            case 0x0d:
                guard !sawDylibID,
                      validDylibIdentity(bytes, range: cursor..<frameEnd)
                else {
                    throw Failure.cacheImageFormat(
                        memberOrdinal: memberOrdinal,
                        .idDylibCommand
                    )
                }
                sawDylibID = true
            case 0x1d:
                guard size == 16,
                      signature == nil,
                      let dataOffset = readUInt32LE(bytes, at: cursor + 8),
                      let dataSize = readUInt32LE(bytes, at: cursor + 12),
                      dataSize > 0,
                      dataSize <= 262_144
                else {
                    throw Failure.cacheImageFormat(
                        memberOrdinal: memberOrdinal,
                        .codeSignature
                    )
                }
                signature = (UInt64(dataOffset), UInt64(dataSize))
            case 0x19:
                if let linkedit = try parseLinkedit(
                    bytes,
                    range: cursor..<frameEnd,
                    memberOrdinal: memberOrdinal
                ) {
                    linkedits.append(linkedit)
                }
            default:
                break
            }
            cursor = frameEnd
            _ = ordinal
        }
        guard cursor == bytes.count else {
            throw Failure.cacheImageFormat(
                memberOrdinal: memberOrdinal,
                .loadCommandFrame
            )
        }
        guard let uuid else {
            throw Failure.cacheImageFormat(memberOrdinal: memberOrdinal, .uuid)
        }
        guard sawDylibID else {
            throw Failure.cacheImageFormat(
                memberOrdinal: memberOrdinal,
                .idDylibCommand
            )
        }
        let loadCommandsSHA256 = sha256Hex(bytes)
        guard hex(uuid) == member.field.machOUUID,
              loadCommandsSHA256 == member.field.loadCommandsSHA256
        else {
            throw Failure.cacheImageFormat(
                memberOrdinal: memberOrdinal,
                .expectedMember
            )
        }
        guard let signature else {
            guard member.field.primaryCodeDirectoryBlobSHA256 ==
                String(repeating: "0", count: 64)
            else {
                throw Failure.cacheImageFormat(
                    memberOrdinal: memberOrdinal,
                    .expectedMember
                )
            }
            return LoadCommandFacts(
                uuid: uuid,
                loadCommandsSHA256: loadCommandsSHA256,
                signature: nil
            )
        }
        guard linkedits.count == 1 else {
            throw Failure.cacheImageFormat(
                memberOrdinal: memberOrdinal,
                .linkedit
            )
        }
        let linkedit = linkedits[0]
        guard signature.offset >= linkedit.fileOffset,
              let signatureFileEnd = optionalCheckedAdd(
                signature.offset,
                signature.size
              ),
              let linkeditFileEnd = optionalCheckedAdd(
                linkedit.fileOffset,
                linkedit.fileSize
              ),
              signatureFileEnd <= linkeditFileEnd
        else {
            throw Failure.cacheImageFormat(
                memberOrdinal: memberOrdinal,
                .linkedit
            )
        }
        let delta = signature.offset - linkedit.fileOffset
        guard let vmStart = optionalCheckedAdd(linkedit.vmAddress, delta),
              let vmEnd = optionalCheckedAdd(vmStart, signature.size),
              let linkeditVMEnd = optionalCheckedAdd(
                linkedit.vmAddress,
                linkedit.vmSize
              ),
              vmEnd <= linkeditVMEnd
        else {
            throw Failure.cacheImageFormat(
                memberOrdinal: memberOrdinal,
                .linkedit
            )
        }
        let translated = try translate(
            vmStart: vmStart,
            length: signature.size,
            mappings: mappings,
            memberOrdinal: memberOrdinal
        )
        return LoadCommandFacts(
            uuid: uuid,
            loadCommandsSHA256: loadCommandsSHA256,
            signature: SignatureRequest(
                fileOrdinal: translated.fileOrdinal,
                fileStart: translated.fileStart,
                length: signature.size
            )
        )
    }

    static func validateSignature(
        _ bytes: Data,
        member: SharedMember
    ) throws -> String {
        do {
            let parsed = try SyntheticMachOIdentityParser
                .parseEmbeddedSignatureForFileTypeIdentity(bytes)
            guard let primary = parsed.codeDirectories.first,
                  primary.blobSHA256 ==
                    member.field.primaryCodeDirectoryBlobSHA256
            else {
                throw Failure.cacheImageFormat(
                    memberOrdinal: member.ordinal,
                    .expectedMember
                )
            }
            return primary.blobSHA256
        } catch let failure as Failure {
            throw failure
        } catch {
            throw Failure.cacheImageFormat(
                memberOrdinal: member.ordinal,
                .signatureParse
            )
        }
    }

    static func cacheMappings(
        files: [HeaderOrderFile]
    ) throws -> [Mapping] {
        var result: [Mapping] = []
        for file in files {
            for fact in file.header.mappings {
                guard let vmEnd = optionalCheckedAdd(
                    fact.address,
                    fact.size
                ),
                      let fileEnd = optionalCheckedAdd(
                        fact.fileOffset,
                        fact.size
                      ),
                      fileEnd <= file.fileBytes
                else {
                    throw Failure.cacheImageFormat(
                        memberOrdinal: 0,
                        .binding
                    )
                }
                result.append(Mapping(
                    fileOrdinal: file.ordinal,
                    vmStart: fact.address,
                    vmEnd: vmEnd,
                    fileStart: fact.fileOffset,
                    fileEnd: fileEnd
                ))
            }
        }
        result.sort {
            if $0.vmStart != $1.vmStart { return $0.vmStart < $1.vmStart }
            if $0.vmEnd != $1.vmEnd { return $0.vmEnd < $1.vmEnd }
            return $0.fileOrdinal < $1.fileOrdinal
        }
        for index in result.indices.dropFirst() {
            guard result[index - 1].vmEnd <= result[index].vmStart else {
                throw Failure.cacheImageFormat(
                    memberOrdinal: 0,
                    .binding
                )
            }
        }
        return result
    }

    static func translate(
        vmStart: UInt64,
        length: UInt64,
        mappings: [Mapping],
        memberOrdinal: Int
    ) throws -> Translation {
        guard length > 0,
              let vmEnd = optionalCheckedAdd(vmStart, length)
        else {
            throw Failure.cacheImageFormat(
                memberOrdinal: memberOrdinal,
                .binding
            )
        }
        let owners = mappings.filter {
            $0.vmStart <= vmStart && vmEnd <= $0.vmEnd
        }
        guard owners.count == 1 else {
            throw Failure.cacheImageFormat(
                memberOrdinal: memberOrdinal,
                .binding
            )
        }
        let owner = owners[0]
        let delta = vmStart - owner.vmStart
        guard let fileStart = optionalCheckedAdd(owner.fileStart, delta),
              let fileEnd = optionalCheckedAdd(fileStart, length),
              fileEnd <= owner.fileEnd
        else {
            throw Failure.cacheImageFormat(
                memberOrdinal: memberOrdinal,
                .binding
            )
        }
        return Translation(
            fileOrdinal: owner.fileOrdinal,
            fileStart: fileStart
        )
    }

    static func parseLinkedit(
        _ bytes: Data,
        range: Range<Int>,
        memberOrdinal: Int
    ) throws -> LinkeditFacts? {
        guard let sectionCountOffset = optionalCheckedAdd(
            range.lowerBound,
            64
        ),
              let nameStart = optionalCheckedAdd(range.lowerBound, 8),
              let nameEnd = optionalCheckedAdd(range.lowerBound, 24),
              let vmAddressOffset = optionalCheckedAdd(
                range.lowerBound,
                24
              ),
              let vmSizeOffset = optionalCheckedAdd(range.lowerBound, 32),
              let fileOffsetOffset = optionalCheckedAdd(
                range.lowerBound,
                40
              ),
              let fileSizeOffset = optionalCheckedAdd(
                range.lowerBound,
                48
              ),
              range.count >= 72,
              let sectionCount = readUInt32LE(
                bytes,
                at: sectionCountOffset
              ),
              let sectionBytes = optionalCheckedMultiply(
                Int(sectionCount),
                80
              ),
              let expectedSize = optionalCheckedAdd(72, sectionBytes),
              range.count == expectedSize
        else {
            throw Failure.cacheImageFormat(
                memberOrdinal: memberOrdinal,
                .linkedit
            )
        }
        let name = Data(bytes[nameStart..<nameEnd])
        let literal = Data("__LINKEDIT".utf8)
        guard name.prefix(literal.count) == literal,
              name.dropFirst(literal.count).allSatisfy({ $0 == 0 })
        else {
            return nil
        }
        guard let vmAddress = readUInt64LE(bytes, at: vmAddressOffset),
              let vmSize = readUInt64LE(bytes, at: vmSizeOffset),
              let fileOffset = readUInt64LE(bytes, at: fileOffsetOffset),
              let fileSize = readUInt64LE(bytes, at: fileSizeOffset),
              vmSize > 0,
              fileSize > 0,
              optionalCheckedAdd(fileOffset, fileSize) != nil
        else {
            throw Failure.cacheImageFormat(
                memberOrdinal: memberOrdinal,
                .linkedit
            )
        }
        return LinkeditFacts(
            vmAddress: vmAddress,
            vmSize: vmSize,
            fileOffset: fileOffset,
            fileSize: fileSize
        )
    }

    static func bytesForRequiredBinding(
        _ kind: SyntheticSharedCacheRangeConsumerKind,
        ordinal: Int,
        rangePlan: RangeComparison,
        memberOrdinal: Int
    ) throws -> Data {
        let matches = rangePlan.consumerBindings.filter {
            $0.consumerKind == kind && $0.consumerOrdinal == ordinal
        }
        guard matches.count == 1 else {
            throw Failure.cacheImageFormat(
                memberOrdinal: memberOrdinal,
                .binding
            )
        }
        return try ownedBytes(
            matches[0],
            rangePlan: rangePlan,
            memberOrdinal: memberOrdinal
        )
    }

    static func ownedBytes(
        _ binding: RangeComparison.ConsumerBinding,
        rangePlan: RangeComparison,
        memberOrdinal: Int
    ) throws -> Data {
        guard let capacity = Int(exactly: binding.length) else {
            throw Failure.cacheImageFormat(
                memberOrdinal: memberOrdinal,
                .binding
            )
        }
        var result = Data()
        result.reserveCapacity(capacity)
        var copied: UInt64 = 0
        for piece in binding.pieces {
            let source: Data
            let lower: Int
            let upper: Int
            let length: UInt64
            switch piece {
            case let .discovery(fileOrdinal, relativeStart, pieceLength):
                guard fileOrdinal == binding.fileOrdinal,
                      fileOrdinal >= 0,
                      fileOrdinal < rangePlan.discoveryStores.count,
                      let start = Int(exactly: relativeStart),
                      let count = Int(exactly: pieceLength),
                      let end = optionalCheckedAdd(start, count),
                      end <= rangePlan.discoveryStores[fileOrdinal].bytes.count
                else {
                    throw Failure.cacheImageFormat(
                        memberOrdinal: memberOrdinal,
                        .binding
                    )
                }
                source = rangePlan.discoveryStores[fileOrdinal].bytes
                lower = start
                upper = end
                length = pieceLength
            case let .additional(rangeOrdinal, relativeStart, pieceLength):
                guard rangeOrdinal >= 0,
                      rangeOrdinal < rangePlan.additionalPhysicalRanges.count,
                      rangePlan.additionalPhysicalRanges[rangeOrdinal]
                        .fileOrdinal == binding.fileOrdinal,
                      let start = Int(exactly: relativeStart),
                      let count = Int(exactly: pieceLength),
                      let end = optionalCheckedAdd(start, count),
                      end <= rangePlan.additionalPhysicalRanges[rangeOrdinal]
                        .bytes.count
                else {
                    throw Failure.cacheImageFormat(
                        memberOrdinal: memberOrdinal,
                        .binding
                    )
                }
                source = rangePlan.additionalPhysicalRanges[rangeOrdinal].bytes
                lower = start
                upper = end
                length = pieceLength
            }
            result.append(source[lower..<upper])
            copied = try checkedAdd(copied, length)
        }
        guard copied == binding.length else {
            throw Failure.cacheImageFormat(
                memberOrdinal: memberOrdinal,
                .binding
            )
        }
        return result
    }

    static func compactEvidenceSemanticBytes(
        cacheSetEvidence: SyntheticSharedCacheSetIdentityEvidence,
        imageEvidence: [SyntheticSharedCacheImageContentIdentityEvidence]
    ) throws -> UInt64 {
        var total = UInt64(cacheSetEvidence.identityPreimage.count)
        for suffix in cacheSetEvidence.decodedSuffixes {
            total = try checkedAdd(total, UInt64(suffix.count))
        }
        for image in imageEvidence {
            total = try checkedAdd(
                total,
                UInt64(image.identityPreimage.count)
            )
            total = try checkedAdd(
                total,
                UInt64(image.decodedInstallName.count)
            )
            total = try checkedAdd(
                total,
                UInt64(image.facts.installNameBase64URL.utf8.count)
            )
            total = try checkedAdd(total, 64)
            total = try checkedAdd(total, 64)
            total = try checkedAdd(total, 32)
        }
        return total
    }

    static func platformTuple(_ fields: RuntimeClosureExpectationFields)
        -> [String]
    {
        [
            fields.platformArchitecture,
            fields.platformHardwareModel,
            fields.platformOSVersion,
            fields.platformOSBuild,
            fields.resolutionProfile,
            fields.environmentProfile,
        ]
    }

    static func firstMetadataDrift(
        _ expected: SyntheticCaptureFileMetadata,
        _ actual: SyntheticCaptureFileMetadata
    ) -> SyntheticSmallArtifactCaptureFailure.MetadataField? {
        let checks: [
            (SyntheticSmallArtifactCaptureFailure.MetadataField, Bool)
        ] = [
            (.device, expected.device == actual.device),
            (.inode, expected.inode == actual.inode),
            (.mode, expected.mode == actual.mode),
            (.linkCount, expected.linkCount == actual.linkCount),
            (.userID, expected.userID == actual.userID),
            (.groupID, expected.groupID == actual.groupID),
            (.size, expected.size == actual.size),
            (.blockCount, expected.blockCount == actual.blockCount),
            (.blockSize, expected.blockSize == actual.blockSize),
            (.flags, expected.flags == actual.flags),
            (.generation, expected.generation == actual.generation),
            (.modificationTimeSeconds,
             expected.modificationTimeSeconds ==
                actual.modificationTimeSeconds),
            (.modificationTimeNanoseconds,
             expected.modificationTimeNanoseconds ==
                actual.modificationTimeNanoseconds),
            (.statusChangeTimeSeconds,
             expected.statusChangeTimeSeconds ==
                actual.statusChangeTimeSeconds),
            (.statusChangeTimeNanoseconds,
             expected.statusChangeTimeNanoseconds ==
                actual.statusChangeTimeNanoseconds),
            (.birthTimeSeconds,
             expected.birthTimeSeconds == actual.birthTimeSeconds),
            (.birthTimeNanoseconds,
             expected.birthTimeNanoseconds ==
                actual.birthTimeNanoseconds),
            (.extendedAttributeSupportMask,
             expected.extendedAttributeSupportMask ==
                actual.extendedAttributeSupportMask),
            (.extendedFlags,
             expected.extendedFlags == actual.extendedFlags),
            (.cloneID, expected.cloneID == actual.cloneID),
            (.cloneReferenceCount,
             expected.cloneReferenceCount == actual.cloneReferenceCount),
        ]
        return checks.first(where: { !$0.1 })?.0
    }

    static func chunkCount(_ bytes: UInt64) -> UInt64 {
        bytes / logicalChunkBytes +
            (bytes % logicalChunkBytes == 0 ? 0 : 1)
    }

    static func maximumAttempts(forChunkCount chunks: UInt64) throws
        -> UInt64
    {
        let fragments = try checkedMultiply(
            chunks,
            UInt64(maximumPositiveFragmentsPerChunk)
        )
        let attemptsWithoutEOF = try checkedMultiply(
            fragments,
            UInt64(maximumInterruptedRetriesPerOffset + 1)
        )
        return try checkedAdd(attemptsWithoutEOF, 1)
    }

    static func checkedAdd(_ left: UInt64, _ right: UInt64) throws
        -> UInt64
    {
        let (result, overflow) = left.addingReportingOverflow(right)
        guard !overflow else {
            throw Failure.completeStream(.resourceArithmetic)
        }
        return result
    }

    static func checkedMultiply(_ left: UInt64, _ right: UInt64) throws
        -> UInt64
    {
        let (result, overflow) = left.multipliedReportingOverflow(by: right)
        guard !overflow else {
            throw Failure.completeStream(.resourceArithmetic)
        }
        return result
    }

    static func optionalCheckedAdd(_ left: UInt64, _ right: UInt64)
        -> UInt64?
    {
        let (result, overflow) = left.addingReportingOverflow(right)
        return overflow ? nil : result
    }

    static func optionalCheckedAdd(_ left: Int, _ right: Int) -> Int? {
        let (result, overflow) = left.addingReportingOverflow(right)
        return overflow ? nil : result
    }

    static func optionalCheckedMultiply(_ left: Int, _ right: Int) -> Int? {
        let (result, overflow) = left.multipliedReportingOverflow(by: right)
        return overflow ? nil : result
    }

    static func readUInt32LE(_ bytes: Data, at offset: Int) -> UInt32? {
        guard offset >= 0,
              let end = optionalCheckedAdd(offset, 4),
              end <= bytes.count
        else {
            return nil
        }
        var value: UInt32 = 0
        for index in 0..<4 {
            value |= UInt32(bytes[offset + index]) << UInt32(index * 8)
        }
        return value
    }

    static func readUInt64LE(_ bytes: Data, at offset: Int) -> UInt64? {
        guard offset >= 0,
              let end = optionalCheckedAdd(offset, 8),
              end <= bytes.count
        else {
            return nil
        }
        var value: UInt64 = 0
        for index in 0..<8 {
            value |= UInt64(bytes[offset + index]) << UInt64(index * 8)
        }
        return value
    }

    static func validDylibIdentity(
        _ bytes: Data,
        range: Range<Int>
    ) -> Bool {
        guard range.count >= 32,
              let rawNameOffset = readUInt32LE(
                bytes,
                at: range.lowerBound + 8
              ),
              let nameOffset = Int(exactly: rawNameOffset),
              nameOffset >= 24,
              nameOffset < range.count,
              let nameStart = optionalCheckedAdd(
                range.lowerBound,
                nameOffset
              )
        else {
            return false
        }
        let fixedEnd = range.lowerBound + 24
        if bytes[fixedEnd..<nameStart].contains(where: { $0 != 0 }) {
            return false
        }
        guard let terminator = bytes[nameStart..<range.upperBound]
            .firstIndex(of: 0)
        else {
            return false
        }
        let length = terminator - nameStart
        guard (1...4_096).contains(length) else { return false }
        return !bytes[(terminator + 1)..<range.upperBound]
            .contains(where: { $0 != 0 })
    }

    static func sha256Hex(_ bytes: Data) -> String {
        SHA256.hash(data: bytes)
            .map { String(format: "%02x", $0) }
            .joined()
    }

    static func hex(_ bytes: Data) -> String {
        bytes.map { String(format: "%02x", $0) }.joined()
    }

    static func isLowercaseHex(_ value: String, count: Int) -> Bool {
        let bytes = Array(value.utf8)
        return bytes.count == count &&
            bytes.allSatisfy {
                (0x30...0x39).contains($0) ||
                    (0x61...0x66).contains($0)
            }
    }

    static func base64URL(_ bytes: Data) -> String {
        bytes.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }
}
