import CryptoKit
import Foundation

enum SyntheticSharedCacheRangeConsumerKind:
    Int,
    Equatable,
    Sendable
{
    case setHeader = 0
    case mappingTable = 1
    case imageTable = 2
    case imageNameWindow = 3
    case installName = 4
    case machOHeader = 5
    case loadCommands = 6
    case codeSignature = 7
}

/// Caller-supplied synthetic bounded-read facts only. This value has no path,
/// descriptor, reader, callback, live source, or operational authority.
struct SyntheticSharedCacheBoundedReadTranscript:
    Equatable,
    Sendable
{
    enum Purpose: Equatable, Sendable {
        case setHeaderDiscovery
        case planProbe(
            consumerKind: SyntheticSharedCacheRangeConsumerKind,
            consumerOrdinal: Int
        )
    }

    let purpose: Purpose
    let fileOrdinal: Int
    let rangeStart: UInt64
    let requestedByteCount: UInt64
    let beforeMetadata: SyntheticCaptureFileMetadata
    let afterMetadata: SyntheticCaptureFileMetadata
    let events: [SyntheticCaptureReadEvent]
}

enum SyntheticSharedCacheDiscoveryRangePlanFailure:
    Error,
    Equatable,
    Sendable
{
    enum MetadataPosition: Equatable, Sendable {
        case before
        case after
    }

    enum DiscoveryTranscriptReason: Equatable, Sendable {
        case count(expected: Int, actual: Int)
        case purpose(transcriptOrdinal: Int)
        case fileOrdinal(
            transcriptOrdinal: Int,
            expected: Int,
            actual: Int
        )
        case rangeStart(
            transcriptOrdinal: Int,
            expected: UInt64,
            actual: UInt64
        )
        case requestedByteCount(
            transcriptOrdinal: Int,
            expected: UInt64,
            actual: UInt64
        )
        case metadata(
            transcriptOrdinal: Int,
            position: MetadataPosition,
            field: SyntheticSmallArtifactCaptureFailure.MetadataField
        )
        case arithmeticOverflow
        case byteLimit(actual: UInt64, maximum: UInt64)
        case attemptLimit(actual: UInt64, maximum: UInt64)
        case offset(
            transcriptOrdinal: Int,
            eventOrdinal: Int,
            expected: UInt64,
            actual: UInt64
        )
        case fragmentSize(
            transcriptOrdinal: Int,
            eventOrdinal: Int,
            bytes: Int
        )
        case fragmentBeyondChunk(
            transcriptOrdinal: Int,
            eventOrdinal: Int,
            chunkEnd: UInt64
        )
        case unexpectedEndOfFile(
            transcriptOrdinal: Int,
            eventOrdinal: Int
        )
        case trailingEvent(
            transcriptOrdinal: Int,
            eventOrdinal: Int
        )
        case incomplete(
            transcriptOrdinal: Int,
            expected: UInt64,
            actual: UInt64
        )
        case eventCountDrift(transcriptOrdinal: Int)
        case payloadCountDrift(
            transcriptOrdinal: Int,
            eventOrdinal: Int
        )
    }

    enum DiscoveryContinuityReason: Equatable, Sendable {
        case bytes
        case derivedRange
    }

    enum AnchoredMemberSetReason: Equatable, Sendable {
        case reanchor(RuntimeClosureExpectationArtifactRole)
        case reanchorMismatch(RuntimeClosureExpectationArtifactRole)
        case role(
            expected: RuntimeClosureExpectationArtifactRole,
            actual: RuntimeClosureExpectationArtifactRole
        )
        case platform
        case cacheSet
        case cacheRecord(fileOrdinal: Int)
        case empty
        case limit(actual: Int, maximum: Int)
        case memberConflict
    }

    enum ImageTableReason: Equatable, Sendable {
        case empty
        case bounds
        case padding(row: Int)
        case nameWindow
        case selectedNameMissing(memberOrdinal: Int)
        case selectedNameDuplicate(memberOrdinal: Int)
        case selectedAddressAlias(memberOrdinal: Int)
    }

    enum InstallNameReason: Equatable, Sendable {
        case bounds(row: Int)
        case terminator(row: Int)
        case syntax(row: Int)
    }

    enum RangePlanReason: Equatable, Sendable {
        case arithmeticOverflow
        case mapping
        case metadata(
            transcriptOrdinal: Int,
            position: MetadataPosition,
            field: SyntheticSmallArtifactCaptureFailure.MetadataField
        )
        case probeCount(expected: Int, actual: Int)
        case invalidInputConsumer(
            transcriptOrdinal: Int,
            consumerKind: SyntheticSharedCacheRangeConsumerKind
        )
        case probePurpose(transcriptOrdinal: Int)
        case probeOrdinal(transcriptOrdinal: Int)
        case probeRange(transcriptOrdinal: Int)
        case probeOrder(transcriptOrdinal: Int)
        case rawByteLimit(actual: UInt64, maximum: UInt64)
        case chunkLimit(actual: UInt64, maximum: UInt64)
        case attemptLimit(actual: UInt64, maximum: UInt64)
        case requestMismatch(transcriptOrdinal: Int)
        case machOHeader(memberOrdinal: Int)
        case loadCommandFrame(memberOrdinal: Int, commandOrdinal: Int)
        case unknownLoadCommand(
            memberOrdinal: Int,
            commandOrdinal: Int,
            command: UInt32
        )
        case forbiddenLoadCommand(
            memberOrdinal: Int,
            commandOrdinal: Int,
            command: UInt32
        )
        case uuid(memberOrdinal: Int)
        case dylibIdentity(memberOrdinal: Int)
        case codeSignature(memberOrdinal: Int)
        case linkedit(memberOrdinal: Int)
        case expectedMember(memberOrdinal: Int)
        case signatureParse(memberOrdinal: Int)
        case overlapBytes
        case logicalRangeLimit(actual: Int, maximum: Int)
        case additionalByteLimit(actual: UInt64, maximum: UInt64)
        case retainedByteLimit(actual: UInt64, maximum: UInt64)
        case transientByteLimit(actual: UInt64, maximum: UInt64)
        case consumerPieces
        case rederivation
    }

    case predecessor(SyntheticSharedCacheSetPlanFailure)
    case discoveryTranscript(DiscoveryTranscriptReason)
    case readInterruptedLimit(
        transcriptOrdinal: Int,
        eventOrdinal: Int
    )
    case readFragmentLimit(
        transcriptOrdinal: Int,
        eventOrdinal: Int
    )
    case readError(
        transcriptOrdinal: Int,
        eventOrdinal: Int,
        code: Int32
    )
    case discoveryContinuity(
        fileOrdinal: Int,
        reason: DiscoveryContinuityReason
    )
    case anchoredMemberSet(AnchoredMemberSetReason)
    case imageTable(ImageTableReason)
    case installName(InstallNameReason)
    case rangePlan(RangePlanReason)
}

/// Immutable synthetic selected-byte comparison only. It contains no ID,
/// evidence object, expectation document, locator, descriptor, or conversion.
struct SyntheticSharedCacheDiscoveryRangePlanComparison:
    Equatable,
    Sendable
{
    struct DiscoveryStore: Equatable, Sendable {
        let fileOrdinal: Int
        let bytes: Data
    }

    struct LogicalSelectedRange: Equatable, Sendable {
        let ordinal: Int
        let fileOrdinal: Int
        let start: UInt64
        let length: UInt64
    }

    struct AdditionalPhysicalRange: Equatable, Sendable {
        let ordinal: Int
        let fileOrdinal: Int
        let start: UInt64
        let length: UInt64
        let bytes: Data
    }

    enum PhysicalPiece: Equatable, Sendable {
        case discovery(
            fileOrdinal: Int,
            relativeStart: UInt64,
            length: UInt64
        )
        case additional(
            rangeOrdinal: Int,
            relativeStart: UInt64,
            length: UInt64
        )
    }

    struct ConsumerBinding: Equatable, Sendable {
        let consumerKind: SyntheticSharedCacheRangeConsumerKind
        let consumerOrdinal: Int
        let fileOrdinal: Int
        let start: UInt64
        let length: UInt64
        let pieces: [PhysicalPiece]
    }

    struct ResourceCounts: Equatable, Sendable {
        let discoveryBytes: UInt64
        let rawProbeRequestedBytes: UInt64
        let probeChunkCount: UInt64
        let discoveryAttemptCount: UInt64
        let planProbeAttemptCount: UInt64
        let totalAttemptCount: UInt64
        let logicalRangeCount: Int
        let additionalBytes: UInt64
        let combinedRetainedBytes: UInt64
    }

    fileprivate enum ConstructionSeal: Equatable, Sendable {
        case verified
    }

    let sourceProfile: SyntheticSharedCacheSourceProfile
    let discoveryStores: [DiscoveryStore]
    let logicalSelectedRanges: [LogicalSelectedRange]
    let additionalPhysicalRanges: [AdditionalPhysicalRange]
    let consumerBindings: [ConsumerBinding]
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
        sourceProfile: SyntheticSharedCacheSourceProfile,
        discoveryStores: [DiscoveryStore],
        logicalSelectedRanges: [LogicalSelectedRange],
        additionalPhysicalRanges: [AdditionalPhysicalRange],
        consumerBindings: [ConsumerBinding],
        resourceCounts: ResourceCounts,
        seal: ConstructionSeal
    ) {
        self.sourceProfile = sourceProfile
        self.discoveryStores = discoveryStores
        self.logicalSelectedRanges = logicalSelectedRanges
        self.additionalPhysicalRanges = additionalPhysicalRanges
        self.consumerBindings = consumerBindings
        self.resourceCounts = resourceCounts
        self.constructionSeal = seal
    }
}

enum SyntheticSharedCacheDiscoveryRangePlanVerifier {
    static let maximumDiscoveryBytes: UInt64 = 1_048_576
    static let maximumLogicalChunkBytes: UInt64 = 1_048_576
    static let maximumPositiveFragmentsPerChunk = 16
    static let maximumInterruptedRetriesPerOffset = 8
    static let maximumDiscoveryAttempts: UInt64 = 9_216
    static let maximumPlanProbeCount = 4_096
    static let maximumRawProbeBytes: UInt64 = 268_435_456
    static let maximumProbeChunks: UInt64 = 4_351
    static let maximumProbeAttempts: UInt64 = 626_544
    static let maximumTotalAttempts: UInt64 = 635_760
    static let maximumMemberCount = 512
    static let maximumLogicalRangeCount = 4_096
    static let maximumAdditionalBytes: UInt64 = 268_435_456
    static let maximumCombinedRetainedBytes: UInt64 = 269_484_032
    static let maximumVerifierOwnedContentBytes: UInt64 = 537_919_488
    static let maximumLoadCommandCount: UInt32 = 256
    static let maximumLoadCommandBytes: UInt32 = 262_144
    static let maximumCodeSignatureBytes: UInt32 = 262_144

    static func compare(
        setPlan: SyntheticSharedCacheSetPlanComparison,
        gitExpectation: AnchoredRuntimeClosureExpectationDocument,
        selfGuardExpectation: AnchoredRuntimeClosureExpectationDocument,
        discoveryTranscripts: [SyntheticSharedCacheBoundedReadTranscript],
        planProbeTranscripts: [SyntheticSharedCacheBoundedReadTranscript]
    ) throws -> SyntheticSharedCacheDiscoveryRangePlanComparison {
        try rederive(setPlan)
        let files = setPlan.headerOrderFiles

        let discoveryEnvelope = try validateDiscoveryEnvelope(
            files: files,
            transcripts: discoveryTranscripts
        )
        var discoveryStores: [
            SyntheticSharedCacheDiscoveryRangePlanComparison.DiscoveryStore
        ] = []
        discoveryStores.reserveCapacity(files.count)
        for index in files.indices {
            let bytes = try replay(
                discoveryTranscripts[index],
                transcriptOrdinal: index
            )
            guard bytes == files[index].discoveryBytes else {
                throw SyntheticSharedCacheDiscoveryRangePlanFailure
                    .discoveryContinuity(
                        fileOrdinal: index,
                        reason: .bytes
                    )
            }
            discoveryStores.append(.init(fileOrdinal: index, bytes: bytes))
        }

        let members = try anchoredMembers(
            git: gitExpectation,
            selfGuard: selfGuardExpectation,
            files: files
        )
        let mappings = try normalizedMappings(files)
        let imageRows = try imageRows(
            main: files[0],
            discovery: discoveryStores[0].bytes,
            membersAreNonempty: !members.isEmpty
        )
        let nameWindow = try nameWindow(
            rows: imageRows,
            mainFileBytes: files[0].fileBytes
        )

        let probeEnvelope = try validateProbeEnvelope(
            planProbeTranscripts,
            files: files,
            memberCount: members.count,
            minimumExpectedCount: 1 + members.count * 2,
            discoveryAttemptCount: discoveryEnvelope.attemptCount
        )

        var rawBindings = try discoveryBindings(
            files: files,
            stores: discoveryStores
        )
        var probeIndex = 0

        let selectedImages: [SelectedImage]
        do {
            let nameProbe = try consumeProbe(
                planProbeTranscripts,
                index: &probeIndex,
                expected: ProbeKey(
                    kind: .imageNameWindow,
                    consumerOrdinal: 0,
                    fileOrdinal: 0,
                    start: nameWindow.lowerBound,
                    length: nameWindow.upperBound - nameWindow.lowerBound
                ),
                globalBase: files.count
            )
            rawBindings.append(nameProbe.binding)

            selectedImages = try Self.selectedImages(
                members: members,
                rows: imageRows,
                window: nameWindow,
                bytes: nameProbe.bytes
            )
            for selected in selectedImages {
                rawBindings.append(
                    RawBinding(
                        kind: .installName,
                        consumerOrdinal: selected.member.ordinal,
                        fileOrdinal: 0,
                        start: selected.pathStart,
                        length: UInt64(selected.member.decodedName.count),
                        byteSource: .owned(
                            nameProbe.bytes,
                            base: nameWindow.lowerBound
                        )
                    )
                )
            }
        }

        var headerRequests: [(selected: SelectedImage, key: ProbeKey)] = []
        headerRequests.reserveCapacity(selectedImages.count)
        for selected in selectedImages {
            let translated = try translate(
                vmStart: selected.row.address,
                length: 32,
                mappings: mappings
            )
            headerRequests.append((
                selected: selected,
                key: ProbeKey(
                    kind: .machOHeader,
                    consumerOrdinal: selected.member.ordinal,
                    fileOrdinal: translated.fileOrdinal,
                    start: translated.fileStart,
                    length: 32
                )
            ))
        }
        headerRequests.sort { $0.key.precedes($1.key) }

        var parsedHeaders: [Int: MachHeaderFacts] = [:]
        parsedHeaders.reserveCapacity(selectedImages.count)
        for request in headerRequests {
            let selected = request.selected
            let probe = try consumeProbe(
                planProbeTranscripts,
                index: &probeIndex,
                expected: request.key,
                globalBase: files.count
            )
            let header = try parseMachHeader(
                probe.bytes,
                memberOrdinal: selected.member.ordinal
            )
            _ = try translate(
                vmStart: try checkedAdd(selected.row.address, 32),
                length: UInt64(header.sizeOfCommands),
                mappings: mappings
            )
            parsedHeaders[selected.member.ordinal] = header
            rawBindings.append(probe.binding)
        }

        var loadRequests: [(
            selected: SelectedImage,
            header: MachHeaderFacts,
            key: ProbeKey
        )] = []
        loadRequests.reserveCapacity(selectedImages.count)
        for selected in selectedImages {
            guard let header = parsedHeaders[selected.member.ordinal] else {
                throw Failure.rangePlan(.rederivation)
            }
            let commandVMStart = try checkedAdd(selected.row.address, 32)
            let translated = try translate(
                vmStart: commandVMStart,
                length: UInt64(header.sizeOfCommands),
                mappings: mappings
            )
            loadRequests.append((
                selected: selected,
                header: header,
                key: ProbeKey(
                    kind: .loadCommands,
                    consumerOrdinal: selected.member.ordinal,
                    fileOrdinal: translated.fileOrdinal,
                    start: translated.fileStart,
                    length: UInt64(header.sizeOfCommands)
                )
            ))
        }
        loadRequests.sort { $0.key.precedes($1.key) }

        var loadFacts: [Int: LoadCommandFacts] = [:]
        loadFacts.reserveCapacity(selectedImages.count)
        for request in loadRequests {
            let selected = request.selected
            let probe = try consumeProbe(
                planProbeTranscripts,
                index: &probeIndex,
                expected: request.key,
                globalBase: files.count
            )
            let facts = try parseLoadCommands(
                probe.bytes,
                header: request.header,
                selected: selected,
                mappings: mappings
            )
            loadFacts[selected.member.ordinal] = facts
            rawBindings.append(probe.binding)
        }

        var signatureRequests: [(
            selected: SelectedImage,
            key: ProbeKey
        )] = []
        for selected in selectedImages {
            guard let signature = loadFacts[selected.member.ordinal]?.signature
            else {
                continue
            }
            signatureRequests.append((
                selected: selected,
                key: ProbeKey(
                    kind: .codeSignature,
                    consumerOrdinal: selected.member.ordinal,
                    fileOrdinal: signature.fileOrdinal,
                    start: signature.fileStart,
                    length: signature.length
                )
            ))
        }
        signatureRequests.sort { $0.key.precedes($1.key) }

        for request in signatureRequests {
            let selected = request.selected
            let probe = try consumeProbe(
                planProbeTranscripts,
                index: &probeIndex,
                expected: request.key,
                globalBase: files.count
            )
            try validateSignatureProbe(probe.bytes, selected: selected)
            rawBindings.append(probe.binding)
        }

        guard probeIndex == planProbeTranscripts.count else {
            throw SyntheticSharedCacheDiscoveryRangePlanFailure
                .rangePlan(.probeCount(
                    expected: probeIndex,
                    actual: planProbeTranscripts.count
                ))
        }

        var normalized = try normalize(
            rawBindings,
            discoveryStores: discoveryStores
        )
        rawBindings.removeAll(keepingCapacity: false)
        let resourceCounts = try resourceCounts(
            discovery: discoveryEnvelope,
            probes: probeEnvelope,
            logicalRangeCount: normalized.logicalRanges.count,
            additionalBytes: normalized.additionalBytes
        )
        try validateFinalStores(
            rawBindings: normalized.canonicalBindings,
            discoveryStores: discoveryStores,
            additional: normalized.additionalRanges,
            bindings: normalized.consumerBindings
        )
        normalized.canonicalBindings.removeAll(keepingCapacity: false)
        try rederiveFinal(
            discoveryStores: discoveryStores,
            additional: normalized.additionalRanges,
            bindings: normalized.consumerBindings,
            files: files,
            members: members,
            mappings: mappings
        )

        return SyntheticSharedCacheDiscoveryRangePlanComparison(
            sourceProfile: setPlan.sourceProfile,
            discoveryStores: discoveryStores,
            logicalSelectedRanges: normalized.logicalRanges,
            additionalPhysicalRanges: normalized.additionalRanges,
            consumerBindings: normalized.consumerBindings,
            resourceCounts: resourceCounts,
            seal: .verified
        )
    }
}

private extension SyntheticSharedCacheDiscoveryRangePlanVerifier {
    typealias Failure = SyntheticSharedCacheDiscoveryRangePlanFailure
    typealias Comparison = SyntheticSharedCacheDiscoveryRangePlanComparison

    struct EnvelopeCounts {
        let bytes: UInt64
        let chunks: UInt64
        let attemptCount: UInt64
    }

    struct SharedMember {
        let ordinal: Int
        let decodedName: Data
        let machOUUID: String
        let primaryCodeDirectoryBlobSHA256: String
        let loadCommandsSHA256: String
        let contentEvidenceID: String
    }

    struct Mapping {
        let fileOrdinal: Int
        let vmStart: UInt64
        let vmEnd: UInt64
        let fileStart: UInt64
        let fileEnd: UInt64
    }

    struct Translation {
        let fileOrdinal: Int
        let fileStart: UInt64
    }

    struct ImageRow {
        let row: Int
        let address: UInt64
        let pathFileOffset: UInt64
    }

    struct SelectedImage {
        let member: SharedMember
        let row: ImageRow
        let pathStart: UInt64
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

    struct LoadCommandFacts {
        let signature: SignatureRequest?
    }

    struct ProbeKey: Equatable {
        let kind: SyntheticSharedCacheRangeConsumerKind
        let consumerOrdinal: Int
        let fileOrdinal: Int
        let start: UInt64
        let length: UInt64

        var end: UInt64 { start + length }
        var phase: Int {
            switch kind {
            case .imageNameWindow: 0
            case .machOHeader: 1
            case .loadCommands: 2
            case .codeSignature: 3
            default: -1
            }
        }

        func precedes(_ other: ProbeKey) -> Bool {
            if phase != other.phase { return phase < other.phase }
            if fileOrdinal != other.fileOrdinal {
                return fileOrdinal < other.fileOrdinal
            }
            if start != other.start { return start < other.start }
            if end != other.end { return end < other.end }
            if kind.rawValue != other.kind.rawValue {
                return kind.rawValue < other.kind.rawValue
            }
            return consumerOrdinal < other.consumerOrdinal
        }
    }

    struct OwnedProbe {
        let key: ProbeKey
        let bytes: Data

        var binding: RawBinding {
            RawBinding(
                kind: key.kind,
                consumerOrdinal: key.consumerOrdinal,
                fileOrdinal: key.fileOrdinal,
                start: key.start,
                length: key.length,
                byteSource: .owned(bytes, base: key.start)
            )
        }
    }

    struct RawBinding {
        enum ByteSource {
            case discovery
            case owned(Data, base: UInt64)
        }

        let kind: SyntheticSharedCacheRangeConsumerKind
        let consumerOrdinal: Int
        let fileOrdinal: Int
        let start: UInt64
        let length: UInt64
        let byteSource: ByteSource

        var end: UInt64 { start + length }
    }

    struct CoalescedInterval {
        let fileOrdinal: Int
        let start: UInt64
        let end: UInt64
    }

    struct NormalizedResult {
        let logicalRanges: [Comparison.LogicalSelectedRange]
        let additionalRanges: [Comparison.AdditionalPhysicalRange]
        let consumerBindings: [Comparison.ConsumerBinding]
        var canonicalBindings: [RawBinding]
        let additionalBytes: UInt64
    }

    struct LinkeditFacts {
        let vmAddress: UInt64
        let vmSize: UInt64
        let fileOffset: UInt64
        let fileSize: UInt64
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

    static func rederive(_ setPlan: SyntheticSharedCacheSetPlanComparison)
        throws
    {
        do {
            let main = setPlan.headerOrderFiles[0]
            let rederived = try SyntheticSharedCacheSetPlanVerifier.compare(
                sourceProfile: setPlan.sourceProfile,
                main: SyntheticSharedCacheDiscoveryFile(
                    metadata: main.metadata,
                    discoveryBytes: Data(main.discoveryBytes)
                ),
                subcaches: setPlan.headerOrderFiles.dropFirst().map {
                    SyntheticSharedCacheDiscoveryFile(
                        metadata: $0.metadata,
                        discoveryBytes: Data($0.discoveryBytes)
                    )
                }
            )
            guard rederived == setPlan else {
                throw Failure.predecessor(.sourceProfile)
            }
        } catch let failure as SyntheticSharedCacheSetPlanFailure {
            throw Failure.predecessor(failure)
        }
    }

    static func validateDiscoveryEnvelope(
        files: [SyntheticSharedCacheSetPlanComparison.HeaderOrderFile],
        transcripts: [SyntheticSharedCacheBoundedReadTranscript]
    ) throws -> EnvelopeCounts {
        guard transcripts.count == files.count else {
            throw Failure.discoveryTranscript(.count(
                expected: files.count,
                actual: transcripts.count
            ))
        }
        var totalBytes: UInt64 = 0
        var totalAttempts: UInt64 = 0
        for index in files.indices {
            let transcript = transcripts[index]
            guard transcript.purpose == .setHeaderDiscovery else {
                throw Failure.discoveryTranscript(.purpose(
                    transcriptOrdinal: index
                ))
            }
            guard transcript.fileOrdinal == index else {
                throw Failure.discoveryTranscript(.fileOrdinal(
                    transcriptOrdinal: index,
                    expected: index,
                    actual: transcript.fileOrdinal
                ))
            }
            guard transcript.rangeStart == 0 else {
                throw Failure.discoveryTranscript(.rangeStart(
                    transcriptOrdinal: index,
                    expected: 0,
                    actual: transcript.rangeStart
                ))
            }
            let expected = UInt64(files[index].discoveryBytes.count)
            guard transcript.requestedByteCount == expected else {
                throw Failure.discoveryTranscript(.requestedByteCount(
                    transcriptOrdinal: index,
                    expected: expected,
                    actual: transcript.requestedByteCount
                ))
            }
            try validateMetadata(
                transcript,
                expected: files[index].metadata,
                transcriptOrdinal: index
            )
            totalBytes = try checkedAdd(totalBytes, expected)
            totalAttempts = try checkedAdd(
                totalAttempts,
                UInt64(transcript.events.count)
            )
        }
        guard totalBytes <= maximumDiscoveryBytes else {
            throw Failure.discoveryTranscript(.byteLimit(
                actual: totalBytes,
                maximum: maximumDiscoveryBytes
            ))
        }
        guard totalAttempts <= maximumDiscoveryAttempts else {
            throw Failure.discoveryTranscript(.attemptLimit(
                actual: totalAttempts,
                maximum: maximumDiscoveryAttempts
            ))
        }
        return EnvelopeCounts(
            bytes: totalBytes,
            chunks: UInt64(files.count),
            attemptCount: totalAttempts
        )
    }

    static func validateProbeEnvelope(
        _ transcripts: [SyntheticSharedCacheBoundedReadTranscript],
        files: [SyntheticSharedCacheSetPlanComparison.HeaderOrderFile],
        memberCount: Int,
        minimumExpectedCount: Int,
        discoveryAttemptCount: UInt64
    ) throws -> EnvelopeCounts {
        guard transcripts.count >= minimumExpectedCount else {
            throw Failure.rangePlan(.probeCount(
                expected: minimumExpectedCount,
                actual: transcripts.count
            ))
        }
        guard transcripts.count <= maximumPlanProbeCount else {
            throw Failure.rangePlan(.probeCount(
                expected: maximumPlanProbeCount,
                actual: transcripts.count
            ))
        }
        var totalBytes: UInt64 = 0
        var totalChunks: UInt64 = 0
        var totalAttempts: UInt64 = 0
        var previous: ProbeKey?
        for index in transcripts.indices {
            let transcript = transcripts[index]
            guard case let .planProbe(kind, ordinal) = transcript.purpose else {
                throw Failure.rangePlan(.probePurpose(
                    transcriptOrdinal: index
                ))
            }
            guard [.imageNameWindow, .machOHeader, .loadCommands,
                   .codeSignature].contains(kind)
            else {
                throw Failure.rangePlan(.invalidInputConsumer(
                    transcriptOrdinal: index,
                    consumerKind: kind
                ))
            }
            guard transcript.fileOrdinal >= 0,
                  transcript.fileOrdinal < files.count,
                  ordinal >= 0,
                  ((kind == .imageNameWindow && ordinal == 0) ||
                   (kind != .imageNameWindow && ordinal < memberCount))
            else {
                throw Failure.rangePlan(.probeOrdinal(
                    transcriptOrdinal: index
                ))
            }
            guard transcript.requestedByteCount > 0,
                  let fileBytes = UInt64(exactly:
                    files[transcript.fileOrdinal].metadata.size),
                  let end = optionalCheckedAdd(
                    transcript.rangeStart,
                    transcript.requestedByteCount
                  ),
                  end <= fileBytes
            else {
                throw Failure.rangePlan(.probeRange(
                    transcriptOrdinal: index
                ))
            }
            let expectedMetadata = files[transcript.fileOrdinal].metadata
            if let field = firstMetadataDrift(
                expectedMetadata,
                transcript.beforeMetadata
            ) {
                throw Failure.rangePlan(.metadata(
                    transcriptOrdinal: index,
                    position: .before,
                    field: field
                ))
            }
            if let field = firstMetadataDrift(
                expectedMetadata,
                transcript.afterMetadata
            ) {
                throw Failure.rangePlan(.metadata(
                    transcriptOrdinal: index,
                    position: .after,
                    field: field
                ))
            }
            let key = ProbeKey(
                kind: kind,
                consumerOrdinal: ordinal,
                fileOrdinal: transcript.fileOrdinal,
                start: transcript.rangeStart,
                length: transcript.requestedByteCount
            )
            if let previous, !previous.precedes(key) {
                throw Failure.rangePlan(.probeOrder(
                    transcriptOrdinal: index
                ))
            }
            previous = key
            totalBytes = try checkedAdd(
                totalBytes,
                transcript.requestedByteCount
            )
            totalChunks = try checkedAdd(
                totalChunks,
                chunkCount(transcript.requestedByteCount)
            )
            totalAttempts = try checkedAdd(
                totalAttempts,
                UInt64(transcript.events.count)
            )
        }
        guard totalBytes <= maximumRawProbeBytes else {
            throw Failure.rangePlan(.rawByteLimit(
                actual: totalBytes,
                maximum: maximumRawProbeBytes
            ))
        }
        guard totalChunks <= maximumProbeChunks else {
            throw Failure.rangePlan(.chunkLimit(
                actual: totalChunks,
                maximum: maximumProbeChunks
            ))
        }
        guard totalAttempts <= maximumProbeAttempts else {
            throw Failure.rangePlan(.attemptLimit(
                actual: totalAttempts,
                maximum: maximumProbeAttempts
            ))
        }
        let allAttempts = try checkedAdd(
            discoveryAttemptCount,
            totalAttempts
        )
        guard allAttempts <= maximumTotalAttempts else {
            throw Failure.rangePlan(.attemptLimit(
                actual: allAttempts,
                maximum: maximumTotalAttempts
            ))
        }
        return EnvelopeCounts(
            bytes: totalBytes,
            chunks: totalChunks,
            attemptCount: totalAttempts
        )
    }

    static func validateMetadata(
        _ transcript: SyntheticSharedCacheBoundedReadTranscript,
        expected: SyntheticCaptureFileMetadata,
        transcriptOrdinal: Int
    ) throws {
        if let field = firstMetadataDrift(
            expected,
            transcript.beforeMetadata
        ) {
            throw Failure.discoveryTranscript(.metadata(
                transcriptOrdinal: transcriptOrdinal,
                position: .before,
                field: field
            ))
        }
        if let field = firstMetadataDrift(
            expected,
            transcript.afterMetadata
        ) {
            throw Failure.discoveryTranscript(.metadata(
                transcriptOrdinal: transcriptOrdinal,
                position: .after,
                field: field
            ))
        }
    }

    static func replay(
        _ transcript: SyntheticSharedCacheBoundedReadTranscript,
        transcriptOrdinal: Int
    ) throws -> Data {
        guard let capacity = Int(exactly: transcript.requestedByteCount),
              transcript.requestedByteCount > 0
        else {
            throw Failure.discoveryTranscript(.arithmeticOverflow)
        }
        let eventCount = transcript.events.count
        var destination = Data(count: capacity)
        var expectedOffset = transcript.rangeStart
        let end = try checkedAdd(
            transcript.rangeStart,
            transcript.requestedByteCount
        )
        var chunkEnd = min(
            try checkedAdd(transcript.rangeStart, maximumLogicalChunkBytes),
            end
        )
        var fragments = 0
        var interruptions = 0

        for index in 0..<eventCount {
            guard transcript.events.count == eventCount else {
                throw Failure.discoveryTranscript(.eventCountDrift(
                    transcriptOrdinal: transcriptOrdinal
                ))
            }
            if expectedOffset == end {
                throw Failure.discoveryTranscript(.trailingEvent(
                    transcriptOrdinal: transcriptOrdinal,
                    eventOrdinal: index
                ))
            }
            let event = transcript.events[index]
            guard event.offset == expectedOffset else {
                throw Failure.discoveryTranscript(.offset(
                    transcriptOrdinal: transcriptOrdinal,
                    eventOrdinal: index,
                    expected: expectedOffset,
                    actual: event.offset
                ))
            }
            switch event {
            case .interrupted:
                interruptions += 1
                guard interruptions <= maximumInterruptedRetriesPerOffset else {
                    throw Failure.readInterruptedLimit(
                        transcriptOrdinal: transcriptOrdinal,
                        eventOrdinal: index
                    )
                }

            case let .error(_, code):
                throw Failure.readError(
                    transcriptOrdinal: transcriptOrdinal,
                    eventOrdinal: index,
                    code: code
                )

            case .endOfFile:
                throw Failure.discoveryTranscript(.unexpectedEndOfFile(
                    transcriptOrdinal: transcriptOrdinal,
                    eventOrdinal: index
                ))

            case let .bytes(_, data):
                let payloadCount = data.count
                guard payloadCount > 0,
                      payloadCount <= Int(maximumLogicalChunkBytes)
                else {
                    throw Failure.discoveryTranscript(.fragmentSize(
                        transcriptOrdinal: transcriptOrdinal,
                        eventOrdinal: index,
                        bytes: payloadCount
                    ))
                }
                fragments += 1
                guard fragments <= maximumPositiveFragmentsPerChunk else {
                    throw Failure.readFragmentLimit(
                        transcriptOrdinal: transcriptOrdinal,
                        eventOrdinal: index
                    )
                }
                guard let payloadBytes = UInt64(exactly: payloadCount),
                      let next = optionalCheckedAdd(
                        expectedOffset,
                        payloadBytes
                      ),
                      next <= end
                else {
                    throw Failure.discoveryTranscript(.fragmentSize(
                        transcriptOrdinal: transcriptOrdinal,
                        eventOrdinal: index,
                        bytes: payloadCount
                    ))
                }
                guard next <= chunkEnd else {
                    throw Failure.discoveryTranscript(.fragmentBeyondChunk(
                        transcriptOrdinal: transcriptOrdinal,
                        eventOrdinal: index,
                        chunkEnd: chunkEnd
                    ))
                }
                let relative = expectedOffset - transcript.rangeStart
                guard let lower = Int(exactly: relative),
                      next >= transcript.rangeStart,
                      let upper = Int(exactly:
                        next - transcript.rangeStart
                      ),
                      upper <= destination.count
                else {
                    throw Failure.discoveryTranscript(.arithmeticOverflow)
                }
                destination.replaceSubrange(lower..<upper, with: data)
                guard transcript.events.count == eventCount else {
                    throw Failure.discoveryTranscript(.eventCountDrift(
                        transcriptOrdinal: transcriptOrdinal
                    ))
                }
                guard data.count == payloadCount else {
                    throw Failure.discoveryTranscript(.payloadCountDrift(
                        transcriptOrdinal: transcriptOrdinal,
                        eventOrdinal: index
                    ))
                }
                expectedOffset = next
                interruptions = 0
                if expectedOffset == chunkEnd, expectedOffset < end {
                    chunkEnd = min(
                        try checkedAdd(
                            expectedOffset,
                            maximumLogicalChunkBytes
                        ),
                        end
                    )
                    fragments = 0
                }
            }
        }
        guard expectedOffset == end else {
            throw Failure.discoveryTranscript(.incomplete(
                transcriptOrdinal: transcriptOrdinal,
                expected: end,
                actual: expectedOffset
            ))
        }
        return destination
    }

    static func anchoredMembers(
        git: AnchoredRuntimeClosureExpectationDocument,
        selfGuard: AnchoredRuntimeClosureExpectationDocument,
        files: [SyntheticSharedCacheSetPlanComparison.HeaderOrderFile]
    ) throws -> [SharedMember] {
        let currentGit = try reanchor(git, expected: .git)
        let currentSelf = try reanchor(selfGuard, expected: .selfGuard)
        guard platformTuple(currentGit.fields) ==
                platformTuple(currentSelf.fields)
        else {
            throw Failure.anchoredMemberSet(.platform)
        }
        guard currentGit.sharedCacheSetEvidence ==
                currentSelf.sharedCacheSetEvidence
        else {
            throw Failure.anchoredMemberSet(.cacheSet)
        }
        let cache = currentGit.sharedCacheSetEvidence
        guard cache.records.count == files.count else {
            throw Failure.anchoredMemberSet(.cacheSet)
        }
        for file in files {
            let matches = cache.decodedSuffixes.indices.filter {
                cache.decodedSuffixes[$0] == file.decodedSuffix
            }
            guard matches.count == 1 else {
                throw Failure.anchoredMemberSet(.cacheRecord(
                    fileOrdinal: file.ordinal
                ))
            }
            let record = cache.records[matches[0]]
            guard record.suffixBytes == file.suffixByteCount,
                  record.suffixBase64URL == file.suffixBase64URL,
                  record.fileBytes == file.fileBytes,
                  record.headerUUID == file.headerUUID
            else {
                throw Failure.anchoredMemberSet(.cacheRecord(
                    fileOrdinal: file.ordinal
                ))
            }
        }

        var union: [Data: RuntimeClosureExpectationMemberFields] = [:]
        for fields in [currentGit.fields, currentSelf.fields] {
            for member in fields.members where member.storage == .sharedCache {
                if let existing = union[member.decodedInstallName] {
                    guard existing == member else {
                        throw Failure.anchoredMemberSet(.memberConflict)
                    }
                } else {
                    union[Data(member.decodedInstallName)] = member
                }
            }
        }
        guard !union.isEmpty else {
            throw Failure.anchoredMemberSet(.empty)
        }
        guard union.count <= maximumMemberCount else {
            throw Failure.anchoredMemberSet(.limit(
                actual: union.count,
                maximum: maximumMemberCount
            ))
        }
        let ordered = union.values.sorted {
            $0.decodedInstallName.lexicographicallyPrecedes(
                $1.decodedInstallName
            )
        }
        return ordered.enumerated().map { ordinal, member in
            SharedMember(
                ordinal: ordinal,
                decodedName: Data(member.decodedInstallName),
                machOUUID: member.machOUUID,
                primaryCodeDirectoryBlobSHA256:
                    member.primaryCodeDirectoryBlobSHA256,
                loadCommandsSHA256: member.loadCommandsSHA256,
                contentEvidenceID: member.contentEvidenceID
            )
        }
    }

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
            throw Failure.anchoredMemberSet(.reanchor(role))
        }
        guard current == input else {
            throw Failure.anchoredMemberSet(.reanchorMismatch(role))
        }
        guard current.fields.artifactRole == role else {
            throw Failure.anchoredMemberSet(.role(
                expected: role,
                actual: current.fields.artifactRole
            ))
        }
        return current
    }

    static func platformTuple(
        _ fields: RuntimeClosureExpectationFields
    ) -> [String] {
        [
            fields.platformArchitecture,
            fields.platformHardwareModel,
            fields.platformOSVersion,
            fields.platformOSBuild,
            fields.resolutionProfile,
            fields.environmentProfile,
        ]
    }

    static func normalizedMappings(
        _ files: [SyntheticSharedCacheSetPlanComparison.HeaderOrderFile]
    ) throws -> [Mapping] {
        var result: [Mapping] = []
        for file in files {
            for fact in file.header.mappings {
                guard let vmEnd = optionalCheckedAdd(fact.address, fact.size),
                      let fileEnd = optionalCheckedAdd(
                        fact.fileOffset,
                        fact.size
                      ),
                      fileEnd <= file.fileBytes
                else {
                    throw Failure.rangePlan(.mapping)
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
                throw Failure.rangePlan(.mapping)
            }
        }
        return result
    }

    static func translate(
        vmStart: UInt64,
        length: UInt64,
        mappings: [Mapping]
    ) throws -> Translation {
        guard length > 0,
              let vmEnd = optionalCheckedAdd(vmStart, length)
        else {
            throw Failure.rangePlan(.mapping)
        }
        let owners = mappings.filter {
            $0.vmStart <= vmStart && vmEnd <= $0.vmEnd
        }
        guard owners.count == 1 else {
            throw Failure.rangePlan(.mapping)
        }
        let owner = owners[0]
        let delta = vmStart - owner.vmStart
        guard let fileStart = optionalCheckedAdd(owner.fileStart, delta),
              let fileEnd = optionalCheckedAdd(fileStart, length),
              fileEnd <= owner.fileEnd
        else {
            throw Failure.rangePlan(.mapping)
        }
        return Translation(
            fileOrdinal: owner.fileOrdinal,
            fileStart: fileStart
        )
    }

    static func imageRows(
        main: SyntheticSharedCacheSetPlanComparison.HeaderOrderFile,
        discovery: Data,
        membersAreNonempty: Bool
    ) throws -> [ImageRow] {
        let count = Int(main.header.imagesCount)
        guard !membersAreNonempty || count > 0 else {
            throw Failure.imageTable(.empty)
        }
        guard let start = Int(exactly: main.header.imagesOffset),
              let byteCount = optionalCheckedMultiply(count, 32),
              let end = optionalCheckedAdd(start, byteCount),
              end <= discovery.count
        else {
            throw Failure.imageTable(.bounds)
        }
        var rows: [ImageRow] = []
        rows.reserveCapacity(count)
        for row in 0..<count {
            guard let rowBytes = optionalCheckedMultiply(row, 32),
                  let offset = optionalCheckedAdd(start, rowBytes),
                  let pathOffset = optionalCheckedAdd(offset, 24),
                  let paddingOffset = optionalCheckedAdd(offset, 28)
            else {
                throw Failure.imageTable(.bounds)
            }
            guard let address = readUInt64LE(discovery, at: offset),
                  let path = readUInt32LE(discovery, at: pathOffset),
                  readUInt32LE(discovery, at: paddingOffset) == 0
            else {
                throw Failure.imageTable(.padding(row: row))
            }
            rows.append(ImageRow(
                row: row,
                address: address,
                pathFileOffset: UInt64(path)
            ))
        }
        return rows
    }

    static func nameWindow(
        rows: [ImageRow],
        mainFileBytes: UInt64
    ) throws -> Range<UInt64> {
        guard let start = rows.map(\.pathFileOffset).min() else {
            throw Failure.imageTable(.empty)
        }
        var candidateEnd: UInt64 = 0
        for row in rows {
            guard row.pathFileOffset < mainFileBytes,
                  let end = optionalCheckedAdd(row.pathFileOffset, 4_097)
            else {
                throw Failure.imageTable(.nameWindow)
            }
            candidateEnd = max(candidateEnd, end)
        }
        let end = min(candidateEnd, mainFileBytes)
        guard start < end else {
            throw Failure.imageTable(.nameWindow)
        }
        return start..<end
    }

    static func selectedImages(
        members: [SharedMember],
        rows: [ImageRow],
        window: Range<UInt64>,
        bytes: Data
    ) throws -> [SelectedImage] {
        var parsedNames: [(row: ImageRow, range: Range<Int>)] = []
        parsedNames.reserveCapacity(rows.count)
        for row in rows {
            guard row.pathFileOffset >= window.lowerBound,
                  row.pathFileOffset < window.upperBound
            else {
                throw Failure.installName(.bounds(row: row.row))
            }
            let relative = row.pathFileOffset - window.lowerBound
            guard let lower = Int(exactly: relative), lower < bytes.count else {
                throw Failure.installName(.bounds(row: row.row))
            }
            guard let maximumEnd = optionalCheckedAdd(lower, 4_097) else {
                throw Failure.installName(.bounds(row: row.row))
            }
            let allowed = min(bytes.count, maximumEnd)
            guard let terminator = bytes[lower..<allowed].firstIndex(of: 0)
            else {
                throw Failure.installName(.terminator(row: row.row))
            }
            let range = lower..<terminator
            guard isCanonicalInstallName(bytes, range: range) else {
                throw Failure.installName(.syntax(row: row.row))
            }
            parsedNames.append((
                row: row,
                range: range
            ))
        }

        var usedAddresses = Set<UInt64>()
        var result: [SelectedImage] = []
        result.reserveCapacity(members.count)
        for member in members {
            let matches = parsedNames.filter {
                $0.range.count == member.decodedName.count &&
                    bytes[$0.range].elementsEqual(member.decodedName)
            }
            guard !matches.isEmpty else {
                throw Failure.imageTable(.selectedNameMissing(
                    memberOrdinal: member.ordinal
                ))
            }
            guard matches.count == 1 else {
                throw Failure.imageTable(.selectedNameDuplicate(
                    memberOrdinal: member.ordinal
                ))
            }
            let match = matches[0]
            guard usedAddresses.insert(match.row.address).inserted else {
                throw Failure.imageTable(.selectedAddressAlias(
                    memberOrdinal: member.ordinal
                ))
            }
            result.append(SelectedImage(
                member: member,
                row: match.row,
                pathStart: match.row.pathFileOffset
            ))
        }
        return result
    }

    static func consumeProbe(
        _ transcripts: [SyntheticSharedCacheBoundedReadTranscript],
        index: inout Int,
        expected: ProbeKey,
        globalBase: Int
    ) throws -> OwnedProbe {
        guard index < transcripts.count else {
            throw Failure.rangePlan(.probeCount(
                expected: index + 1,
                actual: transcripts.count
            ))
        }
        let transcript = transcripts[index]
        guard case let .planProbe(kind, ordinal) = transcript.purpose,
              kind == expected.kind,
              ordinal == expected.consumerOrdinal,
              transcript.fileOrdinal == expected.fileOrdinal,
              transcript.rangeStart == expected.start,
              transcript.requestedByteCount == expected.length
        else {
            throw Failure.rangePlan(.requestMismatch(
                transcriptOrdinal: index
            ))
        }
        let bytes = try replay(
            transcript,
            transcriptOrdinal: globalBase + index
        )
        index += 1
        return OwnedProbe(key: expected, bytes: bytes)
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
              commandCount <= maximumLoadCommandCount,
              let commandBytes = readUInt32LE(bytes, at: 20),
              commandBytes > 0,
              commandBytes <= maximumLoadCommandBytes,
              readUInt32LE(bytes, at: 28) == 0
        else {
            throw Failure.rangePlan(.machOHeader(
                memberOrdinal: memberOrdinal
            ))
        }
        return MachHeaderFacts(
            commandCount: commandCount,
            sizeOfCommands: commandBytes
        )
    }

    static func validateSignatureProbe(
        _ bytes: Data,
        selected: SelectedImage
    ) throws {
        do {
            let parsed = try SyntheticMachOIdentityParser
                .parseEmbeddedSignatureForFileTypeIdentity(bytes)
            guard let primary = parsed.codeDirectories.first,
                  primary.blobSHA256 ==
                    selected.member.primaryCodeDirectoryBlobSHA256
            else {
                throw Failure.rangePlan(.expectedMember(
                    memberOrdinal: selected.member.ordinal
                ))
            }
        } catch let failure as Failure {
            throw failure
        } catch {
            throw Failure.rangePlan(.signatureParse(
                memberOrdinal: selected.member.ordinal
            ))
        }
    }

    static func parseLoadCommands(
        _ bytes: Data,
        header: MachHeaderFacts,
        selected: SelectedImage,
        mappings: [Mapping]
    ) throws -> LoadCommandFacts {
        guard bytes.count == Int(header.sizeOfCommands) else {
            throw Failure.rangePlan(.machOHeader(
                memberOrdinal: selected.member.ordinal
            ))
        }
        var cursor = 0
        var uuid: Data?
        var sawIdentity = false
        var signature: (offset: UInt64, size: UInt64)?
        var linkedits: [LinkeditFacts] = []
        for ordinal in 0..<Int(header.commandCount) {
            guard let frameHeaderEnd = optionalCheckedAdd(cursor, 8),
                  frameHeaderEnd <= bytes.count,
                  let command = readUInt32LE(bytes, at: cursor),
                  let sizeOffset = optionalCheckedAdd(cursor, 4),
                  let rawSize = readUInt32LE(bytes, at: sizeOffset),
                  let size = Int(exactly: rawSize),
                  size >= 8,
                  size.isMultiple(of: 8),
                  let frameEnd = optionalCheckedAdd(cursor, size),
                  frameEnd <= bytes.count
            else {
                throw Failure.rangePlan(.loadCommandFrame(
                    memberOrdinal: selected.member.ordinal,
                    commandOrdinal: ordinal
                ))
            }
            guard recognizedLoadCommands.contains(command) else {
                throw Failure.rangePlan(.unknownLoadCommand(
                    memberOrdinal: selected.member.ordinal,
                    commandOrdinal: ordinal,
                    command: command
                ))
            }
            guard !forbiddenLoadCommands.contains(command) else {
                throw Failure.rangePlan(.forbiddenLoadCommand(
                    memberOrdinal: selected.member.ordinal,
                    commandOrdinal: ordinal,
                    command: command
                ))
            }

            switch command {
            case 0x1b:
                guard size == 24, uuid == nil,
                      let payloadStart = optionalCheckedAdd(cursor, 8),
                      let payloadEnd = optionalCheckedAdd(cursor, 24)
                else {
                    throw Failure.rangePlan(.uuid(
                        memberOrdinal: selected.member.ordinal
                    ))
                }
                let value = Data(bytes[payloadStart..<payloadEnd])
                guard value.contains(where: { $0 != 0 }) else {
                    throw Failure.rangePlan(.uuid(
                        memberOrdinal: selected.member.ordinal
                    ))
                }
                uuid = value

            case 0x0d:
                guard let frameEnd = optionalCheckedAdd(cursor, size),
                      !sawIdentity,
                      validDylibIdentity(
                        bytes,
                        range: cursor..<frameEnd
                      )
                else {
                    throw Failure.rangePlan(.dylibIdentity(
                        memberOrdinal: selected.member.ordinal
                    ))
                }
                sawIdentity = true

            case 0x1d:
                guard let dataOffsetPosition = optionalCheckedAdd(cursor, 8),
                      let dataSizePosition = optionalCheckedAdd(cursor, 12),
                      size == 16, signature == nil,
                      let dataOffset = readUInt32LE(
                        bytes,
                        at: dataOffsetPosition
                      ),
                      let dataSize = readUInt32LE(
                        bytes,
                        at: dataSizePosition
                      ),
                      dataSize > 0,
                      dataSize <= maximumCodeSignatureBytes
                else {
                    throw Failure.rangePlan(.codeSignature(
                        memberOrdinal: selected.member.ordinal
                    ))
                }
                signature = (UInt64(dataOffset), UInt64(dataSize))

            case 0x19:
                guard let frameEnd = optionalCheckedAdd(cursor, size) else {
                    throw Failure.rangePlan(.loadCommandFrame(
                        memberOrdinal: selected.member.ordinal,
                        commandOrdinal: ordinal
                    ))
                }
                if let linkedit = try parseLinkedit(
                    bytes,
                    range: cursor..<frameEnd,
                    memberOrdinal: selected.member.ordinal
                ) {
                    linkedits.append(linkedit)
                }

            default:
                break
            }
            guard let nextCursor = optionalCheckedAdd(cursor, size) else {
                throw Failure.rangePlan(.loadCommandFrame(
                    memberOrdinal: selected.member.ordinal,
                    commandOrdinal: ordinal
                ))
            }
            cursor = nextCursor
        }
        guard cursor == bytes.count else {
            throw Failure.rangePlan(.loadCommandFrame(
                memberOrdinal: selected.member.ordinal,
                commandOrdinal: Int(header.commandCount)
            ))
        }
        guard let uuid, sawIdentity else {
            throw Failure.rangePlan(
                uuid == nil
                    ? .uuid(memberOrdinal: selected.member.ordinal)
                    : .dylibIdentity(memberOrdinal: selected.member.ordinal)
            )
        }
        guard hex(uuid) == selected.member.machOUUID,
              sha256Hex(bytes) == selected.member.loadCommandsSHA256
        else {
            throw Failure.rangePlan(.expectedMember(
                memberOrdinal: selected.member.ordinal
            ))
        }

        guard let signature else {
            guard selected.member.primaryCodeDirectoryBlobSHA256 ==
                    String(repeating: "0", count: 64)
            else {
                throw Failure.rangePlan(.expectedMember(
                    memberOrdinal: selected.member.ordinal
                ))
            }
            return LoadCommandFacts(signature: nil)
        }
        guard linkedits.count == 1
        else {
            throw Failure.rangePlan(.linkedit(
                memberOrdinal: selected.member.ordinal
            ))
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
            throw Failure.rangePlan(.linkedit(
                memberOrdinal: selected.member.ordinal
            ))
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
            throw Failure.rangePlan(.linkedit(
                memberOrdinal: selected.member.ordinal
            ))
        }
        let translated = try translate(
            vmStart: vmStart,
            length: signature.size,
            mappings: mappings
        )
        return LoadCommandFacts(signature: SignatureRequest(
            fileOrdinal: translated.fileOrdinal,
            fileStart: translated.fileStart,
            length: signature.size
        ))
    }

    static func validDylibIdentity(
        _ bytes: Data,
        range: Range<Int>
    ) -> Bool {
        guard let nameOffsetField = optionalCheckedAdd(
            range.lowerBound,
            8
        ) else {
            return false
        }
        guard range.count >= 32,
              let rawNameOffset = readUInt32LE(bytes, at: nameOffsetField),
              let nameOffset = Int(exactly: rawNameOffset),
              nameOffset >= 24,
              nameOffset < range.count
        else {
            return false
        }
        guard let start = optionalCheckedAdd(range.lowerBound, nameOffset),
              let fixedFieldsEnd = optionalCheckedAdd(range.lowerBound, 24),
              start <= range.upperBound
        else {
            return false
        }
        if bytes[fixedFieldsEnd..<start]
            .contains(where: { $0 != 0 })
        {
            return false
        }
        guard let terminator = bytes[start..<range.upperBound]
            .firstIndex(of: 0)
        else {
            return false
        }
        let length = terminator - start
        guard (1...4_096).contains(length) else { return false }
        guard let trailingStart = optionalCheckedAdd(terminator, 1) else {
            return false
        }
        return !bytes[trailingStart..<range.upperBound]
            .contains(where: { $0 != 0 })
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
              )
        else {
            throw Failure.rangePlan(.linkedit(
                memberOrdinal: memberOrdinal
            ))
        }
        guard range.count >= 72,
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
            throw Failure.rangePlan(.linkedit(
                memberOrdinal: memberOrdinal
            ))
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
            throw Failure.rangePlan(.linkedit(
                memberOrdinal: memberOrdinal
            ))
        }
        return LinkeditFacts(
            vmAddress: vmAddress,
            vmSize: vmSize,
            fileOffset: fileOffset,
            fileSize: fileSize
        )
    }

    static func discoveryBindings(
        files: [SyntheticSharedCacheSetPlanComparison.HeaderOrderFile],
        stores: [Comparison.DiscoveryStore]
    ) throws -> [RawBinding] {
        var result: [RawBinding] = []
        for file in files {
            let bytes = stores[file.ordinal].bytes
            result.append(try rawBinding(
                kind: .setHeader,
                ordinal: file.ordinal,
                fileOrdinal: file.ordinal,
                start: 0,
                length: 552,
                source: bytes
            ))
            result.append(try rawBinding(
                kind: .mappingTable,
                ordinal: file.ordinal,
                fileOrdinal: file.ordinal,
                start: UInt64(file.header.mappingOffset),
                length: file.header.mappingTablesEnd -
                    UInt64(file.header.mappingOffset),
                source: bytes
            ))
        }
        let main = files[0]
        let imageBytes = try checkedMultiply(
            UInt64(main.header.imagesCount),
            32
        )
        result.append(try rawBinding(
            kind: .imageTable,
            ordinal: 0,
            fileOrdinal: 0,
            start: main.header.imagesOffset,
            length: imageBytes,
            source: stores[0].bytes
        ))
        return result
    }

    static func rawBinding(
        kind: SyntheticSharedCacheRangeConsumerKind,
        ordinal: Int,
        fileOrdinal: Int,
        start: UInt64,
        length: UInt64,
        source: Data
    ) throws -> RawBinding {
        guard length > 0,
              Int(exactly: start) != nil,
              let end = optionalCheckedAdd(start, length),
              let upper = Int(exactly: end),
              upper <= source.count
        else {
            throw Failure.discoveryContinuity(
                fileOrdinal: fileOrdinal,
                reason: .derivedRange
            )
        }
        return RawBinding(
            kind: kind,
            consumerOrdinal: ordinal,
            fileOrdinal: fileOrdinal,
            start: start,
            length: length,
            byteSource: .discovery
        )
    }

    static func normalize(
        _ input: [RawBinding],
        discoveryStores: [Comparison.DiscoveryStore]
    ) throws -> NormalizedResult {
        var sorted = input.sorted(by: bindingPrecedes)
        var unique: [RawBinding] = []
        for binding in sorted {
            if let previous = unique.last,
               previous.kind == binding.kind,
               previous.consumerOrdinal == binding.consumerOrdinal,
               previous.fileOrdinal == binding.fileOrdinal,
               previous.start == binding.start,
               previous.length == binding.length
            {
                guard bindingBytesEqual(
                    previous,
                    binding,
                    over: previous.start..<previous.end,
                    discoveryStores: discoveryStores
                ) else {
                    throw Failure.rangePlan(.overlapBytes)
                }
                continue
            }
            unique.append(binding)
        }
        sorted = unique

        for binding in sorted {
            let discoveryEnd = UInt64(
                discoveryStores[binding.fileOrdinal].bytes.count
            )
            let intersectionEnd = min(binding.end, discoveryEnd)
            guard binding.start >= intersectionEnd || bindingBytesEqual(
                binding,
                RawBinding(
                    kind: binding.kind,
                    consumerOrdinal: binding.consumerOrdinal,
                    fileOrdinal: binding.fileOrdinal,
                    start: binding.start,
                    length: intersectionEnd - binding.start,
                    byteSource: .discovery
                ),
                over: binding.start..<intersectionEnd,
                discoveryStores: discoveryStores
            ) else {
                throw Failure.rangePlan(.overlapBytes)
            }
        }

        for leftIndex in sorted.indices {
            for rightIndex in sorted.indices where rightIndex > leftIndex {
                let left = sorted[leftIndex]
                let right = sorted[rightIndex]
                if right.fileOrdinal != left.fileOrdinal || right.start >= left.end {
                    break
                }
                let overlapStart = max(left.start, right.start)
                let overlapEnd = min(left.end, right.end)
                guard overlapStart < overlapEnd else { continue }
                guard bindingBytesEqual(
                    left,
                    right,
                    over: overlapStart..<overlapEnd,
                    discoveryStores: discoveryStores
                )
                else {
                    throw Failure.rangePlan(.overlapBytes)
                }
            }
        }

        var intervals: [CoalescedInterval] = []
        for binding in sorted {
            if let last = intervals.last,
               last.fileOrdinal == binding.fileOrdinal,
               binding.start <= last.end
            {
                intervals[intervals.count - 1] = CoalescedInterval(
                    fileOrdinal: last.fileOrdinal,
                    start: last.start,
                    end: max(last.end, binding.end)
                )
            } else {
                intervals.append(CoalescedInterval(
                    fileOrdinal: binding.fileOrdinal,
                    start: binding.start,
                    end: binding.end
                ))
            }
        }
        guard (1...maximumLogicalRangeCount).contains(intervals.count) else {
            throw Failure.rangePlan(.logicalRangeLimit(
                actual: intervals.count,
                maximum: maximumLogicalRangeCount
            ))
        }

        var logical: [Comparison.LogicalSelectedRange] = []
        var additional: [Comparison.AdditionalPhysicalRange] = []
        var additionalBytes: UInt64 = 0
        for (ordinal, interval) in intervals.enumerated() {
            logical.append(.init(
                ordinal: ordinal,
                fileOrdinal: interval.fileOrdinal,
                start: interval.start,
                length: interval.end - interval.start
            ))
            let discoveryEnd = UInt64(
                discoveryStores[interval.fileOrdinal].bytes.count
            )
            let additionalStart = max(interval.start, discoveryEnd)
            if additionalStart < interval.end {
                let bytes = try bytesForRange(
                    additionalStart..<interval.end,
                    fileOrdinal: interval.fileOrdinal,
                    bindings: sorted,
                    discoveryStores: discoveryStores
                )
                let length = interval.end - additionalStart
                additionalBytes = try checkedAdd(additionalBytes, length)
                additional.append(.init(
                    ordinal: additional.count,
                    fileOrdinal: interval.fileOrdinal,
                    start: additionalStart,
                    length: length,
                    bytes: bytes
                ))
            }
        }
        guard additionalBytes <= maximumAdditionalBytes else {
            throw Failure.rangePlan(.additionalByteLimit(
                actual: additionalBytes,
                maximum: maximumAdditionalBytes
            ))
        }

        var consumerBindings: [Comparison.ConsumerBinding] = []
        consumerBindings.reserveCapacity(sorted.count)
        for binding in sorted {
            var pieces: [Comparison.PhysicalPiece] = []
            let discoveryEnd = UInt64(
                discoveryStores[binding.fileOrdinal].bytes.count
            )
            if binding.start < discoveryEnd {
                let end = min(binding.end, discoveryEnd)
                pieces.append(.discovery(
                    fileOrdinal: binding.fileOrdinal,
                    relativeStart: binding.start,
                    length: end - binding.start
                ))
            }
            if binding.end > discoveryEnd {
                let start = max(binding.start, discoveryEnd)
                guard let range = additional.first(where: {
                    guard $0.fileOrdinal == binding.fileOrdinal,
                          $0.start <= start,
                          let rangeEnd = optionalCheckedAdd(
                            $0.start,
                            $0.length
                          )
                    else {
                        return false
                    }
                    return binding.end <= rangeEnd
                }) else {
                    throw Failure.rangePlan(.consumerPieces)
                }
                pieces.append(.additional(
                    rangeOrdinal: range.ordinal,
                    relativeStart: start - range.start,
                    length: binding.end - start
                ))
            }
            var totalPieceLength: UInt64 = 0
            for piece in pieces {
                totalPieceLength = try checkedAdd(
                    totalPieceLength,
                    pieceLength(piece)
                )
            }
            guard totalPieceLength == binding.length else {
                throw Failure.rangePlan(.consumerPieces)
            }
            consumerBindings.append(.init(
                consumerKind: binding.kind,
                consumerOrdinal: binding.consumerOrdinal,
                fileOrdinal: binding.fileOrdinal,
                start: binding.start,
                length: binding.length,
                pieces: pieces
            ))
        }
        return NormalizedResult(
            logicalRanges: logical,
            additionalRanges: additional,
            consumerBindings: consumerBindings,
            canonicalBindings: sorted,
            additionalBytes: additionalBytes
        )
    }

    static func resourceCounts(
        discovery: EnvelopeCounts,
        probes: EnvelopeCounts,
        logicalRangeCount: Int,
        additionalBytes: UInt64
    ) throws -> Comparison.ResourceCounts {
        let totalAttempts = try checkedAdd(
            discovery.attemptCount,
            probes.attemptCount
        )
        let combined = try checkedAdd(discovery.bytes, additionalBytes)
        guard combined <= maximumCombinedRetainedBytes else {
            throw Failure.rangePlan(.retainedByteLimit(
                actual: combined,
                maximum: maximumCombinedRetainedBytes
            ))
        }
        let discoveryAndRaw = try checkedAdd(discovery.bytes, probes.bytes)
        let verifierOwnedPeak = try checkedAdd(
            discoveryAndRaw,
            additionalBytes
        )
        guard verifierOwnedPeak <= maximumVerifierOwnedContentBytes else {
            throw Failure.rangePlan(.transientByteLimit(
                actual: verifierOwnedPeak,
                maximum: maximumVerifierOwnedContentBytes
            ))
        }
        return Comparison.ResourceCounts(
            discoveryBytes: discovery.bytes,
            rawProbeRequestedBytes: probes.bytes,
            probeChunkCount: probes.chunks,
            discoveryAttemptCount: discovery.attemptCount,
            planProbeAttemptCount: probes.attemptCount,
            totalAttemptCount: totalAttempts,
            logicalRangeCount: logicalRangeCount,
            additionalBytes: additionalBytes,
            combinedRetainedBytes: combined
        )
    }

    static func validateFinalStores(
        rawBindings: [RawBinding],
        discoveryStores: [Comparison.DiscoveryStore],
        additional: [Comparison.AdditionalPhysicalRange],
        bindings: [Comparison.ConsumerBinding]
    ) throws {
        guard rawBindings.count == bindings.count else {
            throw Failure.rangePlan(.rederivation)
        }
        for index in rawBindings.indices {
            let raw = rawBindings[index]
            let binding = bindings[index]
            guard raw.kind == binding.consumerKind,
                  raw.consumerOrdinal == binding.consumerOrdinal,
                  raw.fileOrdinal == binding.fileOrdinal,
                  raw.start == binding.start,
                  raw.length == binding.length,
                  try piecesMatch(
                    binding.pieces,
                    rawBinding: raw,
                    discoveryStores: discoveryStores,
                    additional: additional
                  )
            else {
                throw Failure.rangePlan(.rederivation)
            }
        }
    }

    static func rederiveFinal(
        discoveryStores: [Comparison.DiscoveryStore],
        additional: [Comparison.AdditionalPhysicalRange],
        bindings: [Comparison.ConsumerBinding],
        files: [SyntheticSharedCacheSetPlanComparison.HeaderOrderFile],
        members: [SharedMember],
        mappings: [Mapping]
    ) throws {
        for file in files {
            _ = try requireBinding(
                .setHeader,
                ordinal: file.ordinal,
                fileOrdinal: file.ordinal,
                start: 0,
                length: 552,
                from: bindings
            )
            _ = try requireBinding(
                .mappingTable,
                ordinal: file.ordinal,
                fileOrdinal: file.ordinal,
                start: UInt64(file.header.mappingOffset),
                length: file.header.mappingTablesEnd -
                    UInt64(file.header.mappingOffset),
                from: bindings
            )
        }
        let main = files[0]
        let imageTableLength = try checkedMultiply(
            UInt64(main.header.imagesCount),
            32
        )
        _ = try requireBinding(
            .imageTable,
            ordinal: 0,
            fileOrdinal: 0,
            start: main.header.imagesOffset,
            length: imageTableLength,
            from: bindings
        )
        let rows = try imageRows(
            main: main,
            discovery: discoveryStores[0].bytes,
            membersAreNonempty: !members.isEmpty
        )
        let window = try nameWindow(
            rows: rows,
            mainFileBytes: main.fileBytes
        )
        let nameBinding = try requireBinding(
            .imageNameWindow,
            ordinal: 0,
            fileOrdinal: 0,
            start: window.lowerBound,
            length: window.upperBound - window.lowerBound,
            from: bindings
        )
        let selected: [SelectedImage]
        do {
            let nameBytes = try finalOwnedBytes(
                nameBinding,
                discoveryStores: discoveryStores,
                additional: additional
            )
            selected = try selectedImages(
                members: members,
                rows: rows,
                window: window,
                bytes: nameBytes
            )
        }

        var headers: [Int: MachHeaderFacts] = [:]
        for image in selected {
            let install = try requireBinding(
                .installName,
                ordinal: image.member.ordinal,
                fileOrdinal: 0,
                start: image.pathStart,
                length: UInt64(image.member.decodedName.count),
                from: bindings
            )
            guard try finalOwnedBytes(
                install,
                discoveryStores: discoveryStores,
                additional: additional
            ) == image.member.decodedName else {
                throw Failure.rangePlan(.rederivation)
            }
            let translated = try translate(
                vmStart: image.row.address,
                length: 32,
                mappings: mappings
            )
            let headerBinding = try requireBinding(
                .machOHeader,
                ordinal: image.member.ordinal,
                fileOrdinal: translated.fileOrdinal,
                start: translated.fileStart,
                length: 32,
                from: bindings
            )
            let header = try parseMachHeader(
                finalOwnedBytes(
                    headerBinding,
                    discoveryStores: discoveryStores,
                    additional: additional
                ),
                memberOrdinal: image.member.ordinal
            )
            _ = try translate(
                vmStart: try checkedAdd(image.row.address, 32),
                length: UInt64(header.sizeOfCommands),
                mappings: mappings
            )
            headers[image.member.ordinal] = header
        }

        var signatures = 0
        for image in selected {
            guard let header = headers[image.member.ordinal] else {
                throw Failure.rangePlan(.rederivation)
            }
            let commandStart = try checkedAdd(image.row.address, 32)
            let translated = try translate(
                vmStart: commandStart,
                length: UInt64(header.sizeOfCommands),
                mappings: mappings
            )
            let commandsBinding = try requireBinding(
                .loadCommands,
                ordinal: image.member.ordinal,
                fileOrdinal: translated.fileOrdinal,
                start: translated.fileStart,
                length: UInt64(header.sizeOfCommands),
                from: bindings
            )
            let facts = try parseLoadCommands(
                finalOwnedBytes(
                    commandsBinding,
                    discoveryStores: discoveryStores,
                    additional: additional
                ),
                header: header,
                selected: image,
                mappings: mappings
            )
            if let signature = facts.signature {
                signatures += 1
                let signatureBinding = try requireBinding(
                    .codeSignature,
                    ordinal: image.member.ordinal,
                    fileOrdinal: signature.fileOrdinal,
                    start: signature.fileStart,
                    length: signature.length,
                    from: bindings
                )
                try validateSignatureProbe(
                    finalOwnedBytes(
                        signatureBinding,
                        discoveryStores: discoveryStores,
                        additional: additional
                    ),
                    selected: image
                )
            }
        }

        guard let twiceFileCount = optionalCheckedMultiply(files.count, 2),
              let fixedCount = optionalCheckedAdd(twiceFileCount, 2),
              let thriceMemberCount = optionalCheckedMultiply(
                members.count,
                3
              ),
              let withoutSignatures = optionalCheckedAdd(
                fixedCount,
                thriceMemberCount
              ),
                  let expectedCount = optionalCheckedAdd(
                    withoutSignatures,
                    signatures
                  ),
              bindings.count == expectedCount
        else {
            throw Failure.rangePlan(.rederivation)
        }
    }

    static func requireBinding(
        _ kind: SyntheticSharedCacheRangeConsumerKind,
        ordinal: Int,
        fileOrdinal: Int,
        start: UInt64,
        length: UInt64,
        from bindings: [Comparison.ConsumerBinding]
    ) throws -> Comparison.ConsumerBinding {
        let matches = bindings.filter {
            $0.consumerKind == kind && $0.consumerOrdinal == ordinal
        }
        guard matches.count == 1,
              matches[0].fileOrdinal == fileOrdinal,
              matches[0].start == start,
              matches[0].length == length
        else {
            throw Failure.rangePlan(.rederivation)
        }
        return matches[0]
    }

    static func finalOwnedBytes(
        _ binding: Comparison.ConsumerBinding,
        discoveryStores: [Comparison.DiscoveryStore],
        additional: [Comparison.AdditionalPhysicalRange]
    ) throws -> Data {
        guard let capacity = Int(exactly: binding.length) else {
            throw Failure.rangePlan(.arithmeticOverflow)
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
                      fileOrdinal < discoveryStores.count,
                      let start = Int(exactly: relativeStart),
                      let count = Int(exactly: pieceLength),
                      let end = optionalCheckedAdd(start, count),
                      end <= discoveryStores[fileOrdinal].bytes.count
                else {
                    throw Failure.rangePlan(.rederivation)
                }
                source = discoveryStores[fileOrdinal].bytes
                lower = start
                upper = end
                length = pieceLength
            case let .additional(rangeOrdinal, relativeStart, pieceLength):
                guard rangeOrdinal >= 0,
                      rangeOrdinal < additional.count,
                      additional[rangeOrdinal].fileOrdinal ==
                        binding.fileOrdinal,
                      let start = Int(exactly: relativeStart),
                      let count = Int(exactly: pieceLength),
                      let end = optionalCheckedAdd(start, count),
                      end <= additional[rangeOrdinal].bytes.count
                else {
                    throw Failure.rangePlan(.rederivation)
                }
                source = additional[rangeOrdinal].bytes
                lower = start
                upper = end
                length = pieceLength
            }
            result.append(contentsOf: source[lower..<upper])
            copied = try checkedAdd(copied, length)
        }
        guard copied == binding.length,
              result.count == capacity
        else {
            throw Failure.rangePlan(.rederivation)
        }
        return result
    }

    static func piecesMatch(
        _ pieces: [Comparison.PhysicalPiece],
        rawBinding: RawBinding,
        discoveryStores: [Comparison.DiscoveryStore],
        additional: [Comparison.AdditionalPhysicalRange]
    ) throws -> Bool {
        guard let (rawData, rawBase) = bindingDataAndBase(
            rawBinding,
            discoveryStores: discoveryStores
        ) else {
            return false
        }
        var cursor = rawBinding.start
        for piece in pieces {
            let pieceData: Data
            let lower: Int
            let upper: Int
            let length: UInt64
            switch piece {
            case let .discovery(fileOrdinal, relativeStart, pieceLength):
                guard fileOrdinal == rawBinding.fileOrdinal,
                      fileOrdinal >= 0,
                      fileOrdinal < discoveryStores.count,
                      pieceLength > 0,
                      relativeStart == cursor,
                      let pieceLower = Int(exactly: relativeStart),
                      let pieceEnd = optionalCheckedAdd(
                        relativeStart,
                        pieceLength
                      ),
                      let pieceUpper = Int(exactly: pieceEnd),
                      pieceUpper <= discoveryStores[fileOrdinal].bytes.count
                else {
                    return false
                }
                pieceData = discoveryStores[fileOrdinal].bytes
                lower = pieceLower
                upper = pieceUpper
                length = pieceLength
            case let .additional(rangeOrdinal, relativeStart, pieceLength):
                guard rangeOrdinal >= 0,
                      rangeOrdinal < additional.count,
                      additional[rangeOrdinal].fileOrdinal ==
                        rawBinding.fileOrdinal,
                      additional[rangeOrdinal].ordinal == rangeOrdinal,
                      UInt64(additional[rangeOrdinal].bytes.count) ==
                        additional[rangeOrdinal].length,
                      pieceLength > 0,
                      let pieceLogicalStart = optionalCheckedAdd(
                        additional[rangeOrdinal].start,
                        relativeStart
                      ),
                      pieceLogicalStart == cursor,
                      let pieceLower = Int(exactly: relativeStart),
                      let pieceEnd = optionalCheckedAdd(
                        relativeStart,
                        pieceLength
                      ),
                      let pieceUpper = Int(exactly: pieceEnd),
                      pieceUpper <= additional[rangeOrdinal].bytes.count
                else {
                    return false
                }
                pieceData = additional[rangeOrdinal].bytes
                lower = pieceLower
                upper = pieceUpper
                length = pieceLength
            }
            guard cursor >= rawBase,
                  let rawLower = Int(exactly: cursor - rawBase),
                  let next = optionalCheckedAdd(cursor, length),
                  next >= rawBase,
                  let rawUpper = Int(exactly: next - rawBase),
                  rawUpper <= rawData.count,
                  rawData[rawLower..<rawUpper]
                    .elementsEqual(pieceData[lower..<upper])
            else {
                return false
            }
            cursor = next
        }
        return cursor == rawBinding.end
    }

    static func bytesForRange(
        _ range: Range<UInt64>,
        fileOrdinal: Int,
        bindings: [RawBinding],
        discoveryStores: [Comparison.DiscoveryStore]
    ) throws -> Data {
        guard let capacity = Int(exactly: range.upperBound - range.lowerBound)
        else {
            throw Failure.rangePlan(.arithmeticOverflow)
        }
        var result = Data()
        result.reserveCapacity(capacity)
        var cursor = range.lowerBound
        while cursor < range.upperBound {
            guard let binding = bindings.first(where: {
                $0.fileOrdinal == fileOrdinal &&
                $0.start <= cursor && cursor < $0.end
            }) else {
                throw Failure.rangePlan(.consumerPieces)
            }
            let end = min(binding.end, range.upperBound)
            guard let (source, base) = bindingDataAndBase(
                binding,
                discoveryStores: discoveryStores
            ), cursor >= base, end >= base else {
                throw Failure.rangePlan(.consumerPieces)
            }
            let relativeStart = cursor - base
            let relativeEnd = end - base
            guard let lower = Int(exactly: relativeStart),
                  let upper = Int(exactly: relativeEnd),
                  upper <= source.count
            else {
                throw Failure.rangePlan(.arithmeticOverflow)
            }
            result.append(source[lower..<upper])
            cursor = end
        }
        return result
    }

    static func bindingBytesEqual(
        _ left: RawBinding,
        _ right: RawBinding,
        over range: Range<UInt64>,
        discoveryStores: [Comparison.DiscoveryStore]
    ) -> Bool {
        guard let (leftData, leftBase) = bindingDataAndBase(
                left,
                discoveryStores: discoveryStores
              ),
              let (rightData, rightBase) = bindingDataAndBase(
                right,
                discoveryStores: discoveryStores
              ),
              range.lowerBound >= leftBase,
              range.lowerBound >= rightBase,
              let leftLower = Int(exactly: range.lowerBound - leftBase),
              let rightLower = Int(exactly: range.lowerBound - rightBase),
              let count = Int(exactly:
                range.upperBound - range.lowerBound
              ),
              let leftUpper = optionalCheckedAdd(leftLower, count),
              let rightUpper = optionalCheckedAdd(rightLower, count),
              leftUpper <= leftData.count,
              rightUpper <= rightData.count
        else {
            return false
        }
        return leftData[leftLower..<leftUpper]
            .elementsEqual(rightData[rightLower..<rightUpper])
    }

    static func bindingDataAndBase(
        _ binding: RawBinding,
        discoveryStores: [Comparison.DiscoveryStore]
    ) -> (Data, UInt64)? {
        switch binding.byteSource {
        case .discovery:
            guard binding.fileOrdinal >= 0,
                  binding.fileOrdinal < discoveryStores.count
            else {
                return nil
            }
            return (discoveryStores[binding.fileOrdinal].bytes, 0)
        case let .owned(bytes, base):
            return (bytes, base)
        }
    }

    static func bindingPrecedes(_ left: RawBinding, _ right: RawBinding)
        -> Bool
    {
        if left.fileOrdinal != right.fileOrdinal {
            return left.fileOrdinal < right.fileOrdinal
        }
        if left.start != right.start { return left.start < right.start }
        if left.end != right.end { return left.end < right.end }
        if left.kind.rawValue != right.kind.rawValue {
            return left.kind.rawValue < right.kind.rawValue
        }
        return left.consumerOrdinal < right.consumerOrdinal
    }

    static func pieceLength(_ piece: Comparison.PhysicalPiece) -> UInt64 {
        switch piece {
        case let .discovery(_, _, length),
             let .additional(_, _, length):
            return length
        }
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
             expected.modificationTimeSeconds == actual.modificationTimeSeconds),
            (.modificationTimeNanoseconds,
             expected.modificationTimeNanoseconds == actual.modificationTimeNanoseconds),
            (.statusChangeTimeSeconds,
             expected.statusChangeTimeSeconds == actual.statusChangeTimeSeconds),
            (.statusChangeTimeNanoseconds,
             expected.statusChangeTimeNanoseconds == actual.statusChangeTimeNanoseconds),
            (.birthTimeSeconds,
             expected.birthTimeSeconds == actual.birthTimeSeconds),
            (.birthTimeNanoseconds,
             expected.birthTimeNanoseconds == actual.birthTimeNanoseconds),
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
        bytes / maximumLogicalChunkBytes +
            (bytes % maximumLogicalChunkBytes == 0 ? 0 : 1)
    }

    static func checkedAdd(_ left: UInt64, _ right: UInt64) throws
        -> UInt64
    {
        let (result, overflow) = left.addingReportingOverflow(right)
        guard !overflow else {
            throw Failure.discoveryTranscript(.arithmeticOverflow)
        }
        return result
    }

    static func checkedMultiply(_ left: UInt64, _ right: UInt64) throws
        -> UInt64
    {
        let (result, overflow) = left.multipliedReportingOverflow(by: right)
        guard !overflow else {
            throw Failure.rangePlan(.arithmeticOverflow)
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

    static func sha256Hex(_ bytes: Data) -> String {
        SHA256.hash(data: bytes)
            .map { String(format: "%02x", $0) }
            .joined()
    }

    static func hex(_ bytes: Data) -> String {
        bytes.map { String(format: "%02x", $0) }.joined()
    }

    static func isCanonicalInstallName(
        _ bytes: Data,
        range: Range<Int>
    ) -> Bool {
        guard (2...SyntheticRuntimeClosureInstallNameVerifier
            .maximumInstallNameBytes).contains(range.count),
              range.lowerBound >= bytes.startIndex,
              range.upperBound <= bytes.endIndex,
              bytes[range.lowerBound] == 0x2f
        else {
            return false
        }
        for index in (range.lowerBound + 1)..<range.upperBound {
            let byte = bytes[index]
            guard (0x21...0x7e).contains(byte),
                  byte != 0x40,
                  byte != 0x5c
            else {
                return false
            }
        }

        var componentStart = range.lowerBound + 1
        for index in (range.lowerBound + 1)...range.upperBound {
            guard index == range.upperBound || bytes[index] == 0x2f else {
                continue
            }
            let count = index - componentStart
            guard count > 0,
                  count != 1 || bytes[componentStart] != 0x2e,
                  count != 2 ||
                    bytes[componentStart] != 0x2e ||
                    bytes[componentStart + 1] != 0x2e
            else {
                return false
            }
            componentStart = index + 1
        }
        return true
    }
}
