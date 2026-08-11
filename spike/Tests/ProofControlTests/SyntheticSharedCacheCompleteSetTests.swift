import CryptoKit
import Foundation
import XCTest

@testable import ProofControl

final class SyntheticSharedCacheCompleteSetTests: XCTestCase {
    func testE2CExactFourTypeExistenceSentinel() {
        _ = SyntheticSharedCacheCompleteFileTranscript.self
        _ = SyntheticSharedCacheCompleteSetFailure.self
        _ = SyntheticSharedCacheCompleteSetComparison.self
        _ = SyntheticSharedCacheCompleteSetVerifier.self
    }

    func testMainPlusTwoSubcacheFixtureProducesExactInertEvidence()
        throws
    {
        let fixture = try Self.fixture()
        let result = try Self.compare(fixture)

        XCTAssertEqual(
            fixture.setPlan.headerOrderFiles.map(\.decodedSuffix),
            [Data(), Data(".2".utf8), Data(".1".utf8)]
        )
        XCTAssertEqual(result.cacheSetEvidence, fixture.cacheSetEvidence)
        XCTAssertEqual(
            result.sharedCacheImageEvidence,
            [fixture.imageEvidence]
        )
        XCTAssertEqual(result.resourceCounts.fileCount, 3)
        XCTAssertEqual(
            result.resourceCounts.aggregateFileBytes,
            3 * UInt64(Self.fileBytes)
        )
        XCTAssertEqual(result.resourceCounts.streamChunkCount, 6)
        XCTAssertEqual(result.resourceCounts.streamAttemptCount, 9)
        XCTAssertEqual(result.resourceCounts.memberCount, 1)
        XCTAssertGreaterThan(result.resourceCounts.matchedRetainedBytes, 0)
        let expectedCompactEvidenceBytes = UInt64(
            fixture.cacheSetEvidence.identityPreimage.count
                + fixture.cacheSetEvidence.decodedSuffixes.reduce(0) {
                    $0 + $1.count
                }
                + fixture.imageEvidence.identityPreimage.count
                + fixture.imageEvidence.decodedInstallName.count
                + fixture.imageEvidence.facts.installNameBase64URL.utf8.count
                + 64
                + 64
                + 32
        )
        XCTAssertEqual(
            result.resourceCounts.compactEvidenceSemanticBytes,
            expectedCompactEvidenceBytes
        )
        XCTAssertEqual(result.runtimeDecision, .noGo)
        Self.assertNoAuthority(result)
    }

    func testMainOnlyFixtureProducesExactInertEvidence() throws {
        let fixture = try Self.fixture(suffixes: [])
        let result = try Self.compare(fixture)

        XCTAssertEqual(fixture.setPlan.headerOrderFiles.count, 1)
        XCTAssertEqual(result.cacheSetEvidence, fixture.cacheSetEvidence)
        XCTAssertEqual(result.sharedCacheImageEvidence, [fixture.imageEvidence])
        XCTAssertEqual(result.resourceCounts.fileCount, 1)
        XCTAssertEqual(
            result.resourceCounts.aggregateFileBytes,
            UInt64(Self.fileBytes)
        )
        XCTAssertEqual(result.resourceCounts.streamChunkCount, 2)
        XCTAssertEqual(result.resourceCounts.streamAttemptCount, 3)
        Self.assertNoAuthority(result)
    }

    func testE2CReparsesTheAllowedCommandProfileAndPreservesIDNameBoundary()
        throws
    {
        let genericAllowed: [UInt32] = [
            0x00000001, 0x00000002, 0x00000003, 0x00000004,
            0x00000005, 0x00000006, 0x00000007, 0x00000008,
            0x00000009, 0x0000000a, 0x0000000b, 0x0000000c,
            0x00000010, 0x00000011, 0x00000012, 0x00000013,
            0x00000014, 0x00000015, 0x00000016, 0x00000017,
            0x80000018, 0x0000001a, 0x8000001c, 0x0000001e,
            0x8000001f, 0x00000020, 0x00000022, 0x80000022,
            0x80000023, 0x00000024, 0x00000025, 0x00000026,
            0x00000029, 0x0000002a, 0x0000002b, 0x0000002d,
            0x0000002e, 0x0000002f, 0x00000030, 0x00000031,
            0x00000032, 0x80000033, 0x80000034,
        ]
        for command in genericAllowed {
            var commands = Self.fixtureLoadCommands()
            commands.append(Self.passthroughCommand(command, size: 8))
            let fixture = try Self.fixture(
                suffixes: [],
                loadCommands: commands,
                commandCount: 3
            )
            let result = try Self.compare(fixture)
            XCTAssertEqual(
                result.sharedCacheImageEvidence[0]
                    .facts.loadCommandsSHA256,
                Self.sha256(commands),
                "command=\(String(format: "0x%08x", command))"
            )
        }

        var segmentCommands = Self.fixtureLoadCommands()
        segmentCommands.append(Self.nonLinkeditSegmentCommand())
        let segmentFixture = try Self.fixture(
            suffixes: [],
            loadCommands: segmentCommands,
            commandCount: 3
        )
        let segmentResult = try Self.compare(segmentFixture)
        XCTAssertEqual(segmentResult.resourceCounts.memberCount, 1)
        XCTAssertNotEqual(
            Data("x".utf8),
            segmentResult.sharedCacheImageEvidence[0].decodedInstallName
        )
    }

    func testE2CWrapsForbiddenAndUnknownCommandPredecessorFailures()
        throws
    {
        let forbidden: [UInt32] = [
            0x00000027, 0x80000028, 0x0000000e, 0x00000021,
            0x0000002c, 0x80000035, 0x0000000f,
        ]
        for command in forbidden {
            var commands = Self.fixtureLoadCommands()
            commands.append(Self.passthroughCommand(command, size: 8))
            let fixture = try Self.fixture(
                suffixes: [],
                loadCommands: commands,
                commandCount: 3
            )
            Self.assertFailure(try Self.compare(fixture)) {
                $0 == .predecessor(.rangePlan(.forbiddenLoadCommand(
                    memberOrdinal: 0,
                    commandOrdinal: 2,
                    command: command
                )))
            }
        }

        var unknown = Self.fixtureLoadCommands()
        unknown.append(Self.passthroughCommand(0x00000036, size: 8))
        let fixture = try Self.fixture(
            suffixes: [],
            loadCommands: unknown,
            commandCount: 3
        )
        Self.assertFailure(try Self.compare(fixture)) {
            $0 == .predecessor(.rangePlan(.unknownLoadCommand(
                memberOrdinal: 0,
                commandOrdinal: 2,
                command: 0x00000036
            )))
        }
    }

    func testE2CReparsesLinkeditAndPrimaryCodeDirectory() throws {
        let primary = Self.codeDirectory(
            hashType: 2,
            flags: 0,
            signingIdentifier: Data("com.example.e2c".utf8),
            teamIdentifier: Data("TEAM123456".utf8)
        )
        let signature = Self.superBlob(entries: [
            (0, primary),
            (
                0x10000,
                Self.genericBlob(
                    magic: Self.csMagicBlobWrapper,
                    payload: Data([0x30, 0x00])
                )
            ),
        ])
        let signatureStart: UInt64 = 65_536
        let localSignatureOffset = signatureStart - Self.imageFileOffset
        let commands = Self.signedLoadCommands(
            localSignatureOffset: localSignatureOffset,
            signatureBytes: UInt64(signature.count)
        )
        let fixture = try Self.fixture(
            suffixes: [],
            loadCommands: commands,
            commandCount: 4,
            primaryCodeDirectory: .present(
                blobSHA256: Self.sha256(primary)
            ),
            signatureStart: signatureStart,
            signature: signature
        )

        let result = try Self.compare(fixture)
        XCTAssertEqual(
            result.sharedCacheImageEvidence[0]
                .primaryCodeDirectoryBlobSHA256,
            Self.sha256(primary)
        )
        XCTAssertEqual(
            result.sharedCacheImageEvidence[0].facts.primaryCodeDirectory,
            .present(blobSHA256: Self.sha256(primary))
        )
        Self.assertNoAuthority(result)
    }

    func testE2CRejectsPredecessorRoleBeforeCompleteTranscriptShape()
        throws
    {
        let fixture = try Self.fixture(suffixes: [])
        Self.assertFailure(
            try SyntheticSharedCacheCompleteSetVerifier.compare(
                setPlan: fixture.setPlan,
                gitExpectation: fixture.selfGuardExpectation,
                selfGuardExpectation: fixture.gitExpectation,
                discoveryTranscripts: fixture.discoveryTranscripts,
                planProbeTranscripts: fixture.planProbeTranscripts,
                completeFileTranscripts: []
            )
        ) {
            $0 == .predecessor(.anchoredMemberSet(.role(
                expected: .git,
                actual: .selfGuard
            )))
        }
    }

    func testE2CResourceConstantsPinExactArithmeticEnvelope() {
        XCTAssertEqual(
            SyntheticSharedCacheCompleteSetVerifier.maximumFileCount,
            64
        )
        XCTAssertEqual(
            SyntheticSharedCacheCompleteSetVerifier.maximumSubcacheCount,
            63
        )
        XCTAssertEqual(
            SyntheticSharedCacheCompleteSetVerifier.maximumFileBytes,
            17_179_869_184
        )
        XCTAssertEqual(
            SyntheticSharedCacheCompleteSetVerifier.maximumSetBytes,
            68_719_476_736
        )
        XCTAssertEqual(
            SyntheticSharedCacheCompleteSetVerifier.logicalChunkBytes,
            1_048_576
        )
        XCTAssertEqual(
            SyntheticSharedCacheCompleteSetVerifier
                .maximumPositiveFragmentsPerChunk,
            16
        )
        XCTAssertEqual(
            SyntheticSharedCacheCompleteSetVerifier
                .maximumInterruptedRetriesPerOffset,
            8
        )
        XCTAssertEqual(
            SyntheticSharedCacheCompleteSetVerifier.maximumPerFileChunkCount,
            16_384
        )
        XCTAssertEqual(
            SyntheticSharedCacheCompleteSetVerifier.maximumPerFileAttempts,
            2_359_297
        )
        XCTAssertEqual(
            SyntheticSharedCacheCompleteSetVerifier.maximumSetChunkCount,
            65_599
        )
        XCTAssertEqual(
            SyntheticSharedCacheCompleteSetVerifier.maximumSetAttempts,
            9_446_320
        )
        XCTAssertEqual(
            SyntheticSharedCacheCompleteSetVerifier
                .maximumFragmentSnapshotBytes,
            1_048_576
        )
        XCTAssertEqual(
            SyntheticSharedCacheCompleteSetVerifier
                .maximumCompactEvidenceSemanticBytes,
            16_777_216
        )
        XCTAssertEqual(
            SyntheticSharedCacheCompleteSetVerifier
                .maximumVerifierOwnedContentBytes,
            537_919_488
        )

        let chunksAtFileCeiling =
            SyntheticSharedCacheCompleteSetVerifier.maximumFileBytes /
            SyntheticSharedCacheCompleteSetVerifier.logicalChunkBytes
        XCTAssertEqual(chunksAtFileCeiling, 16_384)
        XCTAssertEqual(
            chunksAtFileCeiling * 16 * 9 + 1,
            SyntheticSharedCacheCompleteSetVerifier.maximumPerFileAttempts
        )
        let chunksOneByteOverFileCeiling =
            (SyntheticSharedCacheCompleteSetVerifier.maximumFileBytes + 1 +
             SyntheticSharedCacheCompleteSetVerifier.logicalChunkBytes - 1) /
            SyntheticSharedCacheCompleteSetVerifier.logicalChunkBytes
        XCTAssertEqual(chunksOneByteOverFileCeiling, 16_385)
        XCTAssertGreaterThan(
            chunksOneByteOverFileCeiling,
            SyntheticSharedCacheCompleteSetVerifier
                .maximumPerFileChunkCount
        )

        let maximumSetChunks =
            SyntheticSharedCacheCompleteSetVerifier.maximumSetBytes /
            SyntheticSharedCacheCompleteSetVerifier.logicalChunkBytes + 63
        XCTAssertEqual(
            maximumSetChunks,
            SyntheticSharedCacheCompleteSetVerifier.maximumSetChunkCount
        )
        XCTAssertEqual(
            maximumSetChunks * 16 * 9 + 64,
            SyntheticSharedCacheCompleteSetVerifier.maximumSetAttempts
        )
        XCTAssertGreaterThan(
            SyntheticSharedCacheCompleteSetVerifier.maximumSetBytes + 1,
            SyntheticSharedCacheCompleteSetVerifier.maximumSetBytes
        )
    }

    func testE2CRejectsTranscriptShapeAndMetadataDrift() throws {
        let fixture = try Self.fixture()

        Self.assertFailure(
            try Self.compare(fixture, transcripts: [])
        ) {
            $0 == .completeStream(.transcriptCount(
                expected: 3,
                actual: 0
            ))
        }

        var ordinalDrift = fixture.completeFileTranscripts
        ordinalDrift[0] = Self.replacing(
            ordinalDrift[0],
            fileOrdinal: 2
        )
        Self.assertFailure(
            try Self.compare(fixture, transcripts: ordinalDrift)
        ) {
            $0 == .completeStream(.fileOrdinal(
                transcriptOrdinal: 0,
                expected: 0,
                actual: 2
            ))
        }

        var beforeDrift = fixture.completeFileTranscripts
        var metadata = beforeDrift[0].beforeMetadata
        metadata.inode += 1
        beforeDrift[0] = Self.replacing(
            beforeDrift[0],
            beforeMetadata: metadata
        )
        Self.assertFailure(
            try Self.compare(fixture, transcripts: beforeDrift)
        ) {
            $0 == .metadataDrift(
                fileOrdinal: 0,
                position: .before,
                field: .inode
            )
        }

        var afterDrift = fixture.completeFileTranscripts
        metadata = afterDrift[0].afterMetadata
        metadata.statusChangeTimeNanoseconds += 1
        afterDrift[0] = Self.replacing(
            afterDrift[0],
            afterMetadata: metadata
        )
        Self.assertFailure(
            try Self.compare(fixture, transcripts: afterDrift)
        ) {
            $0 == .metadataDrift(
                fileOrdinal: 0,
                position: .after,
                field: .statusChangeTimeNanoseconds
            )
        }
    }

    func testE2CEnforcesInterruptionAndFragmentLimits() throws {
        let fixture = try Self.fixture()
        let eightInterruptions = Array(
            repeating: SyntheticCaptureReadEvent.interrupted(offset: 0),
            count: 8
        )
        var transcripts = fixture.completeFileTranscripts
        transcripts[0] = Self.replacing(
            transcripts[0],
            events: eightInterruptions + transcripts[0].events
        )
        let accepted = try Self.compare(fixture, transcripts: transcripts)
        XCTAssertEqual(accepted.resourceCounts.streamAttemptCount, 17)
        Self.assertNoAuthority(accepted)

        var tooManyInterruptions = fixture.completeFileTranscripts
        tooManyInterruptions[0] = Self.replacing(
            tooManyInterruptions[0],
            events: Array(
                repeating: SyntheticCaptureReadEvent.interrupted(offset: 0),
                count: 9
            ) + fixture.completeFileTranscripts[0].events
        )
        Self.assertFailure(
            try Self.compare(fixture, transcripts: tooManyInterruptions)
        ) {
            $0 == .readInterruptedLimit(fileOrdinal: 0, eventOrdinal: 8)
        }

        var sixteenFragments = fixture.completeFileTranscripts
        sixteenFragments[0] = Self.replacing(
            sixteenFragments[0],
            events: Self.fragmentedFirstChunkEvents(
                bytes: fixture.completeFileBytes[0],
                fragmentCount: 16
            )
        )
        let fragmented = try Self.compare(
            fixture,
            transcripts: sixteenFragments
        )
        XCTAssertEqual(fragmented.resourceCounts.streamAttemptCount, 24)

        var tooManyFragments = fixture.completeFileTranscripts
        tooManyFragments[0] = Self.replacing(
            tooManyFragments[0],
            events: Self.fragmentedFirstChunkEvents(
                bytes: fixture.completeFileBytes[0],
                fragmentCount: 17
            )
        )
        Self.assertFailure(
            try Self.compare(fixture, transcripts: tooManyFragments)
        ) {
            $0 == .readFragmentLimit(fileOrdinal: 0, eventOrdinal: 16)
        }
    }

    func testE2CEnforcesOffsetsEOFAndReadErrors() throws {
        let fixture = try Self.fixture()

        var gap = fixture.completeFileTranscripts
        gap[0] = Self.replacing(
            gap[0],
            events: [
                .bytes(offset: 1, data: Data([0x01])),
            ] + Array(gap[0].events.dropFirst())
        )
        Self.assertFailure(try Self.compare(fixture, transcripts: gap)) {
            $0 == .completeStream(.offsetGap(
                fileOrdinal: 0,
                eventOrdinal: 0,
                expected: 0,
                actual: 1
            ))
        }

        var overlap = fixture.completeFileTranscripts
        overlap[0] = Self.replacing(
            overlap[0],
            events: [
                overlap[0].events[0],
                overlap[0].events[0],
                overlap[0].events[2],
            ]
        )
        Self.assertFailure(try Self.compare(fixture, transcripts: overlap)) {
            $0 == .completeStream(.offsetOverlapOrReorder(
                fileOrdinal: 0,
                eventOrdinal: 1,
                expected: 1_048_576,
                actual: 0
            ))
        }

        var earlyEOF = fixture.completeFileTranscripts
        earlyEOF[0] = Self.replacing(
            earlyEOF[0],
            events: [.endOfFile(offset: 0)]
        )
        Self.assertFailure(
            try Self.compare(fixture, transcripts: earlyEOF)
        ) {
            $0 == .completeStream(.earlyEOF(
                fileOrdinal: 0,
                eventOrdinal: 0,
                expected: UInt64(Self.fileBytes),
                actual: 0
            ))
        }

        var missingEOF = fixture.completeFileTranscripts
        missingEOF[0] = Self.replacing(
            missingEOF[0],
            events: Array(missingEOF[0].events.dropLast())
        )
        Self.assertFailure(
            try Self.compare(fixture, transcripts: missingEOF)
        ) {
            $0 == .completeStream(.missingEOF(
                fileOrdinal: 0,
                offset: UInt64(Self.fileBytes)
            ))
        }

        var trailing = fixture.completeFileTranscripts
        trailing[0] = Self.replacing(
            trailing[0],
            events: trailing[0].events + [
                .endOfFile(offset: UInt64(Self.fileBytes)),
            ]
        )
        Self.assertFailure(try Self.compare(fixture, transcripts: trailing)) {
            $0 == .unexpectedTrailingByte(fileOrdinal: 0, eventOrdinal: 3)
        }

        var readError = fixture.completeFileTranscripts
        readError[0] = Self.replacing(
            readError[0],
            events: [.error(offset: 0, code: 5)]
        )
        Self.assertFailure(
            try Self.compare(fixture, transcripts: readError)
        ) {
            $0 == .readError(fileOrdinal: 0, eventOrdinal: 0, code: 5)
        }
    }

    func testE2CSeparatesRetainedBytesFromFullSHAValidation() throws {
        let fixture = try Self.fixture()

        var retainedDrift = fixture.completeFileBytes[0]
        retainedDrift[0] ^= 0xff
        var transcripts = fixture.completeFileTranscripts
        transcripts[0] = Self.completeTranscript(
            fileOrdinal: 0,
            metadata: fixture.completeFileTranscripts[0].beforeMetadata,
            bytes: retainedDrift
        )
        Self.assertFailure(
            try Self.compare(fixture, transcripts: transcripts)
        ) {
            $0 == .completeStream(.retainedByteMismatch(
                fileOrdinal: 0,
                offset: 0
            ))
        }

        var unretainedDrift = fixture.completeFileBytes[1]
        unretainedDrift[Self.fileBytes - 1] ^= 0xff
        transcripts = fixture.completeFileTranscripts
        transcripts[1] = Self.completeTranscript(
            fileOrdinal: 1,
            metadata: fixture.completeFileTranscripts[1].beforeMetadata,
            bytes: unretainedDrift
        )
        Self.assertFailure(
            try Self.compare(fixture, transcripts: transcripts)
        ) {
            guard case let .expectedSHA256(
                fileOrdinal,
                .mismatch(expected, actual)
            ) = $0 else {
                return false
            }
            return fileOrdinal == 1 &&
                expected == fixture.cacheSetEvidence.records[2].fileSHA256 &&
                actual != expected
        }
    }

    func testE2CEnforcesChunkAndFinalByteEdgesAndCallerIndependence()
        throws
    {
        let fixture = try Self.fixture(suffixes: [])
        let complete = fixture.completeFileBytes[0]
        let metadata = fixture.completeFileTranscripts[0].beforeMetadata

        var callerFragment = Data(complete[0..<512])
        let copiedEvent = SyntheticCaptureReadEvent.bytes(
            offset: 0,
            data: callerFragment
        )
        callerFragment[0] ^= 0xff
        let shortThenComplete = SyntheticSharedCacheCompleteFileTranscript(
            fileOrdinal: 0,
            beforeMetadata: metadata,
            afterMetadata: metadata,
            events: [
                copiedEvent,
                .bytes(
                    offset: 512,
                    data: Data(complete[512..<1_048_576])
                ),
                .bytes(
                    offset: 1_048_576,
                    data: Data(complete[1_048_576..<Self.fileBytes])
                ),
                .endOfFile(offset: UInt64(Self.fileBytes)),
            ]
        )
        var mutableInputs = [shortThenComplete]
        let independent = try Self.compare(
            fixture,
            transcripts: mutableInputs
        )
        let stableResult = independent
        mutableInputs.removeAll()
        XCTAssertEqual(independent, stableResult)
        XCTAssertEqual(independent.resourceCounts.streamAttemptCount, 4)

        let almostChunk = Data(complete[0..<(1_048_576 - 1)])
        var transcripts = [Self.replacing(
            fixture.completeFileTranscripts[0],
            events: [
                .bytes(offset: 0, data: almostChunk),
                .bytes(
                    offset: 1_048_575,
                    data: Data(complete[1_048_575...1_048_576])
                ),
            ]
        )]
        Self.assertFailure(try Self.compare(fixture, transcripts: transcripts)) {
            $0 == .completeStream(.fragmentBeyondChunk(
                fileOrdinal: 0,
                eventOrdinal: 1,
                chunkEnd: 1_048_576
            ))
        }

        transcripts = [Self.replacing(
            fixture.completeFileTranscripts[0],
            events: [
                fixture.completeFileTranscripts[0].events[0],
                fixture.completeFileTranscripts[0].events[1],
                .bytes(
                    offset: UInt64(Self.fileBytes),
                    data: Data([0])
                ),
            ]
        )]
        Self.assertFailure(try Self.compare(fixture, transcripts: transcripts)) {
            $0 == .completeStream(.fragmentPastMetadata(
                fileOrdinal: 0,
                eventOrdinal: 2,
                fileBytes: UInt64(Self.fileBytes)
            ))
        }

        transcripts = [Self.replacing(
            fixture.completeFileTranscripts[0],
            events: [
                fixture.completeFileTranscripts[0].events[0],
                fixture.completeFileTranscripts[0].events[1],
                .interrupted(offset: UInt64(Self.fileBytes)),
                .endOfFile(offset: UInt64(Self.fileBytes)),
            ]
        )]
        Self.assertFailure(try Self.compare(fixture, transcripts: transcripts)) {
            $0 == .completeStream(.interruptionAfterFinalBytes(
                fileOrdinal: 0,
                eventOrdinal: 2
            ))
        }

        transcripts = [Self.replacing(
            fixture.completeFileTranscripts[0],
            events: [.bytes(offset: 0, data: Data())]
        )]
        Self.assertFailure(try Self.compare(fixture, transcripts: transcripts)) {
            $0 == .completeStream(.fragmentSize(
                fileOrdinal: 0,
                eventOrdinal: 0,
                bytes: 0
            ))
        }
    }
}

private extension SyntheticSharedCacheCompleteSetTests {
    struct Fixture {
        let setPlan: SyntheticSharedCacheSetPlanComparison
        let gitExpectation: AnchoredRuntimeClosureExpectationDocument
        let selfGuardExpectation: AnchoredRuntimeClosureExpectationDocument
        let discoveryTranscripts: [SyntheticSharedCacheBoundedReadTranscript]
        let planProbeTranscripts: [SyntheticSharedCacheBoundedReadTranscript]
        let completeFileTranscripts:
            [SyntheticSharedCacheCompleteFileTranscript]
        let completeFileBytes: [Data]
        let cacheSetEvidence: SyntheticSharedCacheSetIdentityEvidence
        let imageEvidence:
            SyntheticSharedCacheImageContentIdentityEvidence
    }

    static let headerBytes = 552
    static let mappingWithSlideOffset = 608
    static let mappingTablesEnd = 664
    static let modernSubcacheEntryBytes = 56
    static let mainSharedRegionStart: UInt64 = 0x0000_0001_8000_0000
    static let mappingBytes: UInt64 = 1_048_576
    static let fileBytes = 2 * 1_048_576
    static let codeSignatureBytes: UInt64 = 4_096
    static let nameWindowOffset: UInt64 = 4_096
    static let imageFileOffset: UInt64 = 12_288
    static let imageVMAddress = mainSharedRegionStart + imageFileOffset
    static let imageName = Data("/usr/lib/libFixture.dylib".utf8)
    static let rootID = String(repeating: "1", count: 64)
    static let loaderID = String(repeating: "2", count: 64)
    static let suffixes = [Data(".2".utf8), Data(".1".utf8)]
    static let cacheVMOffsets: [UInt64] = [0x0020_0000, 0x0040_0000]
    static let mainUUID = uuid(byte: 0x41)
    static let subcacheUUIDs = [uuid(byte: 0x52), uuid(byte: 0x51)]
    static let imageUUID = uuid(byte: 0x71)
    static let csMagicEmbeddedSignature: UInt32 = 0xfade0cc0
    static let csMagicCodeDirectory: UInt32 = 0xfade0c02
    static let csMagicBlobWrapper: UInt32 = 0xfade0b01

    static func fixture(
        suffixes requestedSuffixes: [Data] = suffixes,
        loadCommands requestedLoadCommands: Data? = nil,
        commandCount requestedCommandCount: UInt32 = 2,
        primaryCodeDirectory:
            SyntheticSharedCacheImagePrimaryCodeDirectory = .absent,
        signatureStart: UInt64? = nil,
        signature: Data? = nil
    ) throws -> Fixture {
        precondition(requestedSuffixes.count <= subcacheUUIDs.count)
        precondition((signatureStart == nil) == (signature == nil))
        let requestedUUIDs = Array(
            subcacheUUIDs.prefix(requestedSuffixes.count)
        )
        let requestedOffsets = Array(
            cacheVMOffsets.prefix(requestedSuffixes.count)
        )
        let metadata = (0...requestedSuffixes.count).map(fileMetadata)
        let loadCommands = requestedLoadCommands ?? fixtureLoadCommands()
        let sharedRegionSize = UInt64(requestedSuffixes.count + 1) *
            UInt64(fileBytes)
        let mainDiscovery = mainDiscoveryBytes(
            suffixes: requestedSuffixes,
            subcacheUUIDs: requestedUUIDs,
            cacheVMOffsets: requestedOffsets,
            sharedRegionSize: sharedRegionSize
        )
        let subcacheDiscoveries = requestedSuffixes.indices.map { index in
            subcacheDiscoveryBytes(
                uuid: requestedUUIDs[index],
                cacheVMOffset: requestedOffsets[index],
                sharedRegionSize: sharedRegionSize
            )
        }
        let discoveryBytes = [mainDiscovery] + subcacheDiscoveries

        var completeBytes = discoveryBytes.map { discovery -> Data in
            var bytes = Data(repeating: 0, count: fileBytes)
            bytes.replaceSubrange(0..<discovery.count, with: discovery)
            return bytes
        }
        var nameWindow = Data(repeating: 0, count: 4_097)
        nameWindow.replaceSubrange(0..<imageName.count, with: imageName)
        completeBytes[0].replaceSubrange(
            Int(nameWindowOffset)..<Int(nameWindowOffset) + nameWindow.count,
            with: nameWindow
        )
        let machHeader = fixtureMachHeader(
            commandCount: requestedCommandCount,
            loadCommandBytes: loadCommands.count
        )
        completeBytes[0].replaceSubrange(
            Int(imageFileOffset)..<Int(imageFileOffset) + machHeader.count,
            with: machHeader
        )
        let loadCommandStart = Int(imageFileOffset + 32)
        let loadCommandEnd = loadCommandStart + loadCommands.count
        completeBytes[0].replaceSubrange(
            loadCommandStart..<loadCommandEnd,
            with: loadCommands
        )
        if let signatureStart, let signature {
            let start = Int(signatureStart)
            completeBytes[0].replaceSubrange(
                start..<(start + signature.count),
                with: signature
            )
        }

        let headerOrderSHA256 = completeBytes.map(sha256)
        let mainRecord = cacheRecord(
            suffix: Data(),
            sha256: headerOrderSHA256[0],
            uuid: mainUUID
        )
        let subcacheRecords = requestedSuffixes.indices.map { index in
            (
                suffix: requestedSuffixes[index],
                record: cacheRecord(
                    suffix: requestedSuffixes[index],
                    sha256: headerOrderSHA256[index + 1],
                    uuid: requestedUUIDs[index]
                )
            )
        }.sorted {
            $0.suffix.lexicographicallyPrecedes($1.suffix)
        }.map(\.record)
        let cacheSet = try SyntheticSharedCacheSetIdentityVerifier.derive(
            records: [mainRecord] + subcacheRecords
        )
        let imageEvidence = try
            SyntheticSharedCacheImageContentIdentityVerifier.derive(
                cacheSetEvidence: cacheSet,
                facts: SyntheticSharedCacheImageContentFacts(
                    installNameBytes: UInt64(imageName.count),
                    installNameBase64URL: base64URL(imageName),
                    machOUUID: hex(imageUUID),
                    primaryCodeDirectory: primaryCodeDirectory,
                    loadCommandsSHA256: sha256(loadCommands)
                )
            )
        let member = RuntimeClosureExpectationMemberFields(
            contentEvidenceID: imageEvidence.contentEvidenceID.sha256,
            storage: .sharedCache,
            installName: SyntheticRuntimeClosureInstallName(
                bytes: UInt64(imageName.count),
                base64URL: base64URL(imageName)
            ),
            decodedInstallName: imageName,
            machOUUID: hex(imageUUID),
            primaryCodeDirectoryBlobSHA256:
                imageEvidence.primaryCodeDirectoryBlobSHA256,
            loadCommandsSHA256: sha256(loadCommands)
        )
        let memberEdge = RuntimeClosureExpectationEdgeFields(
            parentContentEvidenceID: member.contentEvidenceID,
            loadCommandOrdinal: 0,
            kind: .load,
            installName: member.installName,
            decodedInstallName: member.decodedInstallName,
            resolvedContentEvidenceID: member.contentEvidenceID
        )

        let setPlan = try SyntheticSharedCacheSetPlanVerifier.compare(
            sourceProfile: .reviewed,
            main: SyntheticSharedCacheDiscoveryFile(
                metadata: metadata[0],
                discoveryBytes: discoveryBytes[0]
            ),
            subcaches: requestedSuffixes.indices.map { index in
                SyntheticSharedCacheDiscoveryFile(
                    metadata: metadata[index + 1],
                    discoveryBytes: discoveryBytes[index + 1]
                )
            }
        )
        let git = try anchoredExpectation(
            role: .git,
            cacheSet: cacheSet,
            member: member,
            edge: memberEdge
        )
        let selfGuard = try anchoredExpectation(
            role: .selfGuard,
            cacheSet: cacheSet,
            member: member,
            edge: memberEdge
        )
        let discoveryTranscripts = discoveryBytes.indices.map { ordinal in
            boundedTranscript(
                purpose: .setHeaderDiscovery,
                fileOrdinal: ordinal,
                start: 0,
                metadata: metadata[ordinal],
                bytes: discoveryBytes[ordinal]
            )
        }
        var probeTranscripts = [
            boundedTranscript(
                purpose: .planProbe(
                    consumerKind: .imageNameWindow,
                    consumerOrdinal: 0
                ),
                fileOrdinal: 0,
                start: nameWindowOffset,
                metadata: metadata[0],
                bytes: nameWindow
            ),
            boundedTranscript(
                purpose: .planProbe(
                    consumerKind: .machOHeader,
                    consumerOrdinal: 0
                ),
                fileOrdinal: 0,
                start: imageFileOffset,
                metadata: metadata[0],
                bytes: machHeader
            ),
            boundedTranscript(
                purpose: .planProbe(
                    consumerKind: .loadCommands,
                    consumerOrdinal: 0
                ),
                fileOrdinal: 0,
                start: imageFileOffset + 32,
                metadata: metadata[0],
                bytes: loadCommands
            ),
        ]
        if let signatureStart, let signature {
            probeTranscripts.append(boundedTranscript(
                purpose: .planProbe(
                    consumerKind: .codeSignature,
                    consumerOrdinal: 0
                ),
                fileOrdinal: 0,
                start: signatureStart,
                metadata: metadata[0],
                bytes: signature
            ))
        }
        let completeTranscripts = completeBytes.indices.map { ordinal in
            completeTranscript(
                fileOrdinal: ordinal,
                metadata: metadata[ordinal],
                bytes: completeBytes[ordinal]
            )
        }

        return Fixture(
            setPlan: setPlan,
            gitExpectation: git,
            selfGuardExpectation: selfGuard,
            discoveryTranscripts: discoveryTranscripts,
            planProbeTranscripts: probeTranscripts,
            completeFileTranscripts: completeTranscripts,
            completeFileBytes: completeBytes,
            cacheSetEvidence: cacheSet,
            imageEvidence: imageEvidence
        )
    }

    static func compare(
        _ fixture: Fixture
    ) throws -> SyntheticSharedCacheCompleteSetComparison {
        try compare(fixture, transcripts: fixture.completeFileTranscripts)
    }

    static func compare(
        _ fixture: Fixture,
        transcripts: [SyntheticSharedCacheCompleteFileTranscript]
    ) throws -> SyntheticSharedCacheCompleteSetComparison {
        try SyntheticSharedCacheCompleteSetVerifier.compare(
            setPlan: fixture.setPlan,
            gitExpectation: fixture.gitExpectation,
            selfGuardExpectation: fixture.selfGuardExpectation,
            discoveryTranscripts: fixture.discoveryTranscripts,
            planProbeTranscripts: fixture.planProbeTranscripts,
            completeFileTranscripts: transcripts
        )
    }

    static func completeTranscript(
        fileOrdinal: Int,
        metadata: SyntheticCaptureFileMetadata,
        bytes: Data
    ) -> SyntheticSharedCacheCompleteFileTranscript {
        precondition(bytes.count == fileBytes)
        let first = Data(bytes[0..<1_048_576])
        let second = Data(bytes[1_048_576..<2_097_152])
        return SyntheticSharedCacheCompleteFileTranscript(
            fileOrdinal: fileOrdinal,
            beforeMetadata: metadata,
            afterMetadata: metadata,
            events: [
                .bytes(offset: 0, data: first),
                .bytes(offset: 1_048_576, data: second),
                .endOfFile(offset: UInt64(bytes.count)),
            ]
        )
    }

    static func replacing(
        _ transcript: SyntheticSharedCacheCompleteFileTranscript,
        fileOrdinal: Int? = nil,
        beforeMetadata: SyntheticCaptureFileMetadata? = nil,
        afterMetadata: SyntheticCaptureFileMetadata? = nil,
        events: [SyntheticCaptureReadEvent]? = nil
    ) -> SyntheticSharedCacheCompleteFileTranscript {
        SyntheticSharedCacheCompleteFileTranscript(
            fileOrdinal: fileOrdinal ?? transcript.fileOrdinal,
            beforeMetadata: beforeMetadata ?? transcript.beforeMetadata,
            afterMetadata: afterMetadata ?? transcript.afterMetadata,
            events: events ?? transcript.events
        )
    }

    static func fragmentedFirstChunkEvents(
        bytes: Data,
        fragmentCount: Int
    ) -> [SyntheticCaptureReadEvent] {
        precondition(fragmentCount >= 1)
        var events: [SyntheticCaptureReadEvent] = []
        var cursor = 0
        for index in 0..<fragmentCount {
            let remainingFragments = fragmentCount - index
            let remainingBytes = 1_048_576 - cursor
            let length = index == fragmentCount - 1 ?
                remainingBytes :
                max(1, remainingBytes / remainingFragments)
            let end = cursor + length
            events.append(.bytes(
                offset: UInt64(cursor),
                data: Data(bytes[cursor..<end])
            ))
            cursor = end
        }
        events.append(.bytes(
            offset: 1_048_576,
            data: Data(bytes[1_048_576..<2_097_152])
        ))
        events.append(.endOfFile(offset: UInt64(bytes.count)))
        return events
    }

    static func boundedTranscript(
        purpose: SyntheticSharedCacheBoundedReadTranscript.Purpose,
        fileOrdinal: Int,
        start: UInt64,
        metadata: SyntheticCaptureFileMetadata,
        bytes: Data
    ) -> SyntheticSharedCacheBoundedReadTranscript {
        SyntheticSharedCacheBoundedReadTranscript(
            purpose: purpose,
            fileOrdinal: fileOrdinal,
            rangeStart: start,
            requestedByteCount: UInt64(bytes.count),
            beforeMetadata: metadata,
            afterMetadata: metadata,
            events: [.bytes(offset: start, data: bytes)]
        )
    }

    static func mainDiscoveryBytes(
        suffixes: [Data],
        subcacheUUIDs: [Data],
        cacheVMOffsets: [UInt64],
        sharedRegionSize: UInt64
    ) -> Data {
        let imageTableOffset = mappingTablesEnd
        let imageTextOffset = imageTableOffset + 32
        let subcacheTableOffset = imageTextOffset + 32
        var bytes = Data(
            repeating: 0,
            count: subcacheTableOffset +
                suffixes.count * modernSubcacheEntryBytes
        )
        writeCommonHeader(
            into: &bytes,
            headerUUID: mainUUID,
            sharedRegionStart: mainSharedRegionStart,
            sharedRegionSize: sharedRegionSize
        )
        writeUInt32LE(UInt32(imageTableOffset), into: &bytes, at: 448)
        writeUInt32LE(1, into: &bytes, at: 452)
        writeUInt64LE(UInt64(imageTextOffset), into: &bytes, at: 136)
        writeUInt64LE(1, into: &bytes, at: 144)
        writeUInt32LE(UInt32(subcacheTableOffset), into: &bytes, at: 392)
        writeUInt32LE(UInt32(suffixes.count), into: &bytes, at: 396)
        for index in suffixes.indices {
            let offset = subcacheTableOffset +
                index * modernSubcacheEntryBytes
            bytes.replaceSubrange(
                offset..<(offset + 16),
                with: subcacheUUIDs[index]
            )
            writeUInt64LE(cacheVMOffsets[index], into: &bytes, at: offset + 16)
            bytes.replaceSubrange(
                (offset + 24)..<(offset + 24 + suffixes[index].count),
                with: suffixes[index]
            )
        }
        writeUInt64LE(imageVMAddress, into: &bytes, at: imageTableOffset)
        writeUInt64LE(0, into: &bytes, at: imageTableOffset + 8)
        writeUInt64LE(0, into: &bytes, at: imageTableOffset + 16)
        writeUInt32LE(
            UInt32(nameWindowOffset),
            into: &bytes,
            at: imageTableOffset + 24
        )
        writeUInt32LE(0, into: &bytes, at: imageTableOffset + 28)
        return bytes
    }

    static func subcacheDiscoveryBytes(
        uuid: Data,
        cacheVMOffset: UInt64,
        sharedRegionSize: UInt64
    ) -> Data {
        var bytes = Data(repeating: 0, count: mappingTablesEnd)
        writeCommonHeader(
            into: &bytes,
            headerUUID: uuid,
            sharedRegionStart: mainSharedRegionStart + cacheVMOffset,
            sharedRegionSize: sharedRegionSize
        )
        return bytes
    }

    static func writeCommonHeader(
        into bytes: inout Data,
        headerUUID: Data,
        sharedRegionStart: UInt64,
        sharedRegionSize: UInt64
    ) {
        bytes.replaceSubrange(
            0..<16,
            with: Data([
                0x64, 0x79, 0x6c, 0x64, 0x5f, 0x76, 0x31, 0x20,
                0x20, 0x20, 0x61, 0x72, 0x6d, 0x36, 0x34, 0x00,
            ])
        )
        writeUInt32LE(UInt32(headerBytes), into: &bytes, at: 16)
        writeUInt32LE(1, into: &bytes, at: 20)
        writeUInt64LE(
            UInt64(fileBytes) - codeSignatureBytes,
            into: &bytes,
            at: 40
        )
        writeUInt64LE(codeSignatureBytes, into: &bytes, at: 48)
        bytes.replaceSubrange(88..<104, with: headerUUID)
        writeUInt64LE(2, into: &bytes, at: 104)
        writeUInt32LE(1, into: &bytes, at: 216)
        writeUInt32LE(1 << 12, into: &bytes, at: 220)
        writeUInt64LE(sharedRegionStart, into: &bytes, at: 224)
        writeUInt64LE(sharedRegionSize, into: &bytes, at: 232)
        writeUInt32LE(
            UInt32(mappingWithSlideOffset),
            into: &bytes,
            at: 312
        )
        writeUInt32LE(1, into: &bytes, at: 316)
        writeUInt32LE(1, into: &bytes, at: 456)

        writeUInt64LE(sharedRegionStart, into: &bytes, at: headerBytes)
        writeUInt64LE(mappingBytes, into: &bytes, at: headerBytes + 8)
        writeUInt64LE(0, into: &bytes, at: headerBytes + 16)
        writeUInt32LE(5, into: &bytes, at: headerBytes + 24)
        writeUInt32LE(5, into: &bytes, at: headerBytes + 28)

        writeUInt64LE(
            sharedRegionStart,
            into: &bytes,
            at: mappingWithSlideOffset
        )
        writeUInt64LE(
            mappingBytes,
            into: &bytes,
            at: mappingWithSlideOffset + 8
        )
        writeUInt64LE(0, into: &bytes, at: mappingWithSlideOffset + 16)
        writeUInt64LE(0, into: &bytes, at: mappingWithSlideOffset + 24)
        writeUInt64LE(0, into: &bytes, at: mappingWithSlideOffset + 32)
        writeUInt64LE(0, into: &bytes, at: mappingWithSlideOffset + 40)
        writeUInt32LE(5, into: &bytes, at: mappingWithSlideOffset + 48)
        writeUInt32LE(5, into: &bytes, at: mappingWithSlideOffset + 52)
    }

    static func fixtureMachHeader(
        commandCount: UInt32,
        loadCommandBytes: Int
    ) -> Data {
        var bytes = Data(repeating: 0, count: 32)
        writeUInt32LE(0xfeedfacf, into: &bytes, at: 0)
        writeUInt32LE(0x0100000c, into: &bytes, at: 4)
        writeUInt32LE(0, into: &bytes, at: 8)
        writeUInt32LE(0x6, into: &bytes, at: 12)
        writeUInt32LE(commandCount, into: &bytes, at: 16)
        writeUInt32LE(UInt32(loadCommandBytes), into: &bytes, at: 20)
        writeUInt32LE(0, into: &bytes, at: 24)
        writeUInt32LE(0, into: &bytes, at: 28)
        return bytes
    }

    static func fixtureLoadCommands() -> Data {
        var bytes = Data()
        appendUInt32LE(&bytes, 0x1b)
        appendUInt32LE(&bytes, 24)
        bytes.append(imageUUID)
        var identifier = Data(repeating: 0, count: 32)
        writeUInt32LE(0x0d, into: &identifier, at: 0)
        writeUInt32LE(32, into: &identifier, at: 4)
        writeUInt32LE(24, into: &identifier, at: 8)
        writeUInt32LE(0, into: &identifier, at: 12)
        writeUInt32LE(1, into: &identifier, at: 16)
        writeUInt32LE(1, into: &identifier, at: 20)
        identifier[24] = 0x78
        bytes.append(identifier)
        return bytes
    }

    static func passthroughCommand(_ command: UInt32, size: Int) -> Data {
        var bytes = Data(repeating: 0, count: size)
        writeUInt32LE(command, into: &bytes, at: 0)
        writeUInt32LE(UInt32(size), into: &bytes, at: 4)
        return bytes
    }

    static func nonLinkeditSegmentCommand() -> Data {
        var segment = Data(repeating: 0, count: 72)
        writeUInt32LE(0x19, into: &segment, at: 0)
        writeUInt32LE(72, into: &segment, at: 4)
        segment.replaceSubrange(8..<14, with: Data("__TEXT".utf8))
        return segment
    }

    static func signedLoadCommands(
        localSignatureOffset: UInt64,
        signatureBytes: UInt64
    ) -> Data {
        var bytes = Data()
        appendUInt32LE(&bytes, 0x1b)
        appendUInt32LE(&bytes, 24)
        bytes.append(imageUUID)

        var segment = Data(repeating: 0, count: 72)
        writeUInt32LE(0x19, into: &segment, at: 0)
        writeUInt32LE(72, into: &segment, at: 4)
        segment.replaceSubrange(8..<18, with: Data("__LINKEDIT".utf8))
        writeUInt64LE(
            imageVMAddress + localSignatureOffset,
            into: &segment,
            at: 24
        )
        writeUInt64LE(signatureBytes, into: &segment, at: 32)
        writeUInt64LE(localSignatureOffset, into: &segment, at: 40)
        writeUInt64LE(signatureBytes, into: &segment, at: 48)
        writeUInt32LE(1, into: &segment, at: 56)
        writeUInt32LE(1, into: &segment, at: 60)
        bytes.append(segment)
        bytes.append(contentsOf: fixtureLoadCommands().dropFirst(24))
        appendUInt32LE(&bytes, 0x1d)
        appendUInt32LE(&bytes, 16)
        appendUInt32LE(&bytes, UInt32(localSignatureOffset))
        appendUInt32LE(&bytes, UInt32(signatureBytes))
        return bytes
    }

    static func superBlob(entries: [(UInt32, Data)]) -> Data {
        var nextOffset = 12 + entries.count * 8
        var result = Data()
        appendUInt32BE(&result, csMagicEmbeddedSignature)
        appendUInt32BE(&result, 0)
        appendUInt32BE(&result, UInt32(entries.count))
        for (slot, blob) in entries {
            appendUInt32BE(&result, slot)
            appendUInt32BE(&result, UInt32(nextOffset))
            nextOffset += blob.count
        }
        for (_, blob) in entries { result.append(blob) }
        writeUInt32BE(UInt32(result.count), into: &result, at: 4)
        return result
    }

    static func genericBlob(magic: UInt32, payload: Data) -> Data {
        var result = Data()
        appendUInt32BE(&result, magic)
        appendUInt32BE(&result, UInt32(8 + payload.count))
        result.append(payload)
        return result
    }

    static func codeDirectory(
        hashType: UInt8,
        flags: UInt32,
        signingIdentifier: Data,
        teamIdentifier: Data
    ) -> Data {
        let fixedBytes = 52
        var result = Data(repeating: 0, count: fixedBytes)
        let identifierOffset = result.count
        result.append(signingIdentifier)
        result.append(0)
        let teamOffset = result.count
        result.append(teamIdentifier)
        result.append(0)
        let hashOffset = result.count

        writeUInt32BE(csMagicCodeDirectory, into: &result, at: 0)
        writeUInt32BE(UInt32(result.count), into: &result, at: 4)
        writeUInt32BE(0x20200, into: &result, at: 8)
        writeUInt32BE(flags, into: &result, at: 12)
        writeUInt32BE(UInt32(hashOffset), into: &result, at: 16)
        writeUInt32BE(UInt32(identifierOffset), into: &result, at: 20)
        result[36] = hashType == 2 ? 32 : 0
        result[37] = hashType
        writeUInt32BE(UInt32(teamOffset), into: &result, at: 48)
        return result
    }

    static func cacheRecord(
        suffix: Data,
        sha256: String,
        uuid: Data
    ) -> SyntheticSharedCacheFileRecord {
        SyntheticSharedCacheFileRecord(
            suffixBytes: UInt64(suffix.count),
            suffixBase64URL: base64URL(suffix),
            fileSHA256: sha256,
            fileBytes: UInt64(fileBytes),
            headerUUID: hex(uuid)
        )
    }

    static func anchoredExpectation(
        role: RuntimeClosureExpectationArtifactRole,
        cacheSet: SyntheticSharedCacheSetIdentityEvidence,
        member: RuntimeClosureExpectationMemberFields,
        edge: RuntimeClosureExpectationEdgeFields
    ) throws -> AnchoredRuntimeClosureExpectationDocument {
        let bytes = renderExpectation(
            role: role,
            cacheSet: cacheSet,
            member: member,
            edge: edge
        )
        let file = AdmittedFile(
            bytes: bytes,
            sha256: sha256(bytes),
            identity: AdmittedFileIdentity(
                device: 0,
                inode: 0,
                size: UInt64(bytes.count),
                linkCount: 1,
                mode: 0,
                modificationSeconds: 0,
                modificationNanoseconds: 0,
                changeSeconds: 0,
                changeNanoseconds: 0
            )
        )
        return try RuntimeClosureExpectationVerifier.anchor(
            expectationFile: file,
            trustAnchor: RuntimeClosureExpectationTrustAnchor(
                expectedCurrentDocumentSHA256: file.sha256,
                expectedCurrentDocumentBytes: UInt64(file.bytes.count),
                minimumEvidenceGeneration: 9,
                verificationUnixSeconds: 2_000_000_000,
                expectedArtifactRole: role
            )
        )
    }

    static func renderExpectation(
        role: RuntimeClosureExpectationArtifactRole,
        cacheSet: SyntheticSharedCacheSetIdentityEvidence,
        member: RuntimeClosureExpectationMemberFields,
        edge: RuntimeClosureExpectationEdgeFields
    ) -> Data {
        var lines = [
            RuntimeClosureExpectationVerifier.documentDomain,
            "subject=absorbed-mla-source-import-runtime-closure-identity",
            "evidence_generation=9",
            "valid_from_unix_seconds=1900000000",
            "valid_until_unix_seconds=2100000000",
            "artifact_role=\(role.rawValue)",
            "platform_architecture=arm64",
            "platform_hardware_model=Mac15,14",
            "platform_os_version=26.5.2",
            "platform_os_build=25F84",
            "resolution_profile=absolute-static-graph-v1",
            "environment_profile=no-dyld-environment-v1",
            "root_executable_content_evidence_id=\(rootID)",
            "dynamic_loader_content_evidence_id=\(loaderID)",
            "shared_cache_file_count=\(cacheSet.records.count)",
        ]
        for (index, record) in cacheSet.records.enumerated() {
            let ordinal = String(format: "%04d", index)
            lines.append(
                "shared_cache_file_\(ordinal)_suffix_bytes=\(record.suffixBytes)"
            )
            lines.append(
                "shared_cache_file_\(ordinal)_suffix_base64url=\(record.suffixBase64URL)"
            )
            lines.append(
                "shared_cache_file_\(ordinal)_sha256=\(record.fileSHA256)"
            )
            lines.append(
                "shared_cache_file_\(ordinal)_bytes=\(record.fileBytes)"
            )
            lines.append(
                "shared_cache_file_\(ordinal)_header_uuid=\(record.headerUUID)"
            )
        }
        lines.append("shared_cache_set_id=\(cacheSet.sharedCacheSetID.sha256)")
        lines.append("member_count=1")
        lines.append(
            "member_0000_content_evidence_id=\(member.contentEvidenceID)"
        )
        lines.append("member_0000_storage=\(member.storage.rawValue)")
        lines.append("member_0000_install_name_bytes=\(member.installName.bytes)")
        lines.append(
            "member_0000_install_name_base64url=\(member.installName.base64URL)"
        )
        lines.append("member_0000_macho_uuid=\(member.machOUUID)")
        lines.append(
            "member_0000_primary_code_directory_blob_sha256="
                + member.primaryCodeDirectoryBlobSHA256
        )
        lines.append(
            "member_0000_load_commands_sha256=\(member.loadCommandsSHA256)"
        )
        lines.append("edge_count=1")
        lines.append(
            "edge_0000_parent_content_evidence_id=\(edge.parentContentEvidenceID)"
        )
        lines.append(
            "edge_0000_load_command_ordinal=\(edge.loadCommandOrdinal)"
        )
        lines.append("edge_0000_kind=\(edge.kind.rawValue)")
        lines.append("edge_0000_install_name_bytes=\(edge.installName.bytes)")
        lines.append(
            "edge_0000_install_name_base64url=\(edge.installName.base64URL)"
        )
        lines.append(
            "edge_0000_resolved_content_evidence_id=\(edge.resolvedContentEvidenceID)"
        )
        lines.append("runtime_resolution_outcome=unproved-static-comparison-only")
        lines.append("runtime_authority=none")
        return Data((lines.joined(separator: "\n") + "\n").utf8)
    }

    static func fileMetadata(
        ordinal: Int
    ) -> SyntheticCaptureFileMetadata {
        SyntheticCaptureFileMetadata(
            device: 7,
            inode: UInt64(10_000 + ordinal),
            mode: 0o100644,
            linkCount: 1,
            userID: 501,
            groupID: 20,
            size: Int64(fileBytes),
            blockCount: Int64(fileBytes / 512),
            blockSize: 4_096,
            flags: 0,
            generation: UInt32(ordinal + 1),
            modificationTimeSeconds: 1_900_000_000,
            modificationTimeNanoseconds: Int64(ordinal),
            statusChangeTimeSeconds: 1_900_000_000,
            statusChangeTimeNanoseconds: Int64(ordinal),
            birthTimeSeconds: 1_800_000_000,
            birthTimeNanoseconds: Int64(ordinal),
            extendedAttributeSupportMask: 1,
            extendedFlags: 0,
            cloneID: nil,
            cloneReferenceCount: nil
        )
    }

    static func uuid(byte: UInt8) -> Data {
        var bytes = Data(repeating: 0, count: 16)
        bytes[15] = byte
        return bytes
    }

    static func writeUInt32LE(
        _ value: UInt32,
        into bytes: inout Data,
        at offset: Int
    ) {
        for index in 0..<4 {
            bytes[offset + index] = UInt8(
                truncatingIfNeeded: value >> UInt32(index * 8)
            )
        }
    }

    static func writeUInt64LE(
        _ value: UInt64,
        into bytes: inout Data,
        at offset: Int
    ) {
        for index in 0..<8 {
            bytes[offset + index] = UInt8(
                truncatingIfNeeded: value >> UInt64(index * 8)
            )
        }
    }

    static func appendUInt32LE(_ bytes: inout Data, _ value: UInt32) {
        bytes.append(UInt8(truncatingIfNeeded: value))
        bytes.append(UInt8(truncatingIfNeeded: value >> 8))
        bytes.append(UInt8(truncatingIfNeeded: value >> 16))
        bytes.append(UInt8(truncatingIfNeeded: value >> 24))
    }

    static func appendUInt32BE(_ bytes: inout Data, _ value: UInt32) {
        bytes.append(UInt8(truncatingIfNeeded: value >> 24))
        bytes.append(UInt8(truncatingIfNeeded: value >> 16))
        bytes.append(UInt8(truncatingIfNeeded: value >> 8))
        bytes.append(UInt8(truncatingIfNeeded: value))
    }

    static func writeUInt32BE(
        _ value: UInt32,
        into bytes: inout Data,
        at offset: Int
    ) {
        bytes[offset] = UInt8(truncatingIfNeeded: value >> 24)
        bytes[offset + 1] = UInt8(truncatingIfNeeded: value >> 16)
        bytes[offset + 2] = UInt8(truncatingIfNeeded: value >> 8)
        bytes[offset + 3] = UInt8(truncatingIfNeeded: value)
    }

    static func sha256(_ bytes: Data) -> String {
        SHA256.hash(data: bytes)
            .map { String(format: "%02x", $0) }
            .joined()
    }

    static func hex(_ bytes: Data) -> String {
        bytes.map { String(format: "%02x", $0) }.joined()
    }

    static func base64URL(_ bytes: Data) -> String {
        bytes.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    static func assertNoAuthority(
        _ result: SyntheticSharedCacheCompleteSetComparison,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertFalse(result.canExecute, file: file, line: line)
        XCTAssertFalse(result.canSpawn, file: file, line: line)
        XCTAssertFalse(result.canAccessNetwork, file: file, line: line)
        XCTAssertFalse(result.canConsumePack, file: file, line: line)
        XCTAssertFalse(result.canMutateFileSystem, file: file, line: line)
        XCTAssertFalse(result.canImportGitObjects, file: file, line: line)
        XCTAssertFalse(result.canBuild, file: file, line: line)
        XCTAssertFalse(result.canLoadModel, file: file, line: line)
        XCTAssertFalse(result.canReserveOutput, file: file, line: line)
        XCTAssertFalse(result.canPublish, file: file, line: line)
    }

    static func assertFailure(
        _ expression: @autoclosure () throws
            -> SyntheticSharedCacheCompleteSetComparison,
        matches: (SyntheticSharedCacheCompleteSetFailure) -> Bool,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        do {
            _ = try expression()
            XCTFail("expected SyntheticSharedCacheCompleteSetFailure",
                    file: file,
                    line: line)
        } catch let failure as SyntheticSharedCacheCompleteSetFailure {
            XCTAssertTrue(matches(failure), "\(failure)",
                          file: file,
                          line: line)
        } catch {
            XCTFail("unexpected error: \(error)", file: file, line: line)
        }
    }

    /*
     Construction-seal and forbidden-conversion probes intentionally remain
     outside the compiled XCTest suite. They must fail to compile:

     _ = SyntheticSharedCacheCompleteSetComparison()
     _ = SyntheticSharedCacheCompleteSetComparison.ConstructionSeal.verified
     _ = SyntheticSmallArtifactCaptureComparison(result)
     _ = SyntheticSharedCacheSetPlanComparison(result)
     _ = SyntheticSharedCacheDiscoveryRangePlanComparison(result)
     _ = SyntheticSharedCacheSetIdentityEvidence(result)
     _ = SyntheticSharedCacheImageContentIdentityEvidence(result)
     _ = AnchoredRuntimeClosureExpectationDocument(result)
     _ = SyntheticRuntimeClosureGraphComparison(result)
     _ = FileImageExecutionIdentityComparison(result)
     _ = AdmittedFile(result)
     */
}
