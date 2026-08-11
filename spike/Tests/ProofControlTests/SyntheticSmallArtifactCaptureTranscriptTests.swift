import CryptoKit
import Foundation
import XCTest
@testable import ProofControl

final class SyntheticSmallArtifactCaptureTranscriptTests: XCTestCase {
    func testCanonicalTranscriptRetainsExactBytesAndMetadataInertly()
        throws
    {
        var bytes = Self.canonicalBytes
        var before = Self.metadata(size: bytes.count)
        var after = before
        var transcript: [SyntheticCaptureReadEvent] = [
            .interrupted(offset: 0),
            .bytes(offset: 0, data: Data(bytes.prefix(7))),
            .bytes(offset: 7, data: Data(bytes.dropFirst(7))),
            .endOfFile(offset: UInt64(bytes.count)),
        ]
        let effects = FakeOperationalEffects()

        let comparison = try SyntheticSmallArtifactCaptureVerifier.compare(
            role: .gitRoot,
            before: before,
            after: after,
            expectedFileBytes: 30,
            expectedSHA256:
                "8990d279bd0e9ef1c04baf89210ac86b" +
                "a28a1aad4e6d2801785225ca12489ee7",
            transcript: transcript
        )

        XCTAssertEqual(comparison.role, .gitRoot)
        XCTAssertEqual(comparison.metadata, before)
        XCTAssertEqual(comparison.retainedBytes, Self.canonicalBytes)
        XCTAssertEqual(comparison.fileBytes, 30)
        XCTAssertEqual(
            comparison.fileSHA256,
            "8990d279bd0e9ef1c04baf89210ac86b" +
                "a28a1aad4e6d2801785225ca12489ee7"
        )
        XCTAssertEqual(
            Self.sha256Hex(comparison.retainedBytes),
            comparison.fileSHA256
        )
        XCTAssertEqual(
            Set(Mirror(reflecting: comparison).children.compactMap(\.label)),
            [
                "role", "metadata", "retainedBytes", "fileSHA256",
                "fileBytes", "constructionSeal", "canExecute", "canSpawn",
                "canAccessNetwork", "canConsumePack", "canMutateFileSystem",
                "canImportGitObjects", "canBuild", "canLoadModel",
                "canReserveOutput", "canPublish",
            ]
        )
        Self.assertInert(comparison)

        bytes[0] ^= 0xff
        before.device &+= 1
        after.inode &+= 1
        transcript[1] = .bytes(offset: 0, data: Data([0xff]))
        XCTAssertEqual(comparison.metadata.device, 7)
        XCTAssertEqual(comparison.metadata.inode, 11)
        XCTAssertEqual(comparison.retainedBytes, Self.canonicalBytes)
        XCTAssertEqual(effects, .zero)
    }

    func testExactOneMiBAndSixteenFragmentsPerChunkRemainBounded()
        throws
    {
        let bytes = Data(
            (0..<SyntheticSmallArtifactCaptureVerifier.maximumFileBytes)
                .map { UInt8(truncatingIfNeeded: $0) }
        )
        let metadata = Self.metadata(size: bytes.count)
        var transcript: [SyntheticCaptureReadEvent] = []
        var offset = 0
        while offset < bytes.count {
            let chunkEnd = min(
                offset + SyntheticSmallArtifactCaptureVerifier
                    .maximumReadChunkBytes,
                bytes.count
            )
            let fragmentBytes = (chunkEnd - offset) / 16
            for fragment in 0..<16 {
                transcript.append(
                    contentsOf: Array(
                        repeating: SyntheticCaptureReadEvent
                            .interrupted(offset: UInt64(offset)),
                        count: SyntheticSmallArtifactCaptureVerifier
                            .maximumInterruptedRetriesPerOffset
                    )
                )
                let end = fragment == 15
                    ? chunkEnd
                    : offset + fragmentBytes
                transcript.append(
                    .bytes(
                        offset: UInt64(offset),
                        data: bytes.subdata(in: offset..<end)
                    )
                )
                offset = end
            }
        }
        transcript.append(.endOfFile(offset: UInt64(bytes.count)))
        XCTAssertEqual(transcript.count, 2_305)

        let comparison = try SyntheticSmallArtifactCaptureVerifier.compare(
            role: .fileImage,
            before: metadata,
            after: metadata,
            expectedFileBytes: UInt64(bytes.count),
            expectedSHA256: Self.sha256Hex(bytes),
            transcript: transcript
        )

        XCTAssertEqual(
            comparison.fileBytes,
            UInt64(SyntheticSmallArtifactCaptureVerifier.maximumFileBytes)
        )
        XCTAssertEqual(comparison.retainedBytes, bytes)
        Self.assertInert(comparison)
    }

    func testPreReadMetadataRulesRefuseInExactOrder() {
        let valid = Self.metadata(size: Self.canonicalBytes.count)
        let effects = FakeOperationalEffects()
        let cases: [
            (SyntheticCaptureFileMetadata, SyntheticSmallArtifactCaptureFailure)
        ] = [
            (
                Self.changing(valid) { $0.mode = 0o040755 },
                .fileType
            ),
            (
                Self.changing(valid) { $0.linkCount = 2 },
                .linkCount(2)
            ),
            (
                Self.changing(valid) { $0.size = -1 },
                .fileSize(-1)
            ),
            (Self.changing(valid) { $0.size = 0 }, .fileSize(0)),
            (
                Self.changing(valid) {
                    $0.size = Int64(
                        SyntheticSmallArtifactCaptureVerifier.maximumFileBytes
                    ) + 1
                },
                .fileSize(
                    Int64(
                        SyntheticSmallArtifactCaptureVerifier.maximumFileBytes
                    ) + 1
                )
            ),
            (
                Self.changing(valid) { $0.size = Int64.max },
                .fileSize(Int64.max)
            ),
            (
                Self.changing(valid) {
                    $0.flags |= SyntheticSmallArtifactCaptureVerifier
                        .datalessFlag
                },
                .dataless
            ),
            (
                Self.changing(valid) {
                    $0.extendedAttributeSupportMask = 0
                },
                .sparseStateUnavailable
            ),
            (
                Self.changing(valid) {
                    $0.extendedFlags |= SyntheticSmallArtifactCaptureVerifier
                        .sparseFlag
                },
                .sparseFile
            ),
        ]

        for (metadata, expected) in cases {
            Self.assertFailure(
                expected,
                before: metadata,
                after: metadata,
                expectedFileBytes: UInt64(Self.canonicalBytes.count),
                expectedSHA256: Self.canonicalSHA256,
                transcript: [.error(offset: 0, code: 99)]
            )
            XCTAssertEqual(effects, .zero)
        }

        let combined = Self.changing(valid) {
            $0.mode = 0o040755
            $0.linkCount = 2
            $0.flags |= SyntheticSmallArtifactCaptureVerifier.datalessFlag
            $0.extendedAttributeSupportMask = 0
            $0.size = 0
        }
        Self.assertFailure(
            .fileType,
            before: combined,
            after: combined,
            expectedFileBytes: 0,
            expectedSHA256: "NOT-HEX",
            transcript: [.error(offset: 0, code: 99)]
        )
        XCTAssertEqual(effects, .zero)

        let invalidSizeAndDataless = Self.changing(valid) {
            $0.size = 0
            $0.flags |= SyntheticSmallArtifactCaptureVerifier.datalessFlag
        }
        Self.assertFailure(
            .fileSize(0),
            before: invalidSizeAndDataless,
            after: invalidSizeAndDataless,
            expectedFileBytes: 0,
            expectedSHA256: "NOT-HEX",
            transcript: [.error(offset: 0, code: 99)]
        )
    }

    func testExpectedByteCountAndDigestRulesRejectBeforeConstruction() {
        let metadata = Self.metadata(size: Self.canonicalBytes.count)
        let transcript = Self.canonicalTranscript()
        let effects = FakeOperationalEffects()

        for count in [UInt64(0), 29, UInt64.max] {
            Self.assertFailure(
                .expectedByteCount(expected: count, actual: 30),
                before: metadata,
                after: metadata,
                expectedFileBytes: count,
                expectedSHA256: Self.canonicalSHA256,
                transcript: transcript
            )
        }
        for malformed in [
            String(repeating: "a", count: 63),
            String(repeating: "A", count: 64),
            String(repeating: "g", count: 64),
        ] {
            Self.assertFailure(
                .expectedSHA256WhenExplicit(.format),
                before: metadata,
                after: metadata,
                expectedFileBytes: 30,
                expectedSHA256: malformed,
                transcript: transcript
            )
        }
        Self.assertFailure(
            .expectedSHA256WhenExplicit(
                .mismatch(
                    expected: String(repeating: "0", count: 64),
                    actual: Self.canonicalSHA256
                )
            ),
            before: metadata,
            after: metadata,
            expectedFileBytes: 30,
            expectedSHA256: String(repeating: "0", count: 64),
            transcript: transcript
        )
        XCTAssertEqual(effects, .zero)
    }

    func testOffsetGapOverlapDuplicateAndReorderRefuseExactly() {
        let bytes = Data([0x10, 0x11, 0x12])
        let metadata = Self.metadata(size: bytes.count)
        let effects = FakeOperationalEffects()
        let cases: [
            ([SyntheticCaptureReadEvent], SyntheticSmallArtifactCaptureFailure)
        ] = [
            (
                [.bytes(offset: 1, data: bytes)],
                .readError(.offset(index: 0, expected: 0, actual: 1))
            ),
            (
                [
                    .bytes(offset: 0, data: Data([0x10])),
                    .bytes(offset: 0, data: Data([0x11, 0x12])),
                ],
                .readError(.offset(index: 1, expected: 1, actual: 0))
            ),
            (
                [
                    .interrupted(offset: 0),
                    .bytes(offset: 0, data: Data([0x10])),
                    .interrupted(offset: 0),
                ],
                .readError(.offset(index: 2, expected: 1, actual: 0))
            ),
            (
                [.bytes(offset: UInt64.max, data: bytes)],
                .readError(
                    .offset(
                        index: 0,
                        expected: 0,
                        actual: UInt64.max
                    )
                )
            ),
        ]

        for (transcript, expected) in cases {
            Self.assertFailure(
                expected,
                before: metadata,
                after: metadata,
                expectedFileBytes: UInt64(bytes.count),
                expectedSHA256: Self.sha256Hex(bytes),
                transcript: transcript
            )
            XCTAssertEqual(effects, .zero)
        }
    }

    func testRetryFragmentAndChunkCeilingsRefuseExactly() {
        let effects = FakeOperationalEffects()

        let oneByte = Data([0x41])
        let oneByteMetadata = Self.metadata(size: oneByte.count)
        let eightInterruptions = Array(
            repeating: SyntheticCaptureReadEvent.interrupted(offset: 0),
            count: 8
        ) + [
            .bytes(offset: 0, data: oneByte),
            .endOfFile(offset: 1),
        ]
        XCTAssertNoThrow(
            try SyntheticSmallArtifactCaptureVerifier.compare(
                role: .dynamicLoader,
                before: oneByteMetadata,
                after: oneByteMetadata,
                expectedFileBytes: 1,
                expectedSHA256: Self.sha256Hex(oneByte),
                transcript: eightInterruptions
            )
        )
        Self.assertFailure(
            .readInterruptedLimit(offset: 0),
            before: oneByteMetadata,
            after: oneByteMetadata,
            expectedFileBytes: 1,
            expectedSHA256: Self.sha256Hex(oneByte),
            transcript:
                Array(
                    repeating: SyntheticCaptureReadEvent
                        .interrupted(offset: 0),
                    count: 9
                )
        )

        let oneByteMaximumAttempts = 145
        Self.assertFailure(
            .resourceEnvelope(
                .readAttemptLimit(
                    actual: UInt64(oneByteMaximumAttempts + 1),
                    maximum: UInt64(oneByteMaximumAttempts)
                )
            ),
            before: oneByteMetadata,
            after: oneByteMetadata,
            expectedFileBytes: 1,
            expectedSHA256: Self.sha256Hex(oneByte),
            transcript: Array(
                repeating: SyntheticCaptureReadEvent
                    .interrupted(offset: 0),
                count: oneByteMaximumAttempts + 1
            )
        )

        let seventeenBytes = Data(repeating: 0x42, count: 17)
        let seventeenMetadata = Self.metadata(size: seventeenBytes.count)
        let seventeenFragments = (0..<17).map {
            SyntheticCaptureReadEvent.bytes(
                offset: UInt64($0),
                data: Data([0x42])
            )
        }
        Self.assertFailure(
            .readFragmentLimit(chunkOffset: 0),
            before: seventeenMetadata,
            after: seventeenMetadata,
            expectedFileBytes: 17,
            expectedSHA256: Self.sha256Hex(seventeenBytes),
            transcript: seventeenFragments
        )

        let oversizedFragment = Data(
            repeating: 0x43,
            count: SyntheticSmallArtifactCaptureVerifier
                .maximumReadChunkBytes + 1
        )
        let oversizedMetadata = Self.metadata(
            size: oversizedFragment.count
        )
        Self.assertFailure(
            .readError(
                .fragmentSize(
                    index: 0,
                    bytes: SyntheticSmallArtifactCaptureVerifier
                        .maximumReadChunkBytes + 1
                )
            ),
            before: oversizedMetadata,
            after: oversizedMetadata,
            expectedFileBytes: UInt64(oversizedFragment.count),
            expectedSHA256: Self.sha256Hex(oversizedFragment),
            transcript: [.bytes(offset: 0, data: oversizedFragment)]
        )

        let crossingBytes = Data(
            repeating: 0x44,
            count: SyntheticSmallArtifactCaptureVerifier
                .maximumReadChunkBytes + 464
        )
        let crossingMetadata = Self.metadata(size: crossingBytes.count)
        Self.assertFailure(
            .readError(
                .fragmentBeyondChunk(
                    index: 1,
                    chunkEnd: UInt64(
                        SyntheticSmallArtifactCaptureVerifier
                            .maximumReadChunkBytes
                    )
                )
            ),
            before: crossingMetadata,
            after: crossingMetadata,
            expectedFileBytes: UInt64(crossingBytes.count),
            expectedSHA256: Self.sha256Hex(crossingBytes),
            transcript: [
                .bytes(
                    offset: 0,
                    data: Data(
                        crossingBytes.prefix(
                            SyntheticSmallArtifactCaptureVerifier
                                .maximumReadChunkBytes - 536
                        )
                    )
                ),
                .bytes(
                    offset: UInt64(
                        SyntheticSmallArtifactCaptureVerifier
                            .maximumReadChunkBytes - 536
                    ),
                    data: Data(repeating: 0x44, count: 1_000)
                ),
            ]
        )
        XCTAssertEqual(effects, .zero)
    }


    func testReadAttemptArithmeticIsChecked() throws {
        XCTAssertEqual(
            try SyntheticSmallArtifactCaptureVerifier
                .checkedReadAttemptLimit(
                    forFileBytes: UInt64(
                        SyntheticSmallArtifactCaptureVerifier.maximumFileBytes
                    )
                ),
            2_305
        )
        XCTAssertThrowsError(
            try SyntheticSmallArtifactCaptureVerifier
                .checkedReadAttemptLimit(forChunkCount: UInt64.max)
        ) {
            XCTAssertEqual(
                $0 as? SyntheticSmallArtifactCaptureFailure,
                .resourceEnvelope(.arithmeticOverflow)
            )
        }
    }

    func testReadErrorEOFAndTrailingDataRefuseExactly() {
        let bytes = Data([0x51, 0x52])
        let metadata = Self.metadata(size: bytes.count)
        let digest = Self.sha256Hex(bytes)
        let effects = FakeOperationalEffects()

        Self.assertFailure(
            .readError(.system(offset: 0, code: 35)),
            before: metadata,
            after: metadata,
            expectedFileBytes: 2,
            expectedSHA256: digest,
            transcript: [.error(offset: 0, code: 35)]
        )
        Self.assertFailure(
            .shortRead(.endOfFile(expected: 2, actual: 1)),
            before: metadata,
            after: metadata,
            expectedFileBytes: 2,
            expectedSHA256: digest,
            transcript: [
                .bytes(offset: 0, data: Data([0x51])),
                .endOfFile(offset: 1),
            ]
        )
        Self.assertFailure(
            .shortRead(.missingEndOfFile(offset: 2)),
            before: metadata,
            after: metadata,
            expectedFileBytes: 2,
            expectedSHA256: digest,
            transcript: [.bytes(offset: 0, data: bytes)]
        )
        Self.assertFailure(
            .shortRead(.transcriptEnded(expected: 2, actual: 1)),
            before: metadata,
            after: metadata,
            expectedFileBytes: 2,
            expectedSHA256: digest,
            transcript: [.bytes(offset: 0, data: Data([0x51]))]
        )
        Self.assertFailure(
            .unexpectedTrailingByte(index: 0),
            before: metadata,
            after: metadata,
            expectedFileBytes: 2,
            expectedSHA256: digest,
            transcript: [
                .bytes(offset: 0, data: Data([0x51, 0x52, 0x53]))
            ]
        )
        Self.assertFailure(
            .unexpectedTrailingByte(index: 2),
            before: metadata,
            after: metadata,
            expectedFileBytes: 2,
            expectedSHA256: digest,
            transcript: [
                .bytes(offset: 0, data: bytes),
                .endOfFile(offset: 2),
                .interrupted(offset: 2),
            ]
        )
        Self.assertFailure(
            .unexpectedTrailingByte(index: 1),
            before: metadata,
            after: metadata,
            expectedFileBytes: 2,
            expectedSHA256: digest,
            transcript: [
                .bytes(offset: 0, data: bytes),
                .interrupted(offset: 2),
                .endOfFile(offset: 2),
            ]
        )
        Self.assertFailure(
            .readError(.fragmentSize(index: 0, bytes: 0)),
            before: metadata,
            after: metadata,
            expectedFileBytes: 2,
            expectedSHA256: digest,
            transcript: [.bytes(offset: 0, data: Data())]
        )
        XCTAssertEqual(effects, .zero)
    }

    func testEveryMetadataScalarDriftRefusesAfterRead() {
        let before = Self.metadata(size: Self.canonicalBytes.count)
        let effects = FakeOperationalEffects()

        for field in SyntheticSmallArtifactCaptureFailure.MetadataField
            .allCases
        {
            let after = Self.drifted(before, field: field)
            Self.assertFailure(
                .metadataDrift(field),
                before: before,
                after: after,
                expectedFileBytes: 30,
                expectedSHA256: Self.canonicalSHA256,
                transcript: Self.canonicalTranscript()
            )
            XCTAssertEqual(effects, .zero)
        }
    }

    func testReadFailurePrecedesDriftAndDriftPrecedesDigest() {
        let before = Self.metadata(size: Self.canonicalBytes.count)
        let after = Self.drifted(before, field: .inode)
        let effects = FakeOperationalEffects()

        Self.assertFailure(
            .readError(.system(offset: 0, code: 5)),
            before: before,
            after: after,
            expectedFileBytes: 29,
            expectedSHA256: "NOT-HEX",
            transcript: [.error(offset: 0, code: 5)]
        )
        Self.assertFailure(
            .metadataDrift(.inode),
            before: before,
            after: after,
            expectedFileBytes: 29,
            expectedSHA256: "NOT-HEX",
            transcript: Self.canonicalTranscript()
        )
        let multipleDrifts = Self.changing(before) {
            $0.device &+= 1
            $0.inode &+= 1
        }
        Self.assertFailure(
            .metadataDrift(.device),
            before: before,
            after: multipleDrifts,
            expectedFileBytes: 30,
            expectedSHA256: Self.canonicalSHA256,
            transcript: Self.canonicalTranscript()
        )
        XCTAssertEqual(effects, .zero)
    }
}

private extension SyntheticSmallArtifactCaptureTranscriptTests {
    struct FakeOperationalEffects: Equatable {
        var pathCount = 0
        var descriptorCount = 0
        var fileSystemCount = 0
        var securityCount = 0
        var processCount = 0
        var networkCount = 0
        var packCount = 0
        var objectDatabaseCount = 0
        var sourceCount = 0
        var buildCount = 0
        var modelCount = 0
        var reservationCount = 0
        var mergeCount = 0
        var publicationCount = 0
        var performanceCount = 0

        static let zero = Self()
    }

    static let canonicalBytes = Data(
        "fast-mlx-e1-synthetic-capture\n".utf8
    )
    static let canonicalSHA256 =
        "8990d279bd0e9ef1c04baf89210ac86b" +
        "a28a1aad4e6d2801785225ca12489ee7"

    static func metadata(size: Int) -> SyntheticCaptureFileMetadata {
        SyntheticCaptureFileMetadata(
            device: 7,
            inode: 11,
            mode: 0o100755,
            linkCount: 1,
            userID: 501,
            groupID: 20,
            size: Int64(size),
            blockCount: 8,
            blockSize: 4_096,
            flags: 0x20,
            generation: 3,
            modificationTimeSeconds: 1_785_700_001,
            modificationTimeNanoseconds: 101,
            statusChangeTimeSeconds: 1_785_700_002,
            statusChangeTimeNanoseconds: 202,
            birthTimeSeconds: 1_785_700_003,
            birthTimeNanoseconds: 303,
            extendedAttributeSupportMask:
                SyntheticSmallArtifactCaptureVerifier
                .requiredSparseStateSupportBit,
            extendedFlags: 0,
            cloneID: nil,
            cloneReferenceCount: nil
        )
    }

    static func canonicalTranscript() -> [SyntheticCaptureReadEvent] {
        [
            .interrupted(offset: 0),
            .bytes(offset: 0, data: Data(canonicalBytes.prefix(7))),
            .bytes(offset: 7, data: Data(canonicalBytes.dropFirst(7))),
            .endOfFile(offset: 30),
        ]
    }

    static func changing(
        _ metadata: SyntheticCaptureFileMetadata,
        _ body: (inout SyntheticCaptureFileMetadata) -> Void
    ) -> SyntheticCaptureFileMetadata {
        var result = metadata
        body(&result)
        return result
    }

    static func drifted(
        _ metadata: SyntheticCaptureFileMetadata,
        field: SyntheticSmallArtifactCaptureFailure.MetadataField
    ) -> SyntheticCaptureFileMetadata {
        changing(metadata) { value in
            switch field {
            case .device:
                value.device &+= 1
            case .inode:
                value.inode &+= 1
            case .mode:
                value.mode ^= 0o100
            case .linkCount:
                value.linkCount &+= 1
            case .userID:
                value.userID &+= 1
            case .groupID:
                value.groupID &+= 1
            case .size:
                value.size &+= 1
            case .blockCount:
                value.blockCount &+= 1
            case .blockSize:
                value.blockSize &+= 1
            case .flags:
                value.flags ^= 0x20
            case .generation:
                value.generation = nil
            case .modificationTimeSeconds:
                value.modificationTimeSeconds &+= 1
            case .modificationTimeNanoseconds:
                value.modificationTimeNanoseconds &+= 1
            case .statusChangeTimeSeconds:
                value.statusChangeTimeSeconds &+= 1
            case .statusChangeTimeNanoseconds:
                value.statusChangeTimeNanoseconds &+= 1
            case .birthTimeSeconds:
                value.birthTimeSeconds &+= 1
            case .birthTimeNanoseconds:
                value.birthTimeNanoseconds &+= 1
            case .extendedAttributeSupportMask:
                value.extendedAttributeSupportMask &+= 2
            case .extendedFlags:
                value.extendedFlags &+= 2
            case .cloneID:
                value.cloneID = 41
            case .cloneReferenceCount:
                value.cloneReferenceCount = 2
            }
        }
    }

    static func assertFailure(
        _ expected: SyntheticSmallArtifactCaptureFailure,
        before: SyntheticCaptureFileMetadata,
        after: SyntheticCaptureFileMetadata,
        expectedFileBytes: UInt64,
        expectedSHA256: String,
        transcript: [SyntheticCaptureReadEvent],
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(
            try SyntheticSmallArtifactCaptureVerifier.compare(
                role: .gitRoot,
                before: before,
                after: after,
                expectedFileBytes: expectedFileBytes,
                expectedSHA256: expectedSHA256,
                transcript: transcript
            ),
            file: file,
            line: line
        ) {
            XCTAssertEqual(
                $0 as? SyntheticSmallArtifactCaptureFailure,
                expected,
                file: file,
                line: line
            )
        }
    }

    static func assertInert(
        _ value: SyntheticSmallArtifactCaptureComparison,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertFalse(value.canExecute, file: file, line: line)
        XCTAssertFalse(value.canSpawn, file: file, line: line)
        XCTAssertFalse(value.canAccessNetwork, file: file, line: line)
        XCTAssertFalse(value.canConsumePack, file: file, line: line)
        XCTAssertFalse(
            value.canMutateFileSystem,
            file: file,
            line: line
        )
        XCTAssertFalse(
            value.canImportGitObjects,
            file: file,
            line: line
        )
        XCTAssertFalse(value.canBuild, file: file, line: line)
        XCTAssertFalse(value.canLoadModel, file: file, line: line)
        XCTAssertFalse(value.canReserveOutput, file: file, line: line)
        XCTAssertFalse(value.canPublish, file: file, line: line)
    }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}
