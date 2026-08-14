import XCTest
import Foundation

@testable import ProofControl

final class SyntheticSharedCacheSetPlanTests: XCTestCase {
    func testE2ATypeExistenceGreenSentinel() {
        _ = SyntheticSharedCacheSourceProfile.self
        _ = SyntheticSharedCacheDiscoveryFile.self
        _ = SyntheticSharedCacheSetPlanComparison.self
        _ = SyntheticSharedCacheSetPlanFailure.self
        _ = SyntheticSharedCacheSetPlanVerifier.self
    }

    func testMainOnlyAndThreeFileFixturesDeriveExactHeaderOrderPlan()
        throws
    {
        let mainOnly = Self.mainOnlyFixture
        let mainOnlyPlan = try Self.derive(mainOnly)

        Self.assertPlanMatchesFixture(mainOnlyPlan, fixture: mainOnly)
        XCTAssertEqual(mainOnlyPlan.headerOrderFiles.count, 1)
        XCTAssertEqual(mainOnlyPlan.headerOrderFiles[0].ordinal, 0)
        XCTAssertEqual(mainOnlyPlan.headerOrderFiles[0].decodedSuffix, Data())
        XCTAssertEqual(mainOnlyPlan.headerOrderFiles[0].suffixBytes, Data())
        XCTAssertEqual(
            mainOnlyPlan.headerOrderFiles[0].suffixByteCount,
            UInt64(0)
        )
        XCTAssertEqual(mainOnlyPlan.headerOrderFiles[0].suffixBase64URL, "")
        XCTAssertEqual(mainOnlyPlan.headerOrderFiles[0].cacheVMOffset, 0)

        let threeFile = Self.threeFileFixture
        let threeFilePlan = try Self.derive(threeFile)

        Self.assertPlanMatchesFixture(threeFilePlan, fixture: threeFile)
        XCTAssertEqual(
            threeFilePlan.headerOrderFiles.map(\.ordinal),
            [0, 1, 2]
        )
        XCTAssertEqual(
            threeFilePlan.headerOrderFiles.map(\.decodedSuffix),
            [Data(), Data(".2".utf8), Data(".1".utf8)]
        )
        XCTAssertEqual(
            threeFilePlan.headerOrderFiles.map(\.suffixByteCount),
            [UInt64(0), UInt64(2), UInt64(2)]
        )
        XCTAssertEqual(
            threeFilePlan.headerOrderFiles.map(\.suffixBase64URL),
            ["", Self.base64URL(Data(".2".utf8)),
             Self.base64URL(Data(".1".utf8))]
        )
        XCTAssertTrue(Data(".1".utf8).lexicographicallyPrecedes(
            Data(".2".utf8)
        ))
        XCTAssertEqual(
            threeFilePlan.headerOrderFiles.map(\.cacheVMOffset),
            [0, 0x0020_0000, 0x0040_0000]
        )
    }

    func testNonemptyMainImageTableUsesExactUInt32OffsetWidth()
        throws
    {
        var fixture = Self.mainOnlyFixture
        let imagesOffset = fixture.main.discoveryBytes.count
        fixture.main.discoveryBytes.append(
            Data(repeating: 0, count: 2 * 32)
        )
        Self.writeUInt32LE(
            UInt32(imagesOffset),
            into: &fixture.main.discoveryBytes,
            at: 448
        )
        Self.writeUInt32LE(
            1,
            into: &fixture.main.discoveryBytes,
            at: 452
        )
        Self.writeUInt64LE(
            UInt64(imagesOffset + 32),
            into: &fixture.main.discoveryBytes,
            at: 136
        )
        Self.writeUInt64LE(
            1,
            into: &fixture.main.discoveryBytes,
            at: 144
        )
        Self.writeUInt32LE(
            UInt32(imagesOffset + 64),
            into: &fixture.main.discoveryBytes,
            at: 392
        )

        let plan = try Self.derive(fixture)

        XCTAssertEqual(plan.mainHeader.imagesOffset, UInt64(imagesOffset))
        XCTAssertEqual(plan.mainHeader.imagesCount, 1)
        XCTAssertEqual(
            plan.mainHeader.imagesTextOffset,
            UInt64(imagesOffset + 32)
        )
        XCTAssertEqual(plan.mainHeader.imagesTextCount, 1)
        XCTAssertEqual(
            plan.mainHeader.subcacheTableEnd,
            UInt64(imagesOffset + 64)
        )
    }

    func testSourceProfileAndLayoutConstantsAreExact() throws {
        let profile = SyntheticSharedCacheSourceProfile.reviewed
        let plan = try Self.derive(Self.threeFileFixture)

        XCTAssertEqual(plan.sourceProfile, profile)
        XCTAssertEqual(profile.tag, "dyld-1378")
        XCTAssertEqual(
            profile.tagObject,
            "79fdd08288695d6dbbdac41f022a2851d20fa647"
        )
        XCTAssertEqual(
            profile.commit,
            "fd8d0c4d52320ebf64db34f3cb280310d905c5ae"
        )
        XCTAssertEqual(
            profile.dyldCacheFormatHeaderSHA256,
            "dd6f7d9ffc5cb318988c16dbecf958d04b0c65cd9ee1892a838e374f76fd182c"
        )
        XCTAssertEqual(
            profile.dyldSharedCacheImplementationSHA256,
            "2edb158b41203e595b1d95937acad429888ce1464f25b7b620cb9b19f38e6475"
        )
        XCTAssertEqual(
            profile.dyldSharedCacheHeaderSHA256,
            "881a5f2e174f458f5c05c2be04bd4f7eee943d4a3950e3ecd58e26e94b8b743f"
        )
        XCTAssertEqual(
            profile.sharedCacheRuntimeSHA256,
            "5edf7835d47f6b78110da10d556cd45e2ac04c04a9d1627294522dc4aca058be"
        )
        XCTAssertEqual(
            profile.subCacheBuilderSHA256,
            "d4f06686b932360c7d93f5afda8b2e8eb50d6bdcf742dc2a431f0b91c0967b8a"
        )
        XCTAssertEqual(
            profile.newSharedCacheBuilderSHA256,
            "58d1f19d122c16a04b04f27fb7336cb41afc9dfa5ed5cb28c22a93fc9d6d48c5"
        )
        XCTAssertEqual(profile.layout.headerBytes, 552)
        XCTAssertEqual(profile.layout.mappingOffsetOffset, 16)
        XCTAssertEqual(profile.layout.mappingCountOffset, 20)
        XCTAssertEqual(profile.layout.codeSignatureOffsetOffset, 40)
        XCTAssertEqual(profile.layout.uuidOffset, 88)
        XCTAssertEqual(profile.layout.cacheTypeOffset, 104)
        XCTAssertEqual(profile.layout.platformOffset, 216)
        XCTAssertEqual(profile.layout.formatFlagsOffset, 220)
        XCTAssertEqual(profile.layout.mappingWithSlideOffsetOffset, 312)
        XCTAssertEqual(profile.layout.subCacheArrayOffsetOffset, 392)
        XCTAssertEqual(profile.layout.subCacheArrayCountOffset, 396)
        XCTAssertEqual(profile.layout.symbolFileUUIDOffset, 400)
        XCTAssertEqual(profile.layout.imagesOffsetOffset, 448)
        XCTAssertEqual(profile.layout.imagesCountOffset, 452)
        XCTAssertEqual(profile.layout.cacheSubTypeOffset, 456)
        XCTAssertEqual(profile.layout.legacySubcacheEntryBytes, 24)
        XCTAssertEqual(profile.layout.modernSubcacheEntryBytes, 56)
        XCTAssertEqual(profile.layout.modernSubcacheUUIDOffset, 0)
        XCTAssertEqual(profile.layout.modernSubcacheVMOffsetOffset, 16)
        XCTAssertEqual(profile.layout.modernSubcacheSuffixOffset, 24)
        XCTAssertEqual(profile.layout.mappingInfoBytes, 32)
        XCTAssertEqual(profile.layout.mappingAndSlideInfoBytes, 56)
        XCTAssertEqual(profile.layout.imageInfoBytes, 32)
        XCTAssertEqual(profile.layout.imageTextInfoBytes, 32)
        XCTAssertEqual(profile.policy.maximumFileCount, 64)
        XCTAssertEqual(profile.policy.maximumSubcacheCount, 63)
        XCTAssertEqual(profile.policy.maximumDiscoveryBytes, 1_048_576)
        XCTAssertEqual(profile.policy.maximumFileBytes, 17_179_869_184)
        XCTAssertEqual(profile.policy.maximumAggregateBytes, 68_719_476_736)
        XCTAssertEqual(profile.policy.maximumMappingCount, 8)
        XCTAssertEqual(
            SyntheticSharedCacheSetPlanVerifier.dyldCacheHeaderBytes,
            Self.headerBytes
        )
        XCTAssertEqual(
            SyntheticSharedCacheSetPlanVerifier.mappingInfoBytes,
            Self.mappingInfoBytes
        )
        XCTAssertEqual(
            SyntheticSharedCacheSetPlanVerifier.mappingAndSlideInfoBytes,
            Self.mappingAndSlideInfoBytes
        )
        XCTAssertEqual(
            SyntheticSharedCacheSetPlanVerifier.modernSubcacheEntryBytes,
            Self.modernSubcacheEntryBytes
        )
        XCTAssertEqual(SyntheticSharedCacheSetPlanVerifier.imageInfoBytes, 32)
        XCTAssertEqual(
            SyntheticSharedCacheSetPlanVerifier.imageTextInfoBytes,
            32
        )
        XCTAssertEqual(
            SyntheticSharedCacheSetPlanVerifier.maximumCacheFileCount,
            64
        )
        XCTAssertEqual(
            SyntheticSharedCacheSetPlanVerifier.maximumSubcacheFileCount,
            63
        )
        XCTAssertEqual(
            SyntheticSharedCacheSetPlanVerifier.maximumDiscoveryBytes,
            1_048_576
        )
        XCTAssertEqual(
            SyntheticSharedCacheSetPlanVerifier.maximumCacheFileBytes,
            17_179_869_184
        )
        XCTAssertEqual(
            SyntheticSharedCacheSetPlanVerifier.maximumCacheSetBytes,
            68_719_476_736
        )
        XCTAssertEqual(
            SyntheticSharedCacheSetPlanVerifier.maximumMappingCount,
            8
        )
        XCTAssertEqual(
            SyntheticSharedCacheSetPlanVerifier.mappingOffsetFieldOffset,
            16
        )
        XCTAssertEqual(
            SyntheticSharedCacheSetPlanVerifier.mappingCountFieldOffset,
            20
        )
        XCTAssertEqual(
            SyntheticSharedCacheSetPlanVerifier.codeSignatureOffsetFieldOffset,
            40
        )
        XCTAssertEqual(
            SyntheticSharedCacheSetPlanVerifier.uuidFieldOffset,
            88
        )
        XCTAssertEqual(
            SyntheticSharedCacheSetPlanVerifier.cacheTypeFieldOffset,
            104
        )
        XCTAssertEqual(
            SyntheticSharedCacheSetPlanVerifier.platformFieldOffset,
            216
        )
        XCTAssertEqual(
            SyntheticSharedCacheSetPlanVerifier.formatFlagsFieldOffset,
            220
        )
        XCTAssertEqual(
            SyntheticSharedCacheSetPlanVerifier.mappingWithSlideOffsetFieldOffset,
            312
        )
        XCTAssertEqual(
            SyntheticSharedCacheSetPlanVerifier.subCacheArrayOffsetFieldOffset,
            392
        )
        XCTAssertEqual(
            SyntheticSharedCacheSetPlanVerifier.subCacheArrayCountFieldOffset,
            396
        )
        XCTAssertEqual(
            SyntheticSharedCacheSetPlanVerifier.symbolFileUUIDFieldOffset,
            400
        )
        XCTAssertEqual(
            SyntheticSharedCacheSetPlanVerifier.imagesOffsetFieldOffset,
            448
        )
        XCTAssertEqual(
            SyntheticSharedCacheSetPlanVerifier.imagesCountFieldOffset,
            452
        )
        XCTAssertEqual(
            SyntheticSharedCacheSetPlanVerifier.cacheSubTypeFieldOffset,
            456
        )

        XCTAssertEqual(plan.mainHeader.mappingOffset, UInt32(Self.headerBytes))
        XCTAssertEqual(plan.mainHeader.mappingCount, UInt32(Self.mappingCount))
        XCTAssertEqual(
            plan.mainHeader.mappingWithSlideOffset,
            UInt32(Self.mappingWithSlideOffset)
        )
        XCTAssertEqual(
            plan.mainHeader.mappingWithSlideCount,
            UInt32(Self.mappingCount)
        )
        XCTAssertEqual(plan.mainHeader.mappingTablesEnd,
            UInt64(Self.mappingTablesEnd)
        )
        XCTAssertEqual(
            plan.mainHeader.codeSignatureOffset,
            UInt64(Self.mainFileBytes) - Self.codeSignatureBytes
        )
        XCTAssertEqual(
            plan.mainHeader.codeSignatureSize,
            Self.codeSignatureBytes
        )
        XCTAssertEqual(
            plan.mainHeader.subCacheArrayOffset,
            UInt32(Self.mappingTablesEnd)
        )
        XCTAssertEqual(plan.mainHeader.subCacheArrayCount, 2)
        XCTAssertEqual(
            plan.mainHeader.subcacheTableEnd,
            UInt64(Self.mappingTablesEnd + 2 * Self.modernSubcacheEntryBytes)
        )
    }

    func testCallerMutationCannotChangeReturnedFactsAndHeaderOrderWins()
        throws
    {
        var fixture = Self.threeFileFixture
        let originalMainBytes = fixture.main.discoveryBytes
        let originalSubcacheBytes = fixture.subcaches.map(\.discoveryBytes)
        var inputBytes = fixture.main.discoveryBytes
        let discoveryInput = SyntheticSharedCacheDiscoveryFile(
            metadata: fixture.main.metadata,
            discoveryBytes: inputBytes
        )
        let plan = try Self.derive(fixture)

        inputBytes[0] = 0xfe
        fixture.main.metadata.size = 1
        fixture.main.discoveryBytes[0] = 0xff
        fixture.subcaches[0].metadata.inode = 42
        fixture.subcaches[0].discoveryBytes[88] = 0xfe

        XCTAssertEqual(discoveryInput.discoveryBytes, originalMainBytes)
        XCTAssertEqual(plan.mainMetadata, Self.threeFileFixture.main.metadata)
        XCTAssertEqual(plan.headerOrderFiles[0].discoveryBytes,
            originalMainBytes
        )
        XCTAssertEqual(
            plan.headerOrderFiles.dropFirst().map(\.discoveryBytes),
            originalSubcacheBytes
        )
        XCTAssertEqual(
            plan.headerOrderFiles.map(\.decodedSuffix),
            [Data(), Data(".2".utf8), Data(".1".utf8)]
        )
        XCTAssertEqual(
            plan.headerOrderFiles.map(\.suffixBase64URL),
            ["", Self.base64URL(Data(".2".utf8)),
             Self.base64URL(Data(".1".utf8))]
        )
    }

    func testCountOneAndSixtyFourBoundariesAndZeroSixtyFiveRefusals()
        throws
    {
        try Self.assertPlanMatchesFixture(
            Self.derive(Self.mainOnlyFixture),
            fixture: Self.mainOnlyFixture
        )

        let maximum = Self.makeFixture(
            suffixes: (1...63).map {
                Data(String(format: ".%02d", $0).utf8)
            }
        )
        let maximumPlan = try Self.derive(maximum)
        XCTAssertEqual(maximumPlan.headerOrderFiles.count, 64)
        XCTAssertEqual(
            maximumPlan.aggregateFileBytes,
            UInt64(Self.mainFileBytes) +
                UInt64(63) * UInt64(Self.subcacheFileBytes)
        )

        let tooMany = Self.makeFixtureAllowingInvalidSubcacheCount(
            suffixes: (1...64).map {
                Data(String(format: ".%02d", $0).utf8)
            }
        )
        Self.assertFixtureRefuses(
            tooMany,
            as: .cacheSetCount(
                .fileCount(actual: 65, minimum: 1, maximum: 64)
            )
        )
    }

    func testMetadataFileTypeLinkDatalessSparseAndSizeRefusals() {
        Self.assertFixtureMutationRefuses(
            { $0.main.metadata.mode = 0o040755 },
            as: .fileType(fileOrdinal: 0, mode: 0o040755)
        )
        Self.assertFixtureMutationRefuses(
            { $0.main.metadata.linkCount = 2 },
            as: .linkCount(fileOrdinal: 0, actual: 2)
        )
        Self.assertFixtureMutationRefuses(
            { $0.main.metadata.flags = 0x4000_0000 },
            as: .dataless(fileOrdinal: 0)
        )
        Self.assertFixtureMutationRefuses(
            { $0.main.metadata.extendedAttributeSupportMask = 0 },
            as: .sparseStateUnavailable(fileOrdinal: 0)
        )
        Self.assertFixtureMutationRefuses(
            { $0.main.metadata.extendedFlags = 1 },
            as: .sparseFile(fileOrdinal: 0)
        )
        Self.assertFixtureMutationRefuses(
            { $0.subcaches[0].metadata.mode = 0o040755 },
            as: .fileType(fileOrdinal: 1, mode: 0o040755)
        )
        Self.assertFixtureMutationRefuses(
            { $0.subcaches[0].metadata.size = 0 },
            as: .fileSize(fileOrdinal: 1, actual: 0)
        )
        Self.assertFixtureMutationRefuses(
            { $0.subcaches[0].metadata.size = 17_179_869_185 },
            as: .fileSize(fileOrdinal: 1, actual: 17_179_869_185)
        )
        Self.assertFixtureMutationRefuses(
            startingFrom: Self.makeFixture(
                suffixes: (1...4).map {
                    Data(String(format: ".%02d", $0).utf8)
                }
            ),
            {
                $0.main.metadata.size = 17_179_869_184
                Self.writeCodeSignatureEnd(
                    metadataBytes: 17_179_869_184,
                    into: &$0.main.discoveryBytes
                )
                for index in $0.subcaches.indices {
                    $0.subcaches[index].metadata.size = 17_179_869_184
                    Self.writeCodeSignatureEnd(
                        metadataBytes: 17_179_869_184,
                        into: &$0.subcaches[index].discoveryBytes
                    )
                }
            },
            as: .aggregateSize(
                .setLimit(
                    actual: 85_899_345_920,
                    maximum: 68_719_476_736
                )
            )
        )
    }

    func testCommonHeaderFormatMappingDiscoveryAndUUIDRefusals() {
        Self.assertFixtureMutationRefuses(
            { $0.main.discoveryBytes.replaceSubrange(0..<16, with:
                Data("dyld_v1   x86_64".utf8)
            ) },
            as: .cacheMagic(fileOrdinal: 0)
        )
        Self.assertFixtureMutationRefuses(
            { Self.writeUInt32LE(2, into: &$0.main.discoveryBytes, at: 216) },
            as: .cacheFormat(fileOrdinal: 0, field: .platform)
        )
        Self.assertFixtureMutationRefuses(
            { Self.writeUInt32LE(1 << 9, into: &$0.main.discoveryBytes,
                at: 220)
            },
            as: .cacheFormat(fileOrdinal: 0, field: .simulator)
        )
        Self.assertFixtureMutationRefuses(
            { Self.writeUInt32LE(1 << 10, into: &$0.main.discoveryBytes,
                at: 220)
            },
            as: .cacheFormat(fileOrdinal: 0, field: .locallyBuiltCache)
        )
        Self.assertFixtureMutationRefuses(
            { Self.writeUInt32LE((1 << 12) | (1 << 8),
                into: &$0.main.discoveryBytes, at: 220)
            },
            as: .cacheFormat(fileOrdinal: 0, field: .padding)
        )
        Self.assertFixtureMutationRefuses(
            { Self.writeUInt32LE((1 << 12) | (1 << 11),
                into: &$0.main.discoveryBytes, at: 220)
            },
            as: .cacheFormat(fileOrdinal: 0, field: .padding)
        )
        Self.assertFixtureMutationRefuses(
            { Self.writeUInt32LE(1, into: &$0.main.discoveryBytes, at: 220) },
            as: .cacheFormat(fileOrdinal: 0, field: .formatVersion)
        )
        Self.assertFixtureMutationRefuses(
            { Self.writeUInt32LE(1, into: &$0.main.discoveryBytes, at: 104) },
            as: .cacheFormat(fileOrdinal: 0, field: .cacheType)
        )
        Self.assertFixtureMutationRefuses(
            { Self.writeUInt32LE(2, into: &$0.main.discoveryBytes, at: 456) },
            as: .cacheFormat(fileOrdinal: 0, field: .cacheSubType)
        )
        Self.assertFixtureMutationRefuses(
            { Self.writeUInt32LE(553, into: &$0.main.discoveryBytes, at: 16) },
            as: .cacheMapping(fileOrdinal: 0, reason: .mappingOffset)
        )
        Self.assertFixtureMutationRefuses(
            { Self.writeUInt32LE(0, into: &$0.main.discoveryBytes, at: 20) },
            as: .cacheMapping(fileOrdinal: 0, reason: .mappingCount)
        )
        Self.assertFixtureMutationRefuses(
            { Self.writeUInt32LE(2, into: &$0.main.discoveryBytes, at: 316) },
            as: .cacheMapping(
                fileOrdinal: 0,
                reason: .mappingWithSlideCount
            )
        )
        Self.assertFixtureMutationRefuses(
            { Self.writeUInt32LE(UInt32.max, into: &$0.main.discoveryBytes,
                at: 448)
            },
            as: .cacheHeader(fileOrdinal: 0, reason: .imagesOffset)
        )
        Self.assertFixtureMutationRefuses(
            { $0.main.discoveryBytes.removeLast() },
            as: .discoveryBytes(
                fileOrdinal: 0,
                reason: .exactLength(
                    expected: UInt64(
                        Self.mappingTablesEnd +
                            2 * Self.modernSubcacheEntryBytes
                    ),
                    actual: Self.mappingTablesEnd +
                        2 * Self.modernSubcacheEntryBytes - 1
                )
            )
        )
        Self.assertFixtureMutationRefuses(
            { $0.main.discoveryBytes.append(0) },
            as: .discoveryBytes(
                fileOrdinal: 0,
                reason: .exactLength(
                    expected: UInt64(
                        Self.mappingTablesEnd +
                            2 * Self.modernSubcacheEntryBytes
                    ),
                    actual: Self.mappingTablesEnd +
                        2 * Self.modernSubcacheEntryBytes + 1
                )
            )
        )
        Self.assertFixtureMutationRefuses(
            { Self.writeUInt64LE(1, into: &$0.main.discoveryBytes, at: 48) },
            as: .cacheHeader(
                fileOrdinal: 0,
                reason: .codeSignatureEnd(
                    expected: UInt64(Self.mainFileBytes),
                    actual: UInt64(Self.mainFileBytes) -
                        Self.codeSignatureBytes + 1
                )
            )
        )
        Self.assertFixtureMutationRefuses(
            { $0.main.discoveryBytes.replaceSubrange(88..<104, with:
                Data(repeating: 0, count: 16))
            },
            as: .cacheUUID(fileOrdinal: 0, reason: .zero)
        )
    }

    func testSuffixEntryAndSubcacheMembershipRefusals() {
        Self.assertFixtureMutationRefuses(
            startingFrom: Self.makeFixture(suffixes: [Data(".1".utf8)]),
            { $0.main.discoveryBytes.removeLast(32) },
            as: .discoveryBytes(
                fileOrdinal: 0,
                reason: .exactLength(
                    expected: UInt64(
                        Self.mappingTablesEnd + Self.modernSubcacheEntryBytes
                    ),
                    actual: Self.mappingTablesEnd + 24
                )
            )
        )
        Self.assertEntryMutationRefuses(
            entry: 0,
            { entryOffset, bytes in
                bytes.replaceSubrange(
                    (entryOffset + 24)..<(entryOffset + 56),
                    with: Data(repeating: 0x41, count: 32)
                )
            },
            as: .cacheSuffix(entryIndex: 0, reason: .missingTerminator)
        )
        Self.assertEntryMutationRefuses(
            entry: 0,
            { entryOffset, bytes in
                bytes[entryOffset + 27] = 0x78
            },
            as: .cacheSuffix(entryIndex: 0, reason: .nonzeroPadding(index: 3))
        )
        Self.assertEntryMutationRefuses(
            entry: 0,
            { entryOffset, bytes in
                bytes[entryOffset + 25] = 0x2f
            },
            as: .cacheSuffix(entryIndex: 0, reason: .forbiddenByte(index: 1))
        )
        Self.assertFixtureMutationRefuses(
            {
                let first = Self.entryOffset(0) + 24
                let second = Self.entryOffset(1) + 24
                $0.main.discoveryBytes.replaceSubrange(
                    second..<(second + 32),
                    with: Data(
                        $0.main.discoveryBytes[first..<(first + 32)]
                    )
                )
            },
            as: .cacheSuffix(
                entryIndex: 1,
                reason: .duplicate(previousEntryIndex: 0)
            )
        )
        Self.assertEntryMutationRefuses(
            entry: 0,
            { entryOffset, bytes in
                bytes.replaceSubrange(
                    entryOffset..<(entryOffset + 16),
                    with: Data(repeating: 0, count: 16)
                )
            },
            as: .cacheUUID(fileOrdinal: 1, reason: .zero)
        )
        Self.assertEntryMutationRefuses(
            entry: 0,
            { entryOffset, bytes in
                bytes.replaceSubrange(
                    entryOffset..<(entryOffset + 16),
                    with: Self.uuid(byte: 0x01)
                )
            },
            as: .cacheUUID(fileOrdinal: 1, reason: .duplicate(
                previousOrdinal: 0
            ))
        )
        Self.assertEntryMutationRefuses(
            entry: 0,
            { entryOffset, bytes in
                Self.writeUInt64LE(0, into: &bytes, at: entryOffset + 16)
            },
            as: .cacheVMOffset(fileOrdinal: 1, reason: .zero)
        )
        Self.assertEntryMutationRefuses(
            entry: 1,
            { entryOffset, bytes in
                Self.writeUInt64LE(
                    0x0010_0000,
                    into: &bytes,
                    at: entryOffset + 16
                )
            },
            as: .cacheVMOffset(
                fileOrdinal: 2,
                reason: .nonIncreasing(
                    previous: 0x0020_0000,
                    actual: 0x0010_0000
                )
            )
        )

        var missing = Self.threeFileFixture
        missing.subcaches.removeLast()
        Self.assertRefuses(
            main: Self.discoveryFile(missing.main),
            subcaches: missing.subcaches.map(Self.discoveryFile),
            as: .cacheSetCount(.suppliedSubcaches(expected: 2, actual: 1))
        )

        var extra = Self.threeFileFixture
        extra.subcaches.append(extra.subcaches[0])
        Self.assertRefuses(
            main: Self.discoveryFile(extra.main),
            subcaches: extra.subcaches.map(Self.discoveryFile),
            as: .cacheSetCount(.suppliedSubcaches(expected: 2, actual: 3))
        )

        var reordered = Self.threeFileFixture
        reordered.subcaches.swapAt(0, 1)
        Self.assertRefuses(
            main: Self.discoveryFile(reordered.main),
            subcaches: reordered.subcaches.map(Self.discoveryFile),
            as: .cacheUUID(
                fileOrdinal: 1,
                reason: .entryHeaderMismatch(entryOrdinal: 1)
            )
        )

        Self.assertFixtureMutationRefuses(
            { $0.subcaches[0].discoveryBytes.replaceSubrange(88..<104,
                with: Self.uuid(byte: 0x7f))
            },
            as: .cacheUUID(
                fileOrdinal: 1,
                reason: .entryHeaderMismatch(entryOrdinal: 1)
            )
        )
    }

    func testSubcacheNestedTablesAndGlobalMappingAmbiguityRefuse() {
        Self.assertFixtureMutationRefuses(
            { Self.writeUInt32LE(
                UInt32(Self.mappingTablesEnd),
                into: &$0.subcaches[0].discoveryBytes,
                at: 392
            ) },
            as: .cacheHeader(fileOrdinal: 1, reason: .nestedSubcacheTable)
        )
        Self.assertFixtureMutationRefuses(
            { Self.writeUInt32LE(
                UInt32(Self.mappingTablesEnd),
                into: &$0.subcaches[0].discoveryBytes,
                at: 448
            ) },
            as: .cacheHeader(fileOrdinal: 1, reason: .nestedImageTable)
        )
        Self.assertFixtureMutationRefuses(
            {
                let entryOffset = Self.entryOffset(0)
                Self.writeUInt64LE(
                    0x0008_0000,
                    into: &$0.main.discoveryBytes,
                    at: entryOffset + 16
                )
                let overlapStart = Self.mainSharedRegionStart + 0x0008_0000
                Self.writeUInt64LE(
                    overlapStart,
                    into: &$0.subcaches[0].discoveryBytes,
                    at: 224
                )
                Self.writeUInt64LE(
                    overlapStart,
                    into: &$0.subcaches[0].discoveryBytes,
                    at: Self.headerBytes
                )
                Self.writeUInt64LE(
                    overlapStart,
                    into: &$0.subcaches[0].discoveryBytes,
                    at: Self.mappingWithSlideOffset
                )
            },
            as: .cacheSubcache(
                fileOrdinal: 1,
                reason: .globalMappingOverlap(fileOrdinal: 1, row: 0)
            )
        )
    }

    func testMixedFailuresFreezeExactDeterministicOrder() {
        Self.assertFixtureMutationRefuses(
            {
                $0.main.metadata.mode = 0o040755
                $0.main.discoveryBytes[0] = 0xff
            },
            as: .fileType(fileOrdinal: 0, mode: 0o040755)
        )
        Self.assertFixtureMutationRefuses(
            {
                $0.main.discoveryBytes[0] = 0xff
                Self.writeUInt32LE(0, into: &$0.main.discoveryBytes, at: 20)
            },
            as: .cacheMagic(fileOrdinal: 0)
        )
        Self.assertFixtureMutationRefuses(
            {
                Self.writeUInt32LE(0, into: &$0.main.discoveryBytes, at: 20)
                $0.main.discoveryBytes.removeLast()
            },
            as: .cacheMapping(fileOrdinal: 0, reason: .mappingCount)
        )
        Self.assertFixtureMutationRefuses(
            {
                $0.main.discoveryBytes.removeLast()
                $0.subcaches.removeLast()
            },
            as: .discoveryBytes(
                fileOrdinal: 0,
                reason: .exactLength(
                    expected: UInt64(
                        Self.mappingTablesEnd +
                            2 * Self.modernSubcacheEntryBytes
                    ),
                    actual: Self.mappingTablesEnd +
                        2 * Self.modernSubcacheEntryBytes - 1
                )
            )
        )
        Self.assertFixtureMutationRefuses(
            {
                let first = Self.entryOffset(0) + 24
                let second = Self.entryOffset(1) + 24
                $0.main.discoveryBytes.replaceSubrange(
                    second..<(second + 32),
                    with: Data(
                        $0.main.discoveryBytes[first..<(first + 32)]
                    )
                )
                $0.subcaches[0].discoveryBytes.replaceSubrange(
                    88..<104,
                    with: Self.uuid(byte: 0x7f)
                )
            },
            as: .cacheSuffix(
                entryIndex: 1,
                reason: .duplicate(previousEntryIndex: 0)
            )
        )
        Self.assertFixtureMutationRefuses(
            {
                $0.subcaches[0].discoveryBytes.replaceSubrange(
                    88..<104,
                    with: Self.uuid(byte: 0x7f)
                )
                Self.writeUInt32LE(
                    UInt32(Self.mappingTablesEnd),
                    into: &$0.subcaches[0].discoveryBytes,
                    at: 448
                )
            },
            as: .cacheUUID(
                fileOrdinal: 1,
                reason: .entryHeaderMismatch(entryOrdinal: 1)
            )
        )
        Self.assertFixtureMutationRefuses(
            {
                $0.subcaches[0].discoveryBytes[0] = 0xff
                $0.subcaches[1].metadata.mode = 0o040755
            },
            as: .fileType(fileOrdinal: 2, mode: 0o040755)
        )
    }

    func testRuntimeNoGoAndAllAuthorityFlagsAreFalse() throws {
        let plan = try Self.derive(Self.threeFileFixture)

        XCTAssertEqual(plan.runtimeDecision, .noGo)
        Self.assertNoAuthority(plan)
    }
}

private extension SyntheticSharedCacheSetPlanTests {
    struct FixtureFile {
        var metadata: SyntheticCaptureFileMetadata
        var discoveryBytes: Data
    }

    struct Fixture {
        var main: FixtureFile
        var subcaches: [FixtureFile]
        var suffixes: [Data]
        var uuids: [Data]
        var cacheVMOffsets: [UInt64]
    }

    static let headerBytes = 552
    static let mappingInfoBytes = 32
    static let mappingAndSlideInfoBytes = 56
    static let modernSubcacheEntryBytes = 56
    static let mappingCount = 1
    static let mappingWithSlideOffset =
        headerBytes + mappingCount * mappingAndSlideInfoBytes
    static let mappingTablesEnd =
        mappingWithSlideOffset +
        mappingCount * mappingAndSlideInfoBytes
    static let mainSharedRegionStart: UInt64 = 0x0000_0001_8000_0000
    static let mainMappingBytes: UInt64 = 0x0010_0000
    static let mainFileBytes: Int64 = 8 * 1_048_576
    static let subcacheFileBytes: Int64 = 4 * 1_048_576
    static let codeSignatureBytes: UInt64 = 4_096
    static let currentMagic = Data([
        0x64, 0x79, 0x6c, 0x64, 0x5f, 0x76, 0x31, 0x20,
        0x20, 0x20, 0x61, 0x72, 0x6d, 0x36, 0x34, 0x00,
    ])

    static var mainOnlyFixture: Fixture {
        makeFixture(suffixes: [])
    }

    static var threeFileFixture: Fixture {
        makeFixture(suffixes: [Data(".2".utf8), Data(".1".utf8)])
    }

    static func makeFixture(suffixes: [Data]) -> Fixture {
        precondition(suffixes.count <= 63)
        let mainUUID = uuid(byte: 0x01)
        let subcacheUUIDs = suffixes.indices.map {
            uuid(byte: UInt8($0 + 2))
        }
        let cacheVMOffsets = suffixes.indices.map {
            UInt64(($0 + 1) * 0x0020_0000)
        }
        let sharedRegionSize = max(
            UInt64(0x0020_0000),
            UInt64(suffixes.count + 1) * 0x0020_0000
        )

        let mainDiscovery = makeMainDiscovery(
            metadataBytes: mainFileBytes,
            headerUUID: mainUUID,
            suffixes: suffixes,
            subcacheUUIDs: subcacheUUIDs,
            cacheVMOffsets: cacheVMOffsets,
            sharedRegionSize: sharedRegionSize
        )
        let main = FixtureFile(
            metadata: metadata(size: mainFileBytes, ordinal: 0),
            discoveryBytes: mainDiscovery
        )
        let subcaches = suffixes.indices.map { index in
            FixtureFile(
                metadata: metadata(
                    size: subcacheFileBytes,
                    ordinal: index + 1
                ),
                discoveryBytes: makeSubcacheDiscovery(
                    metadataBytes: subcacheFileBytes,
                    headerUUID: subcacheUUIDs[index],
                    cacheVMOffset: cacheVMOffsets[index],
                    sharedRegionSize: sharedRegionSize
                )
            )
        }
        return Fixture(
            main: main,
            subcaches: subcaches,
            suffixes: suffixes,
            uuids: [mainUUID] + subcacheUUIDs,
            cacheVMOffsets: [0] + cacheVMOffsets
        )
    }

    static func makeMainDiscovery(
        metadataBytes: Int64,
        headerUUID: Data,
        suffixes: [Data],
        subcacheUUIDs: [Data],
        cacheVMOffsets: [UInt64],
        sharedRegionSize: UInt64
    ) -> Data {
        let subcacheTableOffset = mappingTablesEnd
        let byteCount = subcacheTableOffset +
            suffixes.count * modernSubcacheEntryBytes
        var bytes = Data(repeating: 0, count: byteCount)
        writeCommonHeader(
            into: &bytes,
            metadataBytes: metadataBytes,
            headerUUID: headerUUID,
            sharedRegionStart: mainSharedRegionStart,
            sharedRegionSize: sharedRegionSize
        )
        writeUInt32LE(
            UInt32(mappingTablesEnd),
            into: &bytes,
            at: 448
        )
        writeUInt32LE(0, into: &bytes, at: 452)
        writeUInt64LE(
            UInt64(mappingTablesEnd),
            into: &bytes,
            at: 136
        )
        writeUInt64LE(0, into: &bytes, at: 144)
        writeUInt32LE(
            UInt32(subcacheTableOffset),
            into: &bytes,
            at: 392
        )
        writeUInt32LE(
            UInt32(suffixes.count),
            into: &bytes,
            at: 396
        )

        for index in suffixes.indices {
            let entryOffset = subcacheTableOffset +
                index * modernSubcacheEntryBytes
            bytes.replaceSubrange(
                entryOffset..<(entryOffset + 16),
                with: subcacheUUIDs[index]
            )
            writeUInt64LE(
                cacheVMOffsets[index],
                into: &bytes,
                at: entryOffset + 16
            )
            let suffix = suffixes[index]
            precondition((1...31).contains(suffix.count))
            bytes.replaceSubrange(
                (entryOffset + 24)..<(entryOffset + 24 + suffix.count),
                with: suffix
            )
        }
        return bytes
    }

    static func makeSubcacheDiscovery(
        metadataBytes: Int64,
        headerUUID: Data,
        cacheVMOffset: UInt64,
        sharedRegionSize: UInt64
    ) -> Data {
        var bytes = Data(repeating: 0, count: mappingTablesEnd)
        writeCommonHeader(
            into: &bytes,
            metadataBytes: metadataBytes,
            headerUUID: headerUUID,
            sharedRegionStart: mainSharedRegionStart + cacheVMOffset,
            sharedRegionSize: sharedRegionSize
        )
        return bytes
    }

    static func writeCommonHeader(
        into bytes: inout Data,
        metadataBytes: Int64,
        headerUUID: Data,
        sharedRegionStart: UInt64,
        sharedRegionSize: UInt64
    ) {
        precondition(bytes.count >= mappingTablesEnd)
        precondition(headerUUID.count == 16)
        precondition(metadataBytes > Int64(codeSignatureBytes))
        bytes.replaceSubrange(0..<16, with: currentMagic)
        writeUInt32LE(UInt32(headerBytes), into: &bytes, at: 16)
        writeUInt32LE(UInt32(mappingCount), into: &bytes, at: 20)
        writeUInt64LE(
            UInt64(metadataBytes) - codeSignatureBytes,
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
        writeUInt32LE(UInt32(mappingCount), into: &bytes, at: 316)
        writeUInt32LE(1, into: &bytes, at: 456)

        writeUInt64LE(sharedRegionStart, into: &bytes, at: headerBytes)
        writeUInt64LE(mainMappingBytes, into: &bytes, at: headerBytes + 8)
        writeUInt64LE(0, into: &bytes, at: headerBytes + 16)
        writeUInt32LE(5, into: &bytes, at: headerBytes + 24)
        writeUInt32LE(5, into: &bytes, at: headerBytes + 28)

        writeUInt64LE(
            sharedRegionStart,
            into: &bytes,
            at: mappingWithSlideOffset
        )
        writeUInt64LE(
            mainMappingBytes,
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

    static func metadata(
        size: Int64,
        ordinal: Int
    ) -> SyntheticCaptureFileMetadata {
        SyntheticCaptureFileMetadata(
            device: 7,
            inode: UInt64(10_000 + ordinal),
            mode: 0o100644,
            linkCount: 1,
            userID: 501,
            groupID: 20,
            size: size,
            blockCount: size / 512,
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

    static func derive(
        _ fixture: Fixture
    ) throws -> SyntheticSharedCacheSetPlanComparison {
        try SyntheticSharedCacheSetPlanVerifier.compare(
            sourceProfile: .reviewed,
            main: discoveryFile(fixture.main),
            subcaches: fixture.subcaches.map(discoveryFile)
        )
    }

    static func discoveryFile(
        _ file: FixtureFile
    ) -> SyntheticSharedCacheDiscoveryFile {
        SyntheticSharedCacheDiscoveryFile(
            metadata: file.metadata,
            discoveryBytes: file.discoveryBytes
        )
    }

    static func assertPlanMatchesFixture(
        _ plan: SyntheticSharedCacheSetPlanComparison,
        fixture: Fixture,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(plan.sourceProfile, .reviewed, file: file, line: line)
        XCTAssertEqual(
            plan.mainMetadata,
            fixture.main.metadata,
            file: file,
            line: line
        )
        XCTAssertEqual(
            plan.mainHeader,
            plan.headerOrderFiles[0].header,
            file: file,
            line: line
        )
        XCTAssertEqual(
            plan.aggregateFileBytes,
            UInt64(fixture.main.metadata.size) +
                UInt64(fixture.subcaches.count) *
                    UInt64(subcacheFileBytes),
            file: file,
            line: line
        )
        XCTAssertEqual(
            plan.headerOrderFiles.count,
            fixture.subcaches.count + 1,
            file: file,
            line: line
        )

        let expectedFiles = [fixture.main] + fixture.subcaches
        for (ordinal, fact) in plan.headerOrderFiles.enumerated() {
            let suffix = ordinal == 0 ? Data() : fixture.suffixes[ordinal - 1]
            XCTAssertEqual(fact.ordinal, ordinal, file: file, line: line)
            XCTAssertEqual(
                fact.metadata,
                expectedFiles[ordinal].metadata,
                file: file,
                line: line
            )
            XCTAssertEqual(
                fact.fileBytes,
                UInt64(expectedFiles[ordinal].metadata.size),
                file: file,
                line: line
            )
            XCTAssertEqual(fact.decodedSuffix, suffix, file: file, line: line)
            XCTAssertEqual(fact.suffixBytes, suffix, file: file, line: line)
            XCTAssertEqual(
                fact.suffixByteCount,
                UInt64(suffix.count),
                file: file,
                line: line
            )
            XCTAssertEqual(
                fact.suffixBase64URL,
                base64URL(suffix),
                file: file,
                line: line
            )
            XCTAssertEqual(
                fact.headerUUID,
                hex(fixture.uuids[ordinal]),
                file: file,
                line: line
            )
            XCTAssertEqual(
                fact.cacheVMOffset,
                fixture.cacheVMOffsets[ordinal],
                file: file,
                line: line
            )
            XCTAssertEqual(
                fact.discoveryBytes,
                expectedFiles[ordinal].discoveryBytes,
                file: file,
                line: line
            )
            assertHeaderFacts(
                fact.header,
                ordinal: ordinal,
                fixture: fixture,
                file: file,
                line: line
            )
        }
        assertNoAuthority(plan, file: file, line: line)
    }

    static func assertHeaderFacts(
        _ header: SyntheticSharedCacheSetPlanComparison.HeaderFacts,
        ordinal: Int,
        fixture: Fixture,
        file: StaticString,
        line: UInt
    ) {
        let metadata = ordinal == 0
            ? fixture.main.metadata
            : fixture.subcaches[ordinal - 1].metadata
        let vmOffset = fixture.cacheVMOffsets[ordinal]
        XCTAssertEqual(header.mappingOffset, UInt32(headerBytes),
            file: file,
            line: line
        )
        XCTAssertEqual(header.mappingCount, UInt32(mappingCount),
            file: file,
            line: line
        )
        XCTAssertEqual(
            header.mappingWithSlideOffset,
            UInt32(mappingWithSlideOffset),
            file: file,
            line: line
        )
        XCTAssertEqual(
            header.mappingWithSlideCount,
            UInt32(mappingCount),
            file: file,
            line: line
        )
        XCTAssertEqual(header.mappingTablesEnd, UInt64(mappingTablesEnd),
            file: file,
            line: line
        )
        XCTAssertEqual(
            header.codeSignatureOffset,
            UInt64(metadata.size) - codeSignatureBytes,
            file: file,
            line: line
        )
        XCTAssertEqual(header.codeSignatureSize, codeSignatureBytes,
            file: file,
            line: line
        )
        XCTAssertEqual(
            header.sharedRegionStart,
            mainSharedRegionStart + vmOffset,
            file: file,
            line: line
        )
        XCTAssertEqual(
            header.sharedRegionSize,
            max(
                UInt64(0x0020_0000),
                UInt64(fixture.suffixes.count + 1) * 0x0020_0000
            ),
            file: file,
            line: line
        )
        XCTAssertEqual(header.mappings.count, 1, file: file, line: line)
        XCTAssertEqual(
            header.mappings[0].address,
            mainSharedRegionStart + vmOffset,
            file: file,
            line: line
        )
        XCTAssertEqual(header.mappings[0].size, mainMappingBytes,
            file: file,
            line: line
        )
        XCTAssertEqual(header.mappings[0].fileOffset, 0,
            file: file,
            line: line
        )
        XCTAssertEqual(header.mappings[0].maximumProtection, 5,
            file: file,
            line: line
        )
        XCTAssertEqual(header.mappings[0].initialProtection, 5,
            file: file,
            line: line
        )
        XCTAssertEqual(header.mappings[0].slideInfoOffset, 0,
            file: file,
            line: line
        )
        XCTAssertEqual(header.mappings[0].slideInfoSize, 0,
            file: file,
            line: line
        )
        XCTAssertEqual(header.mappings[0].flags, 0,
            file: file,
            line: line
        )

        if ordinal == 0 {
            let tableEnd = mappingTablesEnd +
                fixture.suffixes.count * modernSubcacheEntryBytes
            XCTAssertEqual(header.imagesOffset, UInt64(mappingTablesEnd),
                file: file,
                line: line
            )
            XCTAssertEqual(header.imagesCount, 0, file: file, line: line)
            XCTAssertEqual(header.imagesTextOffset, UInt64(mappingTablesEnd),
                file: file,
                line: line
            )
            XCTAssertEqual(header.imagesTextCount, 0,
                file: file,
                line: line
            )
            XCTAssertEqual(
                header.subCacheArrayOffset,
                UInt32(mappingTablesEnd),
                file: file,
                line: line
            )
            XCTAssertEqual(
                header.subCacheArrayCount,
                UInt32(fixture.suffixes.count),
                file: file,
                line: line
            )
            XCTAssertEqual(header.subcacheTableEnd, UInt64(tableEnd),
                file: file,
                line: line
            )
        } else {
            XCTAssertEqual(header.imagesOffset, 0, file: file, line: line)
            XCTAssertEqual(header.imagesCount, 0, file: file, line: line)
            XCTAssertEqual(header.imagesTextOffset, 0, file: file, line: line)
            XCTAssertEqual(header.imagesTextCount, 0, file: file, line: line)
            XCTAssertEqual(header.subCacheArrayOffset, 0,
                file: file,
                line: line
            )
            XCTAssertEqual(header.subCacheArrayCount, 0,
                file: file,
                line: line
            )
            XCTAssertEqual(
                header.subcacheTableEnd,
                UInt64(mappingTablesEnd),
                file: file,
                line: line
            )
        }
    }

    static func assertNoAuthority(
        _ plan: SyntheticSharedCacheSetPlanComparison,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(plan.runtimeDecision, .noGo, file: file, line: line)
        XCTAssertFalse(plan.canExecute, file: file, line: line)
        XCTAssertFalse(plan.canSpawn, file: file, line: line)
        XCTAssertFalse(plan.canAccessNetwork, file: file, line: line)
        XCTAssertFalse(plan.canConsumePack, file: file, line: line)
        XCTAssertFalse(plan.canMutateFileSystem, file: file, line: line)
        XCTAssertFalse(plan.canImportGitObjects, file: file, line: line)
        XCTAssertFalse(plan.canBuild, file: file, line: line)
        XCTAssertFalse(plan.canLoadModel, file: file, line: line)
        XCTAssertFalse(plan.canReserveOutput, file: file, line: line)
        XCTAssertFalse(plan.canPublish, file: file, line: line)
    }

    static func assertFixtureMutationRefuses(
        startingFrom fixture: Fixture = threeFileFixture,
        _ mutate: (inout Fixture) -> Void,
        as expected: SyntheticSharedCacheSetPlanFailure,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        var mutated = fixture
        mutate(&mutated)
        assertFixtureRefuses(mutated, as: expected, file: file, line: line)
    }

    static func assertEntryMutationRefuses(
        entry: Int,
        _ mutate: (Int, inout Data) -> Void,
        as expected: SyntheticSharedCacheSetPlanFailure,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        assertFixtureMutationRefuses(
            { fixture in
                mutate(entryOffset(entry), &fixture.main.discoveryBytes)
            },
            as: expected,
            file: file,
            line: line
        )
    }

    static func assertFixtureRefuses(
        _ fixture: Fixture,
        as expected: SyntheticSharedCacheSetPlanFailure,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        assertRefuses(
            main: discoveryFile(fixture.main),
            subcaches: fixture.subcaches.map(discoveryFile),
            as: expected,
            file: file,
            line: line
        )
    }

    static func assertRefuses(
        main: SyntheticSharedCacheDiscoveryFile,
        subcaches: [SyntheticSharedCacheDiscoveryFile],
        as expected: SyntheticSharedCacheSetPlanFailure,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(
            try SyntheticSharedCacheSetPlanVerifier.compare(
                sourceProfile: .reviewed,
                main: main,
                subcaches: subcaches
            ),
            file: file,
            line: line
        ) { error in
            XCTAssertEqual(
                error as? SyntheticSharedCacheSetPlanFailure,
                expected,
                file: file,
                line: line
            )
        }
    }

    static func makeFixtureAllowingInvalidSubcacheCount(
        suffixes: [Data]
    ) -> Fixture {
        precondition(suffixes.count <= 64)
        let mainUUID = uuid(byte: 0x01)
        let subcacheUUIDs = suffixes.indices.map {
            uuid(byte: UInt8($0 + 2))
        }
        let cacheVMOffsets = suffixes.indices.map {
            UInt64(($0 + 1) * 0x0020_0000)
        }
        let sharedRegionSize = max(
            UInt64(0x0020_0000),
            UInt64(suffixes.count + 1) * 0x0020_0000
        )
        let main = FixtureFile(
            metadata: metadata(size: mainFileBytes, ordinal: 0),
            discoveryBytes: makeMainDiscovery(
                metadataBytes: mainFileBytes,
                headerUUID: mainUUID,
                suffixes: suffixes,
                subcacheUUIDs: subcacheUUIDs,
                cacheVMOffsets: cacheVMOffsets,
                sharedRegionSize: sharedRegionSize
            )
        )
        let subcaches = suffixes.indices.map { index in
            FixtureFile(
                metadata: metadata(
                    size: subcacheFileBytes,
                    ordinal: index + 1
                ),
                discoveryBytes: makeSubcacheDiscovery(
                    metadataBytes: subcacheFileBytes,
                    headerUUID: subcacheUUIDs[index],
                    cacheVMOffset: cacheVMOffsets[index],
                    sharedRegionSize: sharedRegionSize
                )
            )
        }
        return Fixture(
            main: main,
            subcaches: subcaches,
            suffixes: suffixes,
            uuids: [mainUUID] + subcacheUUIDs,
            cacheVMOffsets: [0] + cacheVMOffsets
        )
    }

    static func entryOffset(_ entry: Int) -> Int {
        mappingTablesEnd + entry * modernSubcacheEntryBytes
    }

    static func writeCodeSignatureEnd(
        metadataBytes: Int64,
        into bytes: inout Data
    ) {
        writeUInt64LE(
            UInt64(metadataBytes) - codeSignatureBytes,
            into: &bytes,
            at: 40
        )
        writeUInt64LE(codeSignatureBytes, into: &bytes, at: 48)
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

    /*
     Construction-seal and forbidden-conversion compiler probes intentionally
     remain outside the compiled XCTest suite. They should fail to compile:

     _ = SyntheticSharedCacheSetPlanComparison(
         sourceProfile: .reviewed,
         mainMetadata: Self.mainOnlyFixture.main.metadata,
         mainHeader: fatalError(),
         headerOrderFiles: [],
         aggregateFileBytes: 0,
         seal: .verified
     )
     let plan = try Self.derive(Self.mainOnlyFixture)
     _ = SyntheticSharedCacheFileRecord(plan)
     _ = SyntheticSharedCacheSetIdentityEvidence(plan)
     _ = SyntheticSharedCacheImageContentIdentityEvidence(plan)
     _ = SyntheticSmallArtifactCaptureComparison(plan)
     _ = AnchoredRuntimeClosureExpectationDocument(plan)
     _ = ExecutionIdentityComparisonEvidence(plan)
     _ = FileImageExecutionIdentityComparisonEvidence(plan)
     _ = AdmittedFile(plan)

     The verifier API is `main` plus `subcaches`, so a zero-main call is
     intentionally unrepresentable rather than a runtime assertion:

     _ = try SyntheticSharedCacheSetPlanVerifier.compare(
         sourceProfile: .reviewed,
         subcaches: []
     )
     */
}
