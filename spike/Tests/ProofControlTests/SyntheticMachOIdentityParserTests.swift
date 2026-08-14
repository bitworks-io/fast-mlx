import CryptoKit
import Foundation
import XCTest
@testable import ProofControl

final class SyntheticMachOIdentityParserTests: XCTestCase {
    func testCanonicalSignedThinArm64MachOProducesOnlyInertComparison()
        throws
    {
        let primary = Self.codeDirectory(
            hashType: 2,
            flags: 0,
            signingIdentifier: Data("com.example.git".utf8),
            teamIdentifier: Data("TEAM123456".utf8)
        )
        let alternate = Self.codeDirectory(
            version: 0x20600,
            hashType: 4,
            flags: 0,
            signingIdentifier: Data("com.example.git.alt".utf8),
            teamIdentifier: Data()
        )
        let cms = Self.genericBlob(
            magic: Self.csMagicBlobWrapper,
            payload: Data([0x30, 0x03, 0x02, 0x01, 0x01])
        )
        let signature = Self.superBlob(entries: [
            (0, primary),
            (2, Self.genericBlob(
                magic: Self.csMagicRequirements,
                payload: Data([0x01, 0x02, 0x03])
            )),
            (0x1000, alternate),
            (0x10000, cms),
        ])
        let fixture = Self.machO(signatureRegion: signature)
        let effects = FakeOperationalEffects()

        let comparison = try SyntheticMachOIdentityParser.parse(fixture)

        XCTAssertEqual(comparison.retainedFileBytes, fixture)
        XCTAssertEqual(
            comparison.fileSHA256,
            Self.sha256Hex(fixture)
        )
        XCTAssertEqual(comparison.machHeaderMagic, 0xfeedfacf)
        XCTAssertEqual(comparison.cpuType, 0x0100000c)
        XCTAssertEqual(comparison.cpuSubtype, 2)
        XCTAssertEqual(comparison.fileType, 2)
        XCTAssertEqual(comparison.headerFlags, 0x00200085)
        XCTAssertEqual(comparison.loadCommandCount, 2)
        XCTAssertEqual(comparison.loadCommandBytes.count, 40)
        XCTAssertEqual(
            comparison.loadCommandsSHA256,
            Self.sha256Hex(comparison.loadCommandBytes)
        )
        XCTAssertEqual(
            comparison.machOUUID,
            Data(Self.canonicalUUID)
        )
        XCTAssertEqual(comparison.codeSignatureRegion, signature)
        XCTAssertEqual(
            comparison.codeSignatureRegionSHA256,
            Self.sha256Hex(signature)
        )
        XCTAssertEqual(
            comparison.codeDirectories.map(\.slot),
            [0, 0x1000]
        )
        XCTAssertEqual(comparison.codeDirectories[0].blob, primary)
        XCTAssertEqual(
            comparison.codeDirectories[0].blobSHA256,
            Self.sha256Hex(primary)
        )
        XCTAssertEqual(comparison.codeDirectories[0].version, 0x20200)
        XCTAssertEqual(comparison.codeDirectories[0].flags, 0)
        XCTAssertEqual(comparison.codeDirectories[0].hashType, 2)
        XCTAssertEqual(comparison.codeDirectories[0].hashSize, 32)
        XCTAssertEqual(
            comparison.codeDirectories[0].signingIdentifier,
            Data("com.example.git".utf8)
        )
        XCTAssertEqual(
            comparison.codeDirectories[0].teamIdentifier,
            Data("TEAM123456".utf8)
        )
        XCTAssertEqual(comparison.codeDirectories[1].blob, alternate)
        XCTAssertEqual(comparison.codeDirectories[1].version, 0x20600)
        XCTAssertEqual(comparison.codeDirectories[1].hashType, 4)
        XCTAssertEqual(comparison.codeDirectories[1].hashSize, 48)
        XCTAssertEqual(comparison.cmsBlob, cms)
        XCTAssertEqual(
            comparison.cmsBlobSHA256,
            Self.sha256Hex(cms)
        )
        XCTAssertFalse(comparison.isAdHoc)
        XCTAssertFalse(comparison.canExecute)
        XCTAssertFalse(comparison.canSpawn)
        XCTAssertFalse(comparison.canAccessNetwork)
        XCTAssertFalse(comparison.canConsumePack)
        XCTAssertFalse(comparison.canMutateFileSystem)
        XCTAssertFalse(comparison.canImportGitObjects)
        XCTAssertFalse(comparison.canBuild)
        XCTAssertFalse(comparison.canLoadModel)
        XCTAssertFalse(comparison.canReserveOutput)
        XCTAssertFalse(comparison.canPublish)

        var prefixed = Data([0xff])
        prefixed.append(fixture)
        prefixed.append(0xee)
        let slicedComparison = try SyntheticMachOIdentityParser.parse(
            prefixed[1..<(1 + fixture.count)]
        )
        XCTAssertEqual(slicedComparison.retainedFileBytes, fixture)
        prefixed[1] = 0
        XCTAssertEqual(slicedComparison.retainedFileBytes, fixture)
        XCTAssertEqual(effects, .zero)
    }

    func testAllPinnedXNUHashTypesAndAdHocWithoutCMSRemainInert()
        throws
    {
        let specs: [(UInt32, UInt8)] = [
            (0, 1),
            (0x1000, 2),
            (0x1001, 3),
            (0x1002, 4),
            (0x1003, 2),
            (0x1004, 4),
        ]
        let signature = Self.superBlob(
            entries: specs.map { slot, hashType in
                (
                    slot,
                    Self.codeDirectory(
                        hashType: hashType,
                        flags: Self.csAdHoc,
                        signingIdentifier: Data(
                            "synthetic.hash.\(hashType)".utf8
                        ),
                        teamIdentifier: Data()
                    )
                )
            }
        )
        let effects = FakeOperationalEffects()

        let comparison = try SyntheticMachOIdentityParser.parse(
            Self.machO(signatureRegion: signature)
        )

        XCTAssertEqual(
            comparison.codeDirectories.map(\.hashType),
            [1, 2, 3, 4, 2, 4]
        )
        XCTAssertEqual(
            comparison.codeDirectories.map(\.hashSize),
            [20, 32, 20, 48, 32, 48]
        )
        XCTAssertNil(comparison.cmsBlob)
        XCTAssertEqual(
            comparison.cmsBlobSHA256,
            String(repeating: "0", count: 64)
        )
        XCTAssertTrue(comparison.isAdHoc)
        XCTAssertFalse(comparison.canExecute)
        XCTAssertFalse(comparison.canMutateFileSystem)
        XCTAssertEqual(effects, .zero)
    }

    func testRejectsHeaderAndLoadCommandFailuresBeforeEffects() {
        let canonical = Self.signedFixture()
        var cases: [(Data, SyntheticMachOIdentityFailure)] = [
            (Data(), .fileBounds),
            (
                Data(
                    repeating: 0,
                    count:
                        SyntheticMachOIdentityParser.maximumFileBytes + 1
                ),
                .fileBounds
            ),
            (
                Self.replacingUInt32LE(
                    in: canonical,
                    at: 0,
                    with: 0xcafebabe
                ),
                .unsupportedContainer
            ),
            (
                Self.replacingUInt32LE(
                    in: canonical,
                    at: 0,
                    with: 0xfeedface
                ),
                .unsupportedContainer
            ),
            (
                Self.replacingUInt32LE(
                    in: canonical,
                    at: 0,
                    with: 0xcffaedfe
                ),
                .unsupportedContainer
            ),
            (
                Self.replacingUInt32LE(
                    in: canonical,
                    at: 4,
                    with: 0x01000007
                ),
                .unsupportedArchitecture
            ),
            (
                Self.replacingUInt32LE(
                    in: canonical,
                    at: 12,
                    with: 6
                ),
                .unsupportedFileType
            ),
            (
                Self.replacingUInt32LE(
                    in: canonical,
                    at: 16,
                    with: 0
                ),
                .loadCommandCount
            ),
            (
                Self.replacingUInt32LE(
                    in: canonical,
                    at: 16,
                    with:
                        SyntheticMachOIdentityParser
                            .maximumLoadCommandCount + 1
                ),
                .loadCommandCount
            ),
            (
                Self.replacingUInt32LE(
                    in: canonical,
                    at: 20,
                    with: 0
                ),
                .loadCommandBounds
            ),
            (
                Self.replacingUInt32LE(
                    in: canonical,
                    at: 16,
                    with: 1
                ),
                .malformedLoadCommand
            ),
            (
                Self.replacingUInt32LE(
                    in: canonical,
                    at: 20,
                    with:
                        UInt32(
                            SyntheticMachOIdentityParser
                                .maximumLoadCommandBytes + 8
                        )
                ),
                .loadCommandBounds
            ),
            (
                Self.replacingUInt32LE(
                    in: canonical,
                    at: 28,
                    with: 1
                ),
                .malformedHeader
            ),
            (
                Self.replacingUInt32LE(
                    in: canonical,
                    at: 36,
                    with: 7
                ),
                .malformedLoadCommand
            ),
            (
                Self.replacingUInt32LE(
                    in: canonical,
                    at: 36,
                    with: 12
                ),
                .malformedLoadCommand
            ),
            (
                Self.replacingUInt32LE(
                    in: canonical,
                    at: 36,
                    with: 48
                ),
                .malformedLoadCommand
            ),
            (
                Self.machO(
                    signatureRegion: Self.signedSignature(),
                    uuidCommands: []
                ),
                .missingUUID
            ),
            (
                Self.machO(
                    signatureRegion: Self.signedSignature(),
                    uuidCommands: [
                        Self.canonicalUUID,
                        Array(repeating: 0x44, count: 16),
                    ]
                ),
                .duplicateUUID
            ),
            (
                Self.machO(
                    signatureRegion: Self.signedSignature(),
                    uuidCommands: [Array(repeating: 0, count: 16)]
                ),
                .zeroUUID
            ),
            (
                Self.machO(
                    signatureRegion: Self.signedSignature(),
                    codeSignatureCommandCount: 0
                ),
                .missingCodeSignature
            ),
            (
                Self.machO(
                    signatureRegion: Self.signedSignature(),
                    codeSignatureCommandCount: 2
                ),
                .duplicateCodeSignature
            ),
            (
                Self.machO(
                    signatureRegion: Self.signedSignature(),
                    includeDyldEnvironment: true
                ),
                .unsupportedDyldEnvironment
            ),
        ]

        var overlappingSignature = canonical
        Self.writeUInt32LE(&overlappingSignature, at: 64, value: 32)
        cases.append((overlappingSignature, .codeSignatureBounds))

        var zeroSignature = canonical
        Self.writeUInt32LE(&zeroSignature, at: 68, value: 0)
        cases.append((zeroSignature, .codeSignatureBounds))

        var outOfRangeSignature = canonical
        Self.writeUInt32LE(
            &outOfRangeSignature,
            at: 68,
            value: UInt32(canonical.count)
        )
        cases.append((outOfRangeSignature, .codeSignatureBounds))

        for (index, testCase) in cases.enumerated() {
            assertRejected(
                testCase.0,
                as: testCase.1,
                context: "header/load case \(index)"
            )
        }
    }

    func testRejectsSuperBlobCodeDirectoryAndCMSFailuresBeforeEffects() {
        let primary = Self.codeDirectory(
            hashType: 2,
            flags: 0,
            signingIdentifier: Data("com.example.git".utf8),
            teamIdentifier: Data("TEAM123456".utf8)
        )
        let alternate = Self.codeDirectory(
            hashType: 4,
            flags: 0,
            signingIdentifier: Data("com.example.git.alt".utf8),
            teamIdentifier: Data()
        )
        let cms = Self.genericBlob(
            magic: Self.csMagicBlobWrapper,
            payload: Data([0x30, 0x00])
        )

        var badSuperMagic = Self.superBlob(entries: [
            (0, primary),
            (0x10000, cms),
        ])
        Self.writeUInt32BE(&badSuperMagic, at: 0, value: 0)

        var badSuperLength = Self.superBlob(entries: [
            (0, primary),
            (0x10000, cms),
        ])
        Self.writeUInt32BE(
            &badSuperLength,
            at: 4,
            value: UInt32(badSuperLength.count - 1)
        )

        var zeroEntries = Self.superBlob(entries: [(0, primary)])
        Self.writeUInt32BE(&zeroEntries, at: 8, value: 0)

        var tooManyEntries = Self.superBlob(entries: [(0, primary)])
        Self.writeUInt32BE(
            &tooManyEntries,
            at: 8,
            value:
                SyntheticMachOIdentityParser.maximumSuperBlobEntries + 1
        )

        let duplicateSlot = Self.superBlob(entries: [
            (0, primary),
            (0, alternate),
            (0x10000, cms),
        ])
        let missingPrimary = Self.superBlob(entries: [
            (0x1000, alternate),
            (0x10000, cms),
        ])
        let reorderedDirectories = Self.superBlob(entries: [
            (0x1000, alternate),
            (0, primary),
            (0x10000, cms),
        ])
        let descendingAlternates = Self.superBlob(entries: [
            (0, primary),
            (0x1001, alternate),
            (
                0x1000,
                Self.codeDirectory(
                    hashType: 2,
                    flags: 0,
                    signingIdentifier: Data("descending.invalid".utf8),
                    teamIdentifier: Data()
                )
            ),
            (0x10000, cms),
        ])
        let unsupportedDirectorySlot = Self.superBlob(entries: [
            (0, primary),
            (0x1005, alternate),
            (0x10000, cms),
        ])

        var indexInsideHeader = Self.superBlob(entries: [
            (0, primary),
            (0x10000, cms),
        ])
        Self.writeUInt32BE(&indexInsideHeader, at: 16, value: 12)

        var overlappingBlobs = Self.superBlob(entries: [
            (0, primary),
            (0x1000, alternate),
            (0x10000, cms),
        ])
        let firstBlobOffset = Self.readUInt32BE(
            overlappingBlobs,
            at: 16
        )
        Self.writeUInt32BE(
            &overlappingBlobs,
            at: 24,
            value: firstBlobOffset + 1
        )

        var wrongCodeDirectoryMagic = primary
        Self.writeUInt32BE(
            &wrongCodeDirectoryMagic,
            at: 0,
            value: Self.csMagicRequirements
        )

        var wrongCodeDirectoryLength = primary
        Self.writeUInt32BE(
            &wrongCodeDirectoryLength,
            at: 4,
            value: UInt32(primary.count - 1)
        )

        let unsupportedVersion = Self.codeDirectory(
            version: 0x20000,
            hashType: 2,
            flags: 0,
            signingIdentifier: Data("version.invalid".utf8),
            teamIdentifier: Data()
        )
        let unsupportedHash = Self.codeDirectory(
            hashType: 5,
            flags: 0,
            signingIdentifier: Data("hash.invalid".utf8),
            teamIdentifier: Data()
        )
        let wrongHashSize = Self.codeDirectory(
            hashType: 2,
            hashSize: 20,
            flags: 0,
            signingIdentifier: Data("hash.size.invalid".utf8),
            teamIdentifier: Data()
        )
        var unsupportedScatter = Self.codeDirectory(
            version: 0x20100,
            hashType: 2,
            flags: 0,
            signingIdentifier: Data("scatter.invalid".utf8),
            teamIdentifier: Data()
        )
        Self.writeUInt32BE(&unsupportedScatter, at: 44, value: 1)

        var nonzeroSpare3 = Self.codeDirectory(
            version: 0x20300,
            hashType: 2,
            flags: 0,
            signingIdentifier: Data("spare3.invalid".utf8),
            teamIdentifier: Data()
        )
        Self.writeUInt32BE(&nonzeroSpare3, at: 52, value: 1)

        var unsupportedPreEncrypt = Self.codeDirectory(
            version: 0x20500,
            hashType: 2,
            flags: 0,
            signingIdentifier: Data("preencrypt.invalid".utf8),
            teamIdentifier: Data()
        )
        Self.writeUInt32BE(
            &unsupportedPreEncrypt,
            at: 92,
            value: 1
        )

        var unsupportedLinkage = Self.codeDirectory(
            version: 0x20600,
            hashType: 2,
            flags: 0,
            signingIdentifier: Data("linkage.invalid".utf8),
            teamIdentifier: Data()
        )
        Self.writeUInt32BE(&unsupportedLinkage, at: 100, value: 1)

        var trailingCodeDirectoryBytes = primary
        trailingCodeDirectoryBytes.append(0)
        Self.writeUInt32BE(
            &trailingCodeDirectoryBytes,
            at: 4,
            value: UInt32(trailingCodeDirectoryBytes.count)
        )

        var invalidIdentifier = primary
        Self.writeUInt32BE(
            &invalidIdentifier,
            at: 20,
            value: UInt32(invalidIdentifier.count - 1)
        )
        invalidIdentifier[invalidIdentifier.count - 1] = 0x41

        var invalidTeam = primary
        Self.writeUInt32BE(
            &invalidTeam,
            at: 48,
            value: UInt32(invalidTeam.count - 1)
        )
        invalidTeam[invalidTeam.count - 1] = 0x41

        let malformedCMS = Self.genericBlob(
            magic: Self.csMagicRequirements,
            payload: Data([0x30, 0x00])
        )
        let signedWithoutCMS = Self.superBlob(entries: [(0, primary)])
        let adHocWithCMS = Self.superBlob(entries: [
            (
                0,
                Self.codeDirectory(
                    hashType: 2,
                    flags: Self.csAdHoc,
                    signingIdentifier: Data("adhoc.invalid".utf8),
                    teamIdentifier: Data()
                )
            ),
            (0x10000, cms),
        ])
        let mixedAdHoc = Self.superBlob(entries: [
            (
                0,
                Self.codeDirectory(
                    hashType: 2,
                    flags: Self.csAdHoc,
                    signingIdentifier: Data("adhoc.primary".utf8),
                    teamIdentifier: Data()
                )
            ),
            (0x1000, alternate),
        ])

        let cases: [(Data, SyntheticMachOIdentityFailure)] = [
            (badSuperMagic, .superBlobBounds),
            (badSuperLength, .superBlobBounds),
            (zeroEntries, .superBlobEntryCount),
            (tooManyEntries, .superBlobEntryCount),
            (duplicateSlot, .duplicateSuperBlobSlot),
            (missingPrimary, .missingPrimaryCodeDirectory),
            (reorderedDirectories, .codeDirectoryOrder),
            (descendingAlternates, .codeDirectoryOrder),
            (unsupportedDirectorySlot, .unsupportedCodeDirectorySlot),
            (indexInsideHeader, .blobBounds),
            (overlappingBlobs, .overlappingBlobRanges),
            (
                Self.superBlob(entries: [
                    (0, wrongCodeDirectoryMagic),
                    (0x10000, cms),
                ]),
                .codeDirectoryBounds
            ),
            (
                Self.superBlob(entries: [
                    (0, wrongCodeDirectoryLength),
                    (0x10000, cms),
                ]),
                .codeDirectoryBounds
            ),
            (
                Self.superBlob(entries: [
                    (0, unsupportedVersion),
                    (0x10000, cms),
                ]),
                .unsupportedCodeDirectoryVersion
            ),
            (
                Self.superBlob(entries: [
                    (0, unsupportedHash),
                    (0x10000, cms),
                ]),
                .unsupportedCodeDirectoryHashType
            ),
            (
                Self.superBlob(entries: [
                    (0, wrongHashSize),
                    (0x10000, cms),
                ]),
                .codeDirectoryHashSize
            ),
            (
                Self.superBlob(entries: [
                    (0, unsupportedScatter),
                    (0x10000, cms),
                ]),
                .codeDirectoryBounds
            ),
            (
                Self.superBlob(entries: [
                    (0, nonzeroSpare3),
                    (0x10000, cms),
                ]),
                .codeDirectoryBounds
            ),
            (
                Self.superBlob(entries: [
                    (0, unsupportedPreEncrypt),
                    (0x10000, cms),
                ]),
                .codeDirectoryBounds
            ),
            (
                Self.superBlob(entries: [
                    (0, unsupportedLinkage),
                    (0x10000, cms),
                ]),
                .codeDirectoryBounds
            ),
            (
                Self.superBlob(entries: [
                    (0, trailingCodeDirectoryBytes),
                    (0x10000, cms),
                ]),
                .codeDirectoryBounds
            ),
            (
                Self.superBlob(entries: [
                    (0, invalidIdentifier),
                    (0x10000, cms),
                ]),
                .signingIdentifier
            ),
            (
                Self.superBlob(entries: [
                    (0, invalidTeam),
                    (0x10000, cms),
                ]),
                .teamIdentifier
            ),
            (
                Self.superBlob(entries: [
                    (0, primary),
                    (0x10000, malformedCMS),
                ]),
                .cmsBlob
            ),
            (signedWithoutCMS, .cmsAdHocInconsistency),
            (adHocWithCMS, .cmsAdHocInconsistency),
            (mixedAdHoc, .cmsAdHocInconsistency),
        ]

        for (index, testCase) in cases.enumerated() {
            assertRejected(
                Self.machO(signatureRegion: testCase.0),
                as: testCase.1,
                context: "signature case \(index)"
            )
        }
    }

    private func assertRejected(
        _ bytes: Data,
        as expected: SyntheticMachOIdentityFailure,
        context: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let effects = FakeOperationalEffects()
        XCTAssertThrowsError(
            try SyntheticMachOIdentityParser.parse(bytes),
            context,
            file: file,
            line: line
        ) { error in
            XCTAssertEqual(
                error as? SyntheticMachOIdentityFailure,
                expected,
                context,
                file: file,
                line: line
            )
        }
        XCTAssertEqual(effects, .zero, file: file, line: line)
    }
}

private extension SyntheticMachOIdentityParserTests {
    struct FakeOperationalEffects: Equatable {
        var spawnCount: UInt64 = 0
        var networkCount: UInt64 = 0
        var fileSystemCount: UInt64 = 0
        var packCount: UInt64 = 0
        var objectDatabaseCount: UInt64 = 0
        var sourceMutationCount: UInt64 = 0
        var buildCount: UInt64 = 0
        var modelCount: UInt64 = 0
        var reservationCount: UInt64 = 0
        var publicationCount: UInt64 = 0

        static let zero = Self()
    }

    static let canonicalUUID: [UInt8] = [
        0x01, 0x23, 0x45, 0x67,
        0x89, 0xab, 0xcd, 0xef,
        0x10, 0x32, 0x54, 0x76,
        0x98, 0xba, 0xdc, 0xfe,
    ]
    static let csMagicRequirements: UInt32 = 0xfade0c01
    static let csMagicCodeDirectory: UInt32 = 0xfade0c02
    static let csMagicEmbeddedSignature: UInt32 = 0xfade0cc0
    static let csMagicBlobWrapper: UInt32 = 0xfade0b01
    static let csAdHoc: UInt32 = 0x00000002

    static func signedFixture() -> Data {
        machO(signatureRegion: signedSignature())
    }

    static func signedSignature() -> Data {
        superBlob(entries: [
            (
                0,
                codeDirectory(
                    hashType: 2,
                    flags: 0,
                    signingIdentifier: Data("com.example.git".utf8),
                    teamIdentifier: Data("TEAM123456".utf8)
                )
            ),
            (
                0x10000,
                genericBlob(
                    magic: csMagicBlobWrapper,
                    payload: Data([0x30, 0x00])
                )
            ),
        ])
    }

    static func machO(
        signatureRegion: Data,
        uuidCommands: [[UInt8]] = [canonicalUUID],
        codeSignatureCommandCount: Int = 1,
        includeDyldEnvironment: Bool = false
    ) -> Data {
        var loadCommands = Data()
        for uuid in uuidCommands {
            precondition(uuid.count == 16)
            appendUInt32LE(&loadCommands, 0x1b)
            appendUInt32LE(&loadCommands, 24)
            loadCommands.append(contentsOf: uuid)
        }

        var signatureCommandOffsets: [Int] = []
        for _ in 0..<codeSignatureCommandCount {
            signatureCommandOffsets.append(loadCommands.count)
            appendUInt32LE(&loadCommands, 0x1d)
            appendUInt32LE(&loadCommands, 16)
            appendUInt32LE(&loadCommands, 0)
            appendUInt32LE(
                &loadCommands,
                UInt32(signatureRegion.count)
            )
        }
        if includeDyldEnvironment {
            appendUInt32LE(&loadCommands, 0x27)
            appendUInt32LE(&loadCommands, 8)
        }

        let signatureOffset = 32 + loadCommands.count
        for commandOffset in signatureCommandOffsets {
            writeUInt32LE(
                &loadCommands,
                at: commandOffset + 8,
                value: UInt32(signatureOffset)
            )
        }

        var result = Data()
        appendUInt32LE(&result, 0xfeedfacf)
        appendUInt32LE(&result, 0x0100000c)
        appendUInt32LE(&result, 2)
        appendUInt32LE(&result, 2)
        appendUInt32LE(
            &result,
            UInt32(
                uuidCommands.count +
                    codeSignatureCommandCount +
                    (includeDyldEnvironment ? 1 : 0)
            )
        )
        appendUInt32LE(&result, UInt32(loadCommands.count))
        appendUInt32LE(&result, 0x00200085)
        appendUInt32LE(&result, 0)
        result.append(loadCommands)
        result.append(signatureRegion)
        return result
    }

    static func superBlob(entries: [(UInt32, Data)]) -> Data {
        let indexBytes = entries.count * 8
        var nextOffset = 12 + indexBytes
        var result = Data()
        appendUInt32BE(&result, csMagicEmbeddedSignature)
        appendUInt32BE(&result, 0)
        appendUInt32BE(&result, UInt32(entries.count))
        for (slot, blob) in entries {
            appendUInt32BE(&result, slot)
            appendUInt32BE(&result, UInt32(nextOffset))
            nextOffset += blob.count
        }
        for (_, blob) in entries {
            result.append(blob)
        }
        writeUInt32BE(
            &result,
            at: 4,
            value: UInt32(result.count)
        )
        return result
    }

    static func genericBlob(
        magic: UInt32,
        payload: Data
    ) -> Data {
        var result = Data()
        appendUInt32BE(&result, magic)
        appendUInt32BE(&result, UInt32(8 + payload.count))
        result.append(payload)
        return result
    }

    static func codeDirectory(
        version: UInt32 = 0x20200,
        hashType: UInt8,
        hashSize: UInt8? = nil,
        flags: UInt32,
        signingIdentifier: Data,
        teamIdentifier: Data
    ) -> Data {
        let fixedBytes: Int
        switch version {
        case ..<0x20100:
            fixedBytes = 44
        case ..<0x20200:
            fixedBytes = 48
        case ..<0x20300:
            fixedBytes = 52
        case ..<0x20400:
            fixedBytes = 64
        case ..<0x20500:
            fixedBytes = 88
        case ..<0x20600:
            fixedBytes = 96
        default:
            fixedBytes = 108
        }

        var result = Data(repeating: 0, count: fixedBytes)
        let identifierOffset = result.count
        result.append(signingIdentifier)
        result.append(0)
        let teamOffset: Int
        if version >= 0x20200, !teamIdentifier.isEmpty {
            teamOffset = result.count
            result.append(teamIdentifier)
            result.append(0)
        } else {
            teamOffset = 0
        }
        let hashOffset = result.count

        writeUInt32BE(&result, at: 0, value: csMagicCodeDirectory)
        writeUInt32BE(
            &result,
            at: 4,
            value: UInt32(result.count)
        )
        writeUInt32BE(&result, at: 8, value: version)
        writeUInt32BE(&result, at: 12, value: flags)
        writeUInt32BE(
            &result,
            at: 16,
            value: UInt32(hashOffset)
        )
        writeUInt32BE(
            &result,
            at: 20,
            value: UInt32(identifierOffset)
        )
        writeUInt32BE(&result, at: 24, value: 0)
        writeUInt32BE(&result, at: 28, value: 0)
        writeUInt32BE(&result, at: 32, value: 0)
        result[36] = hashSize ?? expectedHashSize(hashType)
        result[37] = hashType
        result[38] = 0
        result[39] = 0
        writeUInt32BE(&result, at: 40, value: 0)
        if version >= 0x20100 {
            writeUInt32BE(&result, at: 44, value: 0)
        }
        if version >= 0x20200 {
            writeUInt32BE(
                &result,
                at: 48,
                value: UInt32(teamOffset)
            )
        }
        return result
    }

    static func expectedHashSize(_ hashType: UInt8) -> UInt8 {
        switch hashType {
        case 1:
            20
        case 2:
            32
        case 3:
            20
        case 4:
            48
        default:
            0
        }
    }

    static func replacingUInt32LE(
        in data: Data,
        at offset: Int,
        with value: UInt32
    ) -> Data {
        var result = data
        writeUInt32LE(&result, at: offset, value: value)
        return result
    }

    static func appendUInt32LE(_ data: inout Data, _ value: UInt32) {
        data.append(UInt8(truncatingIfNeeded: value))
        data.append(UInt8(truncatingIfNeeded: value >> 8))
        data.append(UInt8(truncatingIfNeeded: value >> 16))
        data.append(UInt8(truncatingIfNeeded: value >> 24))
    }

    static func appendUInt32BE(_ data: inout Data, _ value: UInt32) {
        data.append(UInt8(truncatingIfNeeded: value >> 24))
        data.append(UInt8(truncatingIfNeeded: value >> 16))
        data.append(UInt8(truncatingIfNeeded: value >> 8))
        data.append(UInt8(truncatingIfNeeded: value))
    }

    static func writeUInt32LE(
        _ data: inout Data,
        at offset: Int,
        value: UInt32
    ) {
        data[offset] = UInt8(truncatingIfNeeded: value)
        data[offset + 1] = UInt8(truncatingIfNeeded: value >> 8)
        data[offset + 2] = UInt8(truncatingIfNeeded: value >> 16)
        data[offset + 3] = UInt8(truncatingIfNeeded: value >> 24)
    }

    static func writeUInt32BE(
        _ data: inout Data,
        at offset: Int,
        value: UInt32
    ) {
        data[offset] = UInt8(truncatingIfNeeded: value >> 24)
        data[offset + 1] = UInt8(truncatingIfNeeded: value >> 16)
        data[offset + 2] = UInt8(truncatingIfNeeded: value >> 8)
        data[offset + 3] = UInt8(truncatingIfNeeded: value)
    }

    static func readUInt32BE(_ data: Data, at offset: Int) -> UInt32 {
        UInt32(data[offset]) << 24 |
            UInt32(data[offset + 1]) << 16 |
            UInt32(data[offset + 2]) << 8 |
            UInt32(data[offset + 3])
    }

    static func sha256Hex(_ data: Data) -> String {
        // The production parser owns the exact digest implementation.
        // This test helper independently asks CryptoKit through the same
        // SDK primitive used elsewhere in ProofControl.
        CryptoKit.SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}
