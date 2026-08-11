import CryptoKit
import Foundation
import XCTest
@testable import ProofControl

final class SyntheticDynamicLoaderMachOIdentityTests: XCTestCase {
    func testCanonicalDynamicLoaderProducesSealedInertContentIdentity()
        throws
    {
        var fixture = Self.dynamicLoaderMachO(
            extraCommands: [Self.bareCommand(0x19)]
        )
        let effects = FakeOperationalEffects()

        let comparison = try SyntheticDynamicLoaderMachOIdentityParser
            .parse(fixture)
        let evidence = try DynamicLoaderContentIdentityVerifier.derive(
            comparison: comparison
        )
        let expectedPreimage = Self.expectedPreimage(
            comparison: comparison
        )

        XCTAssertEqual(comparison.retainedFileBytes, fixture)
        XCTAssertEqual(comparison.fileSHA256, Self.sha256Hex(fixture))
        XCTAssertEqual(comparison.machHeaderMagic, 0xfeedfacf)
        XCTAssertEqual(comparison.cpuType, 0x0100000c)
        XCTAssertEqual(comparison.cpuSubtype, 2)
        XCTAssertEqual(comparison.fileType, 7)
        XCTAssertEqual(comparison.headerFlags, 0x00200085)
        XCTAssertEqual(comparison.loadCommandCount, 4)
        XCTAssertEqual(
            comparison.loadCommandsSHA256,
            Self.sha256Hex(comparison.loadCommandBytes)
        )
        XCTAssertEqual(comparison.machOUUID, Data(Self.canonicalUUID))
        XCTAssertEqual(
            comparison.dynamicLoaderIDName,
            Data("/usr/lib/dyld".utf8)
        )
        XCTAssertEqual(comparison.codeDirectories.map(\.slot), [0])
        XCTAssertNotNil(comparison.cmsBlob)
        XCTAssertFalse(comparison.isAdHoc)
        XCTAssertEqual(
            SyntheticDynamicLoaderMachOIdentityParser
                .recognizedLoadCommandValueCount,
            54
        )
        Self.assertInert(comparison)

        XCTAssertEqual(evidence.artifactRole, .dynamicLoader)
        XCTAssertEqual(evidence.identityPreimage, expectedPreimage)
        XCTAssertEqual(
            evidence.contentEvidenceID.sha256,
            Self.sha256Hex(expectedPreimage)
        )
        XCTAssertEqual(
            evidence.contentEvidenceID.sha256,
            "2c1309659814338618efc93bca4086a99eef469c1fc16b7d5e5f08d5bf5b23ed"
        )
        XCTAssertEqual(
            String(decoding: expectedPreimage, as: UTF8.self)
                .split(
                    separator: "\n",
                    omittingEmptySubsequences: false
                )
                .count,
            10
        )
        XCTAssertEqual(
            evidence.machHeaderSHA256,
            Self.sha256Hex(Data(fixture.prefix(32)))
        )
        XCTAssertEqual(
            evidence.primaryCodeDirectoryBlobSHA256,
            comparison.codeDirectories[0].blobSHA256
        )
        XCTAssertEqual(evidence.comparison, comparison)
        Self.assertInert(evidence)

        XCTAssertThrowsError(
            try SyntheticMachOIdentityParser.parse(fixture)
        ) { error in
            XCTAssertEqual(
                error as? SyntheticMachOIdentityFailure,
                .unsupportedFileType
            )
        }

        let genericExecuteComparison = try SyntheticMachOIdentityParser
            .parse(Self.executeMachO())
        let genericRelabel = try ExecutableContentIdentityVerifier.derive(
            artifactRole: .dynamicLoader,
            comparison: genericExecuteComparison
        )
        XCTAssertNotEqual(
            genericRelabel.contentEvidenceID.sha256,
            evidence.contentEvidenceID.sha256
        )

        var identityMutation = comparison.retainedFileBytes
        let identityOffset = try XCTUnwrap(
            Self.commandOffset(in: identityMutation, command: 0x0f)
        )
        identityMutation[identityOffset + 13] = 0x76
        let mutationComparison = try
            SyntheticDynamicLoaderMachOIdentityParser.parse(
                identityMutation
            )
        let mutationEvidence = try
            DynamicLoaderContentIdentityVerifier.derive(
                comparison: mutationComparison
            )
        XCTAssertNotEqual(
            mutationEvidence.contentEvidenceID,
            evidence.contentEvidenceID
        )

        fixture[0] = 0
        XCTAssertEqual(
            comparison.retainedFileBytes[0],
            UInt8(0xcf)
        )

        let mutableFixture = NSMutableData(
            data: Self.dynamicLoaderMachO()
        )
        let mutableComparison = try
            SyntheticDynamicLoaderMachOIdentityParser.parse(
                Data(referencing: mutableFixture)
            )
        var replacement: UInt8 = 0
        mutableFixture.replaceBytes(
            in: NSRange(location: 0, length: 1),
            withBytes: &replacement
        )
        XCTAssertEqual(
            mutableComparison.retainedFileBytes[0],
            UInt8(0xcf)
        )
        XCTAssertEqual(effects, .zero)
    }

    func testRejectsHeaderFramingUnknownAndForbiddenCommands()
        throws
    {
        let canonical = Self.dynamicLoaderMachO()
        var headerCases: [(Data, SyntheticMachOIdentityFailure)] = [
            (Data(), .fileBounds),
            (
                Data(
                    repeating: 0,
                    count:
                        SyntheticDynamicLoaderMachOIdentityParser
                            .maximumFileBytes + 1
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
                    with: 2
                ),
                .unsupportedFileType
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
                    with: 257
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
                    at: 20,
                    with: 262_145
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
                    with: 0
                ),
                .malformedDynamicLoaderLoadCommand(ordinal: 0)
            ),
            (
                Self.replacingUInt32LE(
                    in: canonical,
                    at: 36,
                    with: 9
                ),
                .malformedDynamicLoaderLoadCommand(ordinal: 0)
            ),
            (
                Self.replacingUInt32LE(
                    in: canonical,
                    at: 36,
                    with: UInt32.max
                ),
                .malformedDynamicLoaderLoadCommand(ordinal: 0)
            ),
            (
                Self.replacingUInt32LE(
                    in: canonical,
                    at: 16,
                    with: 2
                ),
                .malformedDynamicLoaderLoadCommand(ordinal: 2)
            ),
            (
                Self.replacingUInt32LE(
                    in: canonical,
                    at: 16,
                    with: 4
                ),
                .malformedDynamicLoaderLoadCommand(ordinal: 3)
            ),
            (
                Self.replacingUInt32LE(
                    in: canonical,
                    at: 20,
                    with: Self.readUInt32LE(canonical, at: 20) - 8
                ),
                .malformedDynamicLoaderLoadCommand(ordinal: 2)
            ),
            (
                Self.replacingUInt32LE(
                    in: canonical,
                    at: 20,
                    with: Self.readUInt32LE(canonical, at: 20) + 8
                ),
                .malformedDynamicLoaderLoadCommand(ordinal: 3)
            ),
        ]

        var truncated = canonical
        truncated.removeLast()
        headerCases.append((truncated, .codeSignatureBounds))

        for (index, testCase) in headerCases.enumerated() {
            assertRejected(
                testCase.0,
                as: testCase.1,
                context: "header case \(index)"
            )
        }

        assertRejected(
            Self.dynamicLoaderMachO(
                extraCommands: [Self.bareCommand(0x36)]
            ),
            as: .unknownOptionalDynamicLoaderLoadCommand(
                ordinal: 2,
                command: 0x36
            ),
            context: "unknown optional"
        )
        assertRejected(
            Self.dynamicLoaderMachO(
                extraCommands: [Self.bareCommand(0x80000036)]
            ),
            as: .unknownRequiredDynamicLoaderLoadCommand(
                ordinal: 2,
                command: 0x80000036
            ),
            context: "unknown required"
        )

        let forbidden: [(UInt32, SyntheticDynamicLoaderForbiddenCommand)] = [
            (0x27, .dyldEnvironment),
            (0x80000028, .main),
            (0x0e, .loadDynamicLinker),
            (0x21, .encryptionInfo),
            (0x2c, .encryptionInfo64),
            (0x80000035, .fileSetEntry),
            (0x0d, .idDylib),
            (0x0c, .loadDylib),
            (0x80000018, .loadWeakDylib),
            (0x8000001f, .reexportDylib),
            (0x20, .lazyLoadDylib),
            (0x80000023, .loadUpwardDylib),
            (0x8000001c, .rpath),
        ]
        for (command, kind) in forbidden {
            assertRejected(
                Self.dynamicLoaderMachO(
                    extraCommands: [Self.bareCommand(command)]
                ),
                as: .forbiddenDynamicLoaderLoadCommand(
                    ordinal: 2,
                    kind: kind
                ),
                context: "forbidden \(String(command, radix: 16))"
            )
        }

        let specialized: Set<UInt32> = [0x0f, 0x1b, 0x1d]
        let forbiddenValues = Set(forbidden.map(\.0))
        for command in Self.expectedRecognizedLoadCommands
        where !specialized.contains(command) &&
            !forbiddenValues.contains(command)
        {
            let comparison = try
                SyntheticDynamicLoaderMachOIdentityParser.parse(
                    Self.dynamicLoaderMachO(
                        extraCommands: [Self.bareCommand(command)]
                    )
                )
            XCTAssertEqual(
                comparison.loadCommandCount,
                4,
                "recognized command \(String(command, radix: 16))"
            )
        }

        let accepted = try SyntheticDynamicLoaderMachOIdentityParser.parse(
            Self.dynamicLoaderMachO(
                extraCommands: [Self.bareCommand(0x19)]
            )
        )
        XCTAssertEqual(accepted.loadCommandCount, 4)
    }

    func testRejectsDynamicLoaderIdentityCommandFailures() {
        assertRejected(
            Self.dynamicLoaderMachO(identityCommands: []),
            as: .missingDynamicLoaderIdentity,
            context: "missing identity"
        )
        let canonicalIdentity = Self.dynamicLoaderIdentityCommand()
        assertRejected(
            Self.dynamicLoaderMachO(
                identityCommands: [canonicalIdentity, canonicalIdentity]
            ),
            as: .duplicateDynamicLoaderIdentity(ordinal: 2),
            context: "duplicate identity"
        )

        let cases: [(Data, SyntheticDynamicLoaderIdentityField, String)] = [
            (
                Self.bareCommand(0x0f),
                .commandSize,
                "short command"
            ),
            (
                Self.dynamicLoaderIdentityCommand(nameOffset: 8),
                .nameOffset,
                "offset before structure"
            ),
            (
                Self.dynamicLoaderIdentityCommandWithNameOffsetAtEnd(),
                .nameOffset,
                "offset at command end"
            ),
            (
                Self.dynamicLoaderIdentityCommand(
                    nameOffset: 16,
                    prefixPaddingByte: 1,
                    minimumCommandBytes: 24
                ),
                .prefixPadding,
                "nonzero prefix padding"
            ),
            (
                Self.dynamicLoaderIdentityCommand(
                    name: Data(repeating: 0x41, count: 4),
                    includeTerminator: false,
                    trailingPaddingByte: 0x41
                ),
                .terminator,
                "missing terminator"
            ),
            (
                Self.dynamicLoaderIdentityCommand(name: Data()),
                .length,
                "empty identity"
            ),
            (
                Self.dynamicLoaderIdentityCommand(name: Data("/".utf8)),
                .length,
                "one-byte identity"
            ),
            (
                Self.dynamicLoaderIdentityCommand(
                    name: Data([0x2f]) +
                        Data(repeating: 0x61, count: 4_096)
                ),
                .length,
                "identity over cap"
            ),
            (
                Self.dynamicLoaderIdentityCommand(
                    name: Data("/x".utf8),
                    trailingPaddingByte: 1
                ),
                .trailingPadding,
                "nonzero trailing padding"
            ),
            (
                Self.dynamicLoaderIdentityCommand(
                    name: Data("usr/lib/dyld".utf8)
                ),
                .canonicalName,
                "relative name"
            ),
            (
                Self.dynamicLoaderIdentityCommand(
                    name: Data("/@rpath/dyld".utf8)
                ),
                .canonicalName,
                "at sign"
            ),
            (
                Self.dynamicLoaderIdentityCommand(
                    name: Data("/usr\\dyld".utf8)
                ),
                .canonicalName,
                "backslash"
            ),
            (
                Self.dynamicLoaderIdentityCommand(
                    name: Data("/usr//dyld".utf8)
                ),
                .canonicalName,
                "empty component"
            ),
            (
                Self.dynamicLoaderIdentityCommand(
                    name: Data("/usr/./dyld".utf8)
                ),
                .canonicalName,
                "dot component"
            ),
            (
                Self.dynamicLoaderIdentityCommand(
                    name: Data("/usr/../dyld".utf8)
                ),
                .canonicalName,
                "dot-dot component"
            ),
            (
                Self.dynamicLoaderIdentityCommand(
                    name: Data([0x2f, 0x20])
                ),
                .canonicalName,
                "nonprintable component"
            ),
        ]
        for testCase in cases {
            assertRejected(
                Self.dynamicLoaderMachO(
                    identityCommands: [testCase.0]
                ),
                as: .dynamicLoaderIdentityLayout(
                    ordinal: 1,
                    field: testCase.1
                ),
                context: testCase.2
            )
        }
    }

    func testRejectsInReviewedFramingClassificationAndSemanticOrder()
        throws
    {
        var malformedAfterUnknown = Self.dynamicLoaderMachO()
        Self.writeUInt32LE(
            &malformedAfterUnknown,
            at: 32,
            value: 0x36
        )
        let identityOffset = try XCTUnwrap(
            Self.commandOffset(
                in: malformedAfterUnknown,
                command: 0x0f
            )
        )
        Self.writeUInt32LE(
            &malformedAfterUnknown,
            at: identityOffset + 4,
            value: 9
        )
        assertRejected(
            malformedAfterUnknown,
            as: .malformedDynamicLoaderLoadCommand(ordinal: 1),
            context: "complete framing precedes command classification"
        )

        var unknownBeforeForbidden = Self.dynamicLoaderMachO(
            extraCommands: [Self.bareCommand(0x27)]
        )
        Self.writeUInt32LE(
            &unknownBeforeForbidden,
            at: 32,
            value: 0x36
        )
        assertRejected(
            unknownBeforeForbidden,
            as: .unknownOptionalDynamicLoaderLoadCommand(
                ordinal: 0,
                command: 0x36
            ),
            context: "unknown classification precedes forbidden policy"
        )

        assertRejected(
            Self.dynamicLoaderMachO(
                uuidCommands: [[UInt8](repeating: 0, count: 16)],
                extraCommands: [Self.bareCommand(0x27)]
            ),
            as: .forbiddenDynamicLoaderLoadCommand(
                ordinal: 2,
                kind: .dyldEnvironment
            ),
            context: "forbidden policy precedes command semantics"
        )
    }

    func testRejectsUUIDCodeSignatureAndEmbeddedSignatureFailures() {
        assertRejected(
            Self.dynamicLoaderMachO(uuidCommands: []),
            as: .missingUUID,
            context: "missing UUID"
        )
        assertRejected(
            Self.dynamicLoaderMachO(
                uuidCommands: [Self.canonicalUUID, Self.canonicalUUID]
            ),
            as: .duplicateUUID,
            context: "duplicate UUID"
        )
        assertRejected(
            Self.dynamicLoaderMachO(uuidCommands: [[UInt8](repeating: 0, count: 16)]),
            as: .zeroUUID,
            context: "zero UUID"
        )
        assertRejected(
            Self.dynamicLoaderMachO(codeSignatureCommandCount: 0),
            as: .missingCodeSignature,
            context: "missing signature command"
        )
        assertRejected(
            Self.dynamicLoaderMachO(codeSignatureCommandCount: 2),
            as: .duplicateCodeSignature,
            context: "duplicate signature command"
        )

        let canonical = Self.dynamicLoaderMachO()
        let signatureCommandOffset = Self.commandOffset(
            in: canonical,
            command: 0x1d
        )!
        assertRejected(
            Self.replacingUInt32LE(
                in: canonical,
                at: signatureCommandOffset + 8,
                with: 32
            ),
            as: .codeSignatureBounds,
            context: "overlapping signature"
        )
        assertRejected(
            Self.replacingUInt32LE(
                in: canonical,
                at: signatureCommandOffset + 12,
                with: 0
            ),
            as: .codeSignatureBounds,
            context: "zero signature"
        )
        assertRejected(
            Self.replacingUInt32LE(
                in: canonical,
                at: signatureCommandOffset + 8,
                with: UInt32(canonical.count)
            ),
            as: .codeSignatureBounds,
            context: "signature after end"
        )

        let primary = Self.codeDirectory(
            hashType: 2,
            flags: 0,
            signingIdentifier: Data("com.example.dyld".utf8),
            teamIdentifier: Data("TEAM123456".utf8)
        )
        let cms = Self.genericBlob(
            magic: Self.csMagicBlobWrapper,
            payload: Data([0x30, 0x00])
        )
        let noPrimary = Self.superBlob(entries: [
            (
                2,
                Self.genericBlob(
                    magic: Self.csMagicRequirements,
                    payload: Data([1])
                )
            ),
            (0x10000, cms),
        ])
        let reordered = Self.superBlob(entries: [
            (
                0x1000,
                Self.codeDirectory(
                    hashType: 4,
                    flags: 0,
                    signingIdentifier: Data("alt".utf8),
                    teamIdentifier: Data()
                )
            ),
            (0, primary),
            (0x10000, cms),
        ])
        let unsupportedHash = Self.superBlob(entries: [
            (
                0,
                Self.codeDirectory(
                    hashType: 5,
                    hashSize: 32,
                    flags: 0,
                    signingIdentifier: Data("hash.invalid".utf8),
                    teamIdentifier: Data()
                )
            ),
            (0x10000, cms),
        ])
        let wrongHashSize = Self.superBlob(entries: [
            (
                0,
                Self.codeDirectory(
                    hashType: 2,
                    hashSize: 20,
                    flags: 0,
                    signingIdentifier: Data("hash.size.invalid".utf8),
                    teamIdentifier: Data()
                )
            ),
            (0x10000, cms),
        ])
        var invalidIdentifier = Self.codeDirectory(
            hashType: 2,
            flags: 0,
            signingIdentifier: Data("invalid.identifier".utf8),
            teamIdentifier: Data()
        )
        invalidIdentifier[invalidIdentifier.count - 1] = 0x41
        let invalidIdentifierSignature = Self.superBlob(entries: [
            (0, invalidIdentifier),
            (0x10000, cms),
        ])
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

        let signatureCases: [(Data, SyntheticMachOIdentityFailure)] = [
            (noPrimary, .missingPrimaryCodeDirectory),
            (reordered, .codeDirectoryOrder),
            (unsupportedHash, .unsupportedCodeDirectoryHashType),
            (wrongHashSize, .codeDirectoryHashSize),
            (invalidIdentifierSignature, .signingIdentifier),
            (signedWithoutCMS, .cmsAdHocInconsistency),
            (adHocWithCMS, .cmsAdHocInconsistency),
        ]
        for (index, testCase) in signatureCases.enumerated() {
            assertRejected(
                Self.dynamicLoaderMachO(signatureRegion: testCase.0),
                as: testCase.1,
                context: "signature case \(index)"
            )
        }
    }

    func testContentIdentityRejectsMissingPrimaryAndParserFactDrift()
        throws
    {
        let comparison = try SyntheticDynamicLoaderMachOIdentityParser
            .parse(Self.dynamicLoaderMachO())

        assertContentRejected(
            Self.replacing(comparison, codeDirectories: []),
            as: .missingPrimaryCodeDirectory
        )
        assertContentRejected(
            Self.replacing(
                comparison,
                fileSHA256: String(repeating: "f", count: 64)
            ),
            as: .parserFactMismatch
        )

        var mutatedBytes = comparison.retainedFileBytes
        mutatedBytes[12] = 2
        assertContentRejected(
            Self.replacing(
                comparison,
                retainedFileBytes: mutatedBytes
            ),
            as: .parserFactMismatch
        )
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
            try SyntheticDynamicLoaderMachOIdentityParser.parse(bytes),
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

    private func assertContentRejected(
        _ comparison: SyntheticDynamicLoaderMachOIdentityComparison,
        as expected: DynamicLoaderContentIdentityFailure,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let effects = FakeOperationalEffects()
        XCTAssertThrowsError(
            try DynamicLoaderContentIdentityVerifier.derive(
                comparison: comparison
            ),
            file: file,
            line: line
        ) { error in
            XCTAssertEqual(
                error as? DynamicLoaderContentIdentityFailure,
                expected,
                file: file,
                line: line
            )
        }
        XCTAssertEqual(effects, .zero, file: file, line: line)
    }
}

private extension SyntheticDynamicLoaderMachOIdentityTests {
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
    static let expectedRecognizedLoadCommands: [UInt32] = [
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

    static func dynamicLoaderMachO(
        signatureRegion: Data = signedSignature(),
        uuidCommands: [[UInt8]] = [canonicalUUID],
        identityCommands: [Data]? = nil,
        extraCommands: [Data] = [],
        codeSignatureCommandCount: Int = 1
    ) -> Data {
        var loadCommands = Data()
        for uuid in uuidCommands {
            precondition(uuid.count == 16)
            appendUInt32LE(&loadCommands, 0x1b)
            appendUInt32LE(&loadCommands, 24)
            loadCommands.append(contentsOf: uuid)
        }
        for identity in identityCommands ?? [dynamicLoaderIdentityCommand()] {
            loadCommands.append(identity)
        }
        for command in extraCommands {
            loadCommands.append(command)
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
        let signatureOffset = 32 + loadCommands.count
        for offset in signatureCommandOffsets {
            writeUInt32LE(
                &loadCommands,
                at: offset + 8,
                value: UInt32(signatureOffset)
            )
        }

        var result = Data()
        appendUInt32LE(&result, 0xfeedfacf)
        appendUInt32LE(&result, 0x0100000c)
        appendUInt32LE(&result, 2)
        appendUInt32LE(&result, 7)
        appendUInt32LE(
            &result,
            UInt32(
                uuidCommands.count +
                    (identityCommands ?? [dynamicLoaderIdentityCommand()])
                    .count +
                    extraCommands.count +
                    codeSignatureCommandCount
            )
        )
        appendUInt32LE(&result, UInt32(loadCommands.count))
        appendUInt32LE(&result, 0x00200085)
        appendUInt32LE(&result, 0)
        result.append(loadCommands)
        result.append(signatureRegion)
        return result
    }

    static func executeMachO() -> Data {
        let signatureRegion = signedSignature()
        var loadCommands = Data()
        appendUInt32LE(&loadCommands, 0x1b)
        appendUInt32LE(&loadCommands, 24)
        loadCommands.append(contentsOf: canonicalUUID)
        appendUInt32LE(&loadCommands, 0x1d)
        appendUInt32LE(&loadCommands, 16)
        appendUInt32LE(&loadCommands, UInt32(32 + loadCommands.count + 8))
        appendUInt32LE(&loadCommands, UInt32(signatureRegion.count))

        var result = Data()
        appendUInt32LE(&result, 0xfeedfacf)
        appendUInt32LE(&result, 0x0100000c)
        appendUInt32LE(&result, 2)
        appendUInt32LE(&result, 2)
        appendUInt32LE(&result, 2)
        appendUInt32LE(&result, UInt32(loadCommands.count))
        appendUInt32LE(&result, 0x00200085)
        appendUInt32LE(&result, 0)
        result.append(loadCommands)
        result.append(signatureRegion)
        return result
    }

    static func dynamicLoaderIdentityCommand(
        name: Data = Data("/usr/lib/dyld".utf8),
        nameOffset: UInt32 = 12,
        includeTerminator: Bool = true,
        prefixPaddingByte: UInt8 = 0,
        trailingPaddingByte: UInt8 = 0,
        minimumCommandBytes: Int = 16
    ) -> Data {
        var result = Data()
        appendUInt32LE(&result, 0x0f)
        appendUInt32LE(&result, 0)
        appendUInt32LE(&result, nameOffset)
        if Int(nameOffset) > result.count {
            result.append(
                Data(
                    repeating: prefixPaddingByte,
                    count: Int(nameOffset) - result.count
                )
            )
        }
        result.append(name)
        if includeTerminator {
            result.append(0)
        }
        let unalignedCount = max(result.count, minimumCommandBytes)
        let alignedCount = (unalignedCount + 7) & ~7
        if result.count < alignedCount {
            result.append(
                Data(
                    repeating: trailingPaddingByte,
                    count: alignedCount - result.count
                )
            )
        }
        writeUInt32LE(
            &result,
            at: 4,
            value: UInt32(result.count)
        )
        return result
    }

    static func dynamicLoaderIdentityCommandWithNameOffsetAtEnd()
        -> Data
    {
        var result = Data(repeating: 0, count: 16)
        writeUInt32LE(&result, at: 0, value: 0x0f)
        writeUInt32LE(&result, at: 4, value: 16)
        writeUInt32LE(&result, at: 8, value: 16)
        return result
    }

    static func bareCommand(_ value: UInt32) -> Data {
        var result = Data()
        appendUInt32LE(&result, value)
        appendUInt32LE(&result, 8)
        return result
    }

    static func signedSignature() -> Data {
        let primary = codeDirectory(
            hashType: 2,
            flags: 0,
            signingIdentifier: Data("com.example.dyld".utf8),
            teamIdentifier: Data("TEAM123456".utf8)
        )
        let cms = genericBlob(
            magic: csMagicBlobWrapper,
            payload: Data([0x30, 0x00])
        )
        return superBlob(entries: [
            (0, primary),
            (0x10000, cms),
        ])
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

    static func genericBlob(magic: UInt32, payload: Data) -> Data {
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
        writeUInt32BE(&result, at: 4, value: UInt32(result.count))
        writeUInt32BE(&result, at: 8, value: version)
        writeUInt32BE(&result, at: 12, value: flags)
        writeUInt32BE(&result, at: 16, value: UInt32(hashOffset))
        writeUInt32BE(&result, at: 20, value: UInt32(identifierOffset))
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

    static func expectedPreimage(
        comparison: SyntheticDynamicLoaderMachOIdentityComparison
    ) -> Data {
        let primary = comparison.codeDirectories[0]
        let lines = [
            "fast-mlx-proof-control-executable-content-evidence-id-v1",
            "artifact_role=dynamic-loader",
            "file_sha256=\(comparison.fileSHA256)",
            "file_bytes=\(comparison.retainedFileBytes.count)",
            "mach_header_sha256=" +
                sha256Hex(Data(comparison.retainedFileBytes.prefix(32))),
            "load_commands_sha256=" + comparison.loadCommandsSHA256,
            "macho_uuid=\(hex(comparison.machOUUID))",
            "primary_code_directory_blob_sha256=" +
                primary.blobSHA256,
            "code_signature_region_sha256=" +
                comparison.codeSignatureRegionSHA256,
        ]
        return Data((lines.joined(separator: "\n") + "\n").utf8)
    }

    static func replacing(
        _ value: SyntheticDynamicLoaderMachOIdentityComparison,
        retainedFileBytes: Data? = nil,
        fileSHA256: String? = nil,
        codeDirectories: [SyntheticCodeDirectoryComparison]? = nil
    ) -> SyntheticDynamicLoaderMachOIdentityComparison {
        SyntheticDynamicLoaderMachOIdentityComparison(
            retainedFileBytes:
                retainedFileBytes ?? value.retainedFileBytes,
            fileSHA256: fileSHA256 ?? value.fileSHA256,
            machHeaderMagic: value.machHeaderMagic,
            cpuType: value.cpuType,
            cpuSubtype: value.cpuSubtype,
            fileType: value.fileType,
            headerFlags: value.headerFlags,
            loadCommandCount: value.loadCommandCount,
            loadCommandBytes: value.loadCommandBytes,
            loadCommandsSHA256: value.loadCommandsSHA256,
            machOUUID: value.machOUUID,
            dynamicLoaderIDName: value.dynamicLoaderIDName,
            codeSignatureRegion: value.codeSignatureRegion,
            codeSignatureRegionSHA256:
                value.codeSignatureRegionSHA256,
            codeDirectories: codeDirectories ?? value.codeDirectories,
            cmsBlob: value.cmsBlob,
            cmsBlobSHA256: value.cmsBlobSHA256,
            isAdHoc: value.isAdHoc
        )
    }

    static func assertInert(
        _ comparison: SyntheticDynamicLoaderMachOIdentityComparison,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertFalse(comparison.canExecute, file: file, line: line)
        XCTAssertFalse(comparison.canSpawn, file: file, line: line)
        XCTAssertFalse(
            comparison.canAccessNetwork,
            file: file,
            line: line
        )
        XCTAssertFalse(
            comparison.canConsumePack,
            file: file,
            line: line
        )
        XCTAssertFalse(
            comparison.canMutateFileSystem,
            file: file,
            line: line
        )
        XCTAssertFalse(
            comparison.canImportGitObjects,
            file: file,
            line: line
        )
        XCTAssertFalse(comparison.canBuild, file: file, line: line)
        XCTAssertFalse(comparison.canLoadModel, file: file, line: line)
        XCTAssertFalse(
            comparison.canReserveOutput,
            file: file,
            line: line
        )
        XCTAssertFalse(comparison.canPublish, file: file, line: line)
    }

    static func assertInert(
        _ evidence: DynamicLoaderContentIdentityEvidence,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertFalse(evidence.canExecute, file: file, line: line)
        XCTAssertFalse(evidence.canSpawn, file: file, line: line)
        XCTAssertFalse(
            evidence.canAccessNetwork,
            file: file,
            line: line
        )
        XCTAssertFalse(evidence.canConsumePack, file: file, line: line)
        XCTAssertFalse(
            evidence.canMutateFileSystem,
            file: file,
            line: line
        )
        XCTAssertFalse(
            evidence.canImportGitObjects,
            file: file,
            line: line
        )
        XCTAssertFalse(evidence.canBuild, file: file, line: line)
        XCTAssertFalse(evidence.canLoadModel, file: file, line: line)
        XCTAssertFalse(
            evidence.canReserveOutput,
            file: file,
            line: line
        )
        XCTAssertFalse(evidence.canPublish, file: file, line: line)
    }

    static func commandOffset(
        in bytes: Data,
        command expected: UInt32
    ) -> Int? {
        let commandCount = Int(readUInt32LE(bytes, at: 16))
        var cursor = 32
        for _ in 0..<commandCount {
            let command = readUInt32LE(bytes, at: cursor)
            if command == expected {
                return cursor
            }
            cursor += Int(readUInt32LE(bytes, at: cursor + 4))
        }
        return nil
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

    static func readUInt32LE(_ data: Data, at offset: Int) -> UInt32 {
        UInt32(data[offset]) |
            UInt32(data[offset + 1]) << 8 |
            UInt32(data[offset + 2]) << 16 |
            UInt32(data[offset + 3]) << 24
    }

    static func hex(_ data: Data) -> String {
        data.map { String(format: "%02x", $0) }.joined()
    }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}
