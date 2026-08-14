import CryptoKit
import Foundation
import XCTest
@testable import ProofControl

final class SyntheticAcceptedDependencyCommandInventoryTests:
    XCTestCase
{
    func testSharedCanonicalInstallNameVerifierIsTheSingleInventoryGrammar()
        throws
    {
        let name = Self.installName("/usr/lib/libSharedGrammar.dylib")

        XCTAssertEqual(
            try SyntheticRuntimeClosureInstallNameVerifier.validate(name),
            Data("/usr/lib/libSharedGrammar.dylib".utf8)
        )
    }

    func testRootGitSelfGuardAndSharedCacheInventoriesAcceptOnlyLoadAndReexport()
        throws
    {
        let effects = FakeOperationalEffects()
        let loadName = "/usr/lib/libAcceptedLoad.dylib"
        let reexportName = "/usr/lib/libAcceptedReexport.dylib"
        let commands = [
            Self.opaqueCommand(cmd: Self.lcSegment64),
            Self.dylibCommand(cmd: Self.lcLoadDylib, name: loadName),
            Self.dylibCommand(
                cmd: Self.lcReexportDylib,
                name: reexportName
            ),
        ]

        for role in [
            ExecutableContentArtifactRole.git,
            .selfGuard,
        ] {
            let root = try Self.rootEvidence(
                role: role,
                loadCommands: commands
            )
            let inventory =
                try SyntheticAcceptedDependencyCommandInventoryVerifier
                    .root(root)

            XCTAssertEqual(
                inventory.source,
                .root(root)
            )
            XCTAssertEqual(
                inventory.parentContentEvidenceID,
                root.contentEvidenceID.sha256
            )
            XCTAssertEqual(
                inventory.sourceLoadCommandsSHA256,
                root.comparison.loadCommandsSHA256
            )
            XCTAssertEqual(inventory.commandCount, 5)
            XCTAssertEqual(
                inventory.entries.map(\.loadCommandOrdinal),
                [2, 3]
            )
            XCTAssertEqual(
                inventory.entries.map(\.kind),
                [.load, .reexport]
            )
            XCTAssertEqual(
                inventory.entries.map(\.installName),
                [
                    Self.installName(loadName),
                    Self.installName(reexportName),
                ]
            )
            XCTAssertEqual(
                inventory.entries.map(\.decodedInstallName),
                [
                    Data(loadName.utf8),
                    Data(reexportName.utf8),
                ]
            )
            XCTAssertEqual(
                inventory.sourceLoadCommandsSHA256,
                try SyntheticMachOIdentityParser
                    .parse(root.comparison.retainedFileBytes)
                    .loadCommandsSHA256
            )
            Self.assertInert(inventory)
            XCTAssertEqual(effects, .zero)
        }

        let snapshot = try Self.sharedCacheSnapshot(
            loadCommands: commands,
            installName: "/usr/lib/libParent.dylib"
        )
        XCTAssertEqual(snapshot.loadCommandCount, 3)
        XCTAssertEqual(
            snapshot.loadCommandsSHA256,
            Self.sha256Hex(snapshot.loadCommandBytes)
        )
        XCTAssertEqual(
            snapshot.imageEvidence.facts.loadCommandsSHA256,
            snapshot.loadCommandsSHA256
        )
        let shared =
            try SyntheticAcceptedDependencyCommandInventoryVerifier
                .sharedCacheMember(snapshot)

        XCTAssertEqual(shared.source, .sharedCacheMember(snapshot))
        XCTAssertEqual(
            shared.parentContentEvidenceID,
            snapshot.imageEvidence.contentEvidenceID.sha256
        )
        XCTAssertEqual(
            shared.sourceLoadCommandsSHA256,
            snapshot.loadCommandsSHA256
        )
        XCTAssertEqual(shared.commandCount, 3)
        XCTAssertEqual(
            shared.entries.map(\.loadCommandOrdinal),
            [1, 2]
        )
        XCTAssertEqual(
            shared.entries.map(\.kind),
            [.load, .reexport]
        )
        XCTAssertEqual(
            shared.entries.map(\.decodedInstallName),
            [Data(loadName.utf8), Data(reexportName.utf8)]
        )
        Self.assertInert(shared)
        XCTAssertEqual(effects, .zero)
    }

    func testFileMemberEntryPointFailsClosedForCurrentExecutableEvidenceRoles()
        throws
    {
        let effects = FakeOperationalEffects()
        let fileImage = try Self.rootEvidence(
            role: .fileImage,
            loadCommands: []
        )
        XCTAssertThrowsError(
            try SyntheticAcceptedDependencyCommandInventoryVerifier
                .fileMember(fileImage)
        ) {
            XCTAssertEqual(
                $0 as? SyntheticAcceptedDependencyCommandInventoryFailure,
                .unsupportedFileMemberIdentity(.fileImage)
            )
        }
        XCTAssertEqual(effects, .zero)

        for role in [
            ExecutableContentArtifactRole.git,
            .selfGuard,
            .dynamicLoader,
        ] {
            let evidence = try Self.rootEvidence(
                role: role,
                loadCommands: []
            )
            XCTAssertThrowsError(
                try SyntheticAcceptedDependencyCommandInventoryVerifier
                    .fileMember(evidence)
            ) {
                XCTAssertEqual(
                    $0 as?
                        SyntheticAcceptedDependencyCommandInventoryFailure,
                    .unsupportedFileMemberIdentity(role)
                )
            }
            XCTAssertEqual(effects, .zero)
        }
    }

    func testRootEntryPointRejectsUnsupportedRolesBeforeEffects()
        throws
    {
        let effects = FakeOperationalEffects()

        for role in [
            ExecutableContentArtifactRole.fileImage,
            .dynamicLoader,
        ] {
            let evidence = try Self.rootEvidence(
                role: role,
                loadCommands: [
                    Self.dylibCommand(
                        cmd: Self.lcLoadDylib,
                        name: "/usr/lib/libA.dylib"
                    ),
                ]
            )
            XCTAssertThrowsError(
                try SyntheticAcceptedDependencyCommandInventoryVerifier
                    .root(evidence)
            ) {
                XCTAssertEqual(
                    $0 as?
                        SyntheticAcceptedDependencyCommandInventoryFailure,
                    .sourceRole(role)
                )
            }
            XCTAssertEqual(effects, .zero)
        }
    }

    func testZeroAcceptedDependenciesAndMaximumSharedCacheEntriesAreDeterministic()
        throws
    {
        let effects = FakeOperationalEffects()
        let root = try Self.rootEvidence(
            role: .git,
            loadCommands: [
                Self.opaqueCommand(cmd: Self.lcSegment64),
                Self.opaqueCommand(cmd: Self.lcIdDylib),
            ]
        )
        let rootInventory =
            try SyntheticAcceptedDependencyCommandInventoryVerifier
                .root(root)
        XCTAssertEqual(rootInventory.commandCount, 4)
        XCTAssertTrue(rootInventory.entries.isEmpty)
        Self.assertInert(rootInventory)

        let zeroSnapshot = try Self.sharedCacheSnapshot(
            loadCommands: [
                Self.opaqueCommand(cmd: Self.lcSegment64),
                Self.opaqueCommand(cmd: Self.lcIdDylib),
            ],
            installName: "/usr/lib/libZeroParent.dylib"
        )
        let zeroShared =
            try SyntheticAcceptedDependencyCommandInventoryVerifier
                .sharedCacheMember(zeroSnapshot)
        XCTAssertEqual(zeroShared.commandCount, 2)
        XCTAssertTrue(zeroShared.entries.isEmpty)
        Self.assertInert(zeroShared)

        let maximumCommands = (0..<256).map {
            Self.dylibCommand(
                cmd: Self.lcLoadDylib,
                name: "/usr/lib/libAccepted\($0).dylib"
            )
        }
        let maximumSnapshot = try Self.sharedCacheSnapshot(
            loadCommands: maximumCommands,
            installName: "/usr/lib/libMaxParent.dylib"
        )
        let maximum =
            try SyntheticAcceptedDependencyCommandInventoryVerifier
                .sharedCacheMember(maximumSnapshot)
        XCTAssertEqual(maximum.commandCount, 256)
        XCTAssertEqual(maximum.entries.count, 256)
        XCTAssertEqual(
            maximum.entries.first?.loadCommandOrdinal,
            0
        )
        XCTAssertEqual(
            maximum.entries.last?.loadCommandOrdinal,
            255
        )
        XCTAssertEqual(
            maximum.entries.last?.decodedInstallName,
            Data("/usr/lib/libAccepted255.dylib".utf8)
        )
        Self.assertInert(maximum)

        XCTAssertEqual(effects, .zero)
    }

    func testSharedCacheSnapshotContinuityRejectsLengthDigestAndCountDrift()
        throws
    {
        let effects = FakeOperationalEffects()
        let commands = Self.commandRegion([
            Self.dylibCommand(
                cmd: Self.lcLoadDylib,
                name: "/usr/lib/libA.dylib"
            ),
        ])

        let digestDriftEvidence = try Self.sharedCacheImageEvidence(
            installName: "/usr/lib/libDigestParent.dylib",
            loadCommandsSHA256: String(repeating: "0", count: 64)
        )
        XCTAssertThrowsError(
            try SyntheticSharedCacheImageLoadCommandSnapshotVerifier
                .derive(
                    imageEvidence: digestDriftEvidence,
                    loadCommandBytes: commands
                )
        ) {
            XCTAssertEqual(
                $0 as? SyntheticAcceptedDependencyCommandInventoryFailure,
                .snapshotDigestMismatch
            )
        }
        XCTAssertEqual(effects, .zero)

        let oversized = Data(
            repeating: 0,
            count:
                SyntheticMachOIdentityParser.maximumLoadCommandBytes + 8
        )
        let oversizedEvidence = try Self.sharedCacheImageEvidence(
            installName: "/usr/lib/libOversizedParent.dylib",
            loadCommandsSHA256: Self.sha256Hex(oversized)
        )
        XCTAssertThrowsError(
            try SyntheticSharedCacheImageLoadCommandSnapshotVerifier
                .derive(
                    imageEvidence: oversizedEvidence,
                    loadCommandBytes: oversized
                )
        ) {
            XCTAssertEqual(
                $0 as? SyntheticAcceptedDependencyCommandInventoryFailure,
                .snapshotCountOrLength(.regionBytesOutOfRange)
            )
        }
        XCTAssertEqual(effects, .zero)

        let tooManyCommands = Self.commandRegion(
            Array(
                repeating: Self.opaqueCommand(cmd: Self.lcSegment64),
                count: 257
            )
        )
        let tooManyEvidence = try Self.sharedCacheImageEvidence(
            installName: "/usr/lib/libTooManyParent.dylib",
            loadCommandsSHA256: Self.sha256Hex(tooManyCommands)
        )
        XCTAssertThrowsError(
            try SyntheticSharedCacheImageLoadCommandSnapshotVerifier
                .derive(
                    imageEvidence: tooManyEvidence,
                    loadCommandBytes: tooManyCommands
                )
        ) {
            XCTAssertEqual(
                $0 as? SyntheticAcceptedDependencyCommandInventoryFailure,
                .snapshotCountOrLength(.commandCountOutOfRange(257))
            )
        }
        XCTAssertEqual(effects, .zero)
    }

    func testMalformedCommandFramingRangesAndTrailingBytesRejectInOrdinalOrder()
        throws
    {
        let effects = FakeOperationalEffects()
        let cases: [
            (
                Data,
                SyntheticAcceptedDependencyCommandInventoryFailure
            )
        ] = [
            (
                Data([0x0c, 0, 0, 0]),
                .commandFraming(ordinal: 0, field: .header)
            ),
            (
                Self.headerOnlyCommand(cmd: Self.lcLoadDylib, size: 0),
                .commandFraming(ordinal: 0, field: .cmdsizeTooSmall)
            ),
            (
                Self.headerOnlyCommand(cmd: Self.lcLoadDylib, size: 4),
                .commandFraming(ordinal: 0, field: .cmdsizeTooSmall)
            ),
            (
                Self.headerOnlyCommand(cmd: Self.lcLoadDylib, size: 25),
                .commandFraming(ordinal: 0, field: .cmdsizeAlignment)
            ),
            (
                Self.headerOnlyCommand(
                    cmd: Self.lcLoadDylib,
                    size: UInt32.max
                ),
                .commandFraming(ordinal: 0, field: .cmdsizeRange)
            ),
            (
                Self.commandWithTrailingByte(),
                .commandFraming(ordinal: 1, field: .trailingBytes)
            ),
        ]

        for (bytes, expected) in cases {
            let snapshot = try Self.sharedCacheImageEvidence(
                installName: "/usr/lib/libMalformedParent.dylib",
                loadCommandsSHA256: Self.sha256Hex(bytes)
            )
            XCTAssertThrowsError(
                try SyntheticSharedCacheImageLoadCommandSnapshotVerifier
                    .derive(
                        imageEvidence: snapshot,
                        loadCommandBytes: bytes
                    )
            ) {
                XCTAssertEqual(
                    $0 as?
                        SyntheticAcceptedDependencyCommandInventoryFailure,
                    expected
                )
            }
            XCTAssertEqual(effects, .zero)
        }
    }

    func testDylibCommandLayoutNameOffsetNULTerminatorAndPaddingRejectExactly()
        throws
    {
        let effects = FakeOperationalEffects()
        let validName = "/usr/lib/libLayout.dylib"
        let cases: [
            (
                Data,
                SyntheticAcceptedDependencyCommandInventoryFailure
            )
        ] = [
            (
                Self.framedCommand(cmd: Self.lcLoadDylib, size: 16),
                .dependencyCommandLayout(
                    ordinal: 0,
                    field: .dylibCommandSize
                )
            ),
            (
                Self.dylibCommand(
                    cmd: Self.lcLoadDylib,
                    name: validName,
                    nameOffset: 23
                ),
                .dependencyCommandLayout(
                    ordinal: 0,
                    field: .nameOffset
                )
            ),
            (
                Self.dylibCommand(
                    cmd: Self.lcLoadDylib,
                    name: validName,
                    nameOffset: 64
                ),
                .dependencyCommandLayout(
                    ordinal: 0,
                    field: .nameOffset
                )
            ),
            (
                Self.dylibCommand(
                    cmd: Self.lcLoadDylib,
                    name: validName,
                    includeNUL: false
                ),
                .dependencyCommandLayout(
                    ordinal: 0,
                    field: .nameTerminator
                )
            ),
            (
                Self.dylibCommand(
                    cmd: Self.lcLoadDylib,
                    name: validName,
                    paddingByte: 0xff
                ),
                .dependencyCommandLayout(
                    ordinal: 0,
                    field: .zeroPadding
                )
            ),
        ]

        for (bytes, expected) in cases {
            let snapshot = try Self.sharedCacheSnapshot(
                loadCommandBytes: bytes,
                installName: "/usr/lib/libLayoutParent.dylib"
            )
            XCTAssertThrowsError(
                try SyntheticAcceptedDependencyCommandInventoryVerifier
                    .sharedCacheMember(snapshot)
            ) {
                XCTAssertEqual(
                    $0 as?
                        SyntheticAcceptedDependencyCommandInventoryFailure,
                    expected
                )
            }
            XCTAssertEqual(effects, .zero)
        }
    }

    func testCanonicalInstallNameMaximumAndInvalidFormsRejectPrecisely()
        throws
    {
        let effects = FakeOperationalEffects()
        let maximumName = "/" + String(repeating: "a", count: 4_095)
        let maximum = try Self.sharedCacheSnapshot(
            loadCommands: [
                Self.dylibCommand(
                    cmd: Self.lcLoadDylib,
                    name: maximumName
                ),
            ],
            installName: "/usr/lib/libMaxNameParent.dylib"
        )
        let accepted =
            try SyntheticAcceptedDependencyCommandInventoryVerifier
                .sharedCacheMember(maximum)
        XCTAssertEqual(
            accepted.entries.single?.decodedInstallName.count,
            4_096
        )
        XCTAssertEqual(
            accepted.entries.single?.installName,
            Self.installName(maximumName)
        )
        Self.assertInert(accepted)

        let invalidCases: [
            (
                String,
                SyntheticAcceptedDependencyCommandInventoryFailure
            )
        ] = [
            (
                "",
                .installName(ordinal: 0, field: .installNameBytes)
            ),
            (
                "/",
                .installName(ordinal: 0, field: .installNameBytes)
            ),
            (
                "usr/lib/libRelative.dylib",
                .installName(ordinal: 0, field: .installNameSyntax)
            ),
            (
                "@rpath/libToken.dylib",
                .installName(ordinal: 0, field: .installNameSyntax)
            ),
            (
                "/usr/lib/@loader_path/libToken.dylib",
                .installName(ordinal: 0, field: .installNameSyntax)
            ),
            (
                "/usr/lib/lib\\Backslash.dylib",
                .installName(ordinal: 0, field: .installNameSyntax)
            ),
            (
                "/usr//libEmpty.dylib",
                .installName(ordinal: 0, field: .installNameSyntax)
            ),
            (
                "/usr/./libDot.dylib",
                .installName(ordinal: 0, field: .installNameSyntax)
            ),
            (
                "/usr/../libDotDot.dylib",
                .installName(ordinal: 0, field: .installNameSyntax)
            ),
            (
                "/usr/lib/lib Space.dylib",
                .installName(ordinal: 0, field: .installNameSyntax)
            ),
            (
                "/" + String(repeating: "a", count: 4_096),
                .installName(ordinal: 0, field: .installNameBytes)
            ),
        ]

        for (name, expected) in invalidCases {
            let snapshot = try Self.sharedCacheSnapshot(
                loadCommands: [
                    Self.dylibCommand(
                        cmd: Self.lcLoadDylib,
                        name: name
                    ),
                ],
                installName: "/usr/lib/libInvalidNameParent.dylib"
            )
            XCTAssertThrowsError(
                try SyntheticAcceptedDependencyCommandInventoryVerifier
                    .sharedCacheMember(snapshot)
            ) {
                XCTAssertEqual(
                    $0 as?
                        SyntheticAcceptedDependencyCommandInventoryFailure,
                    expected
                )
            }
            XCTAssertEqual(effects, .zero)
        }
    }

    func testWeakUpwardLazyRpathAndDyldEnvironmentCommandsRejectTyped()
        throws
    {
        let effects = FakeOperationalEffects()
        let dependencyName = "/usr/lib/libUnsupported.dylib"
        let cases: [
            (
                Data,
                SyntheticAcceptedDependencyCommandInventoryFailure
            )
        ] = [
            (
                Self.dylibCommand(
                    cmd: Self.lcLoadWeakDylib,
                    name: dependencyName
                ),
                .unsupportedCommand(ordinal: 0, kind: .weakDylib)
            ),
            (
                Self.dylibCommand(
                    cmd: Self.lcLoadUpwardDylib,
                    name: dependencyName
                ),
                .unsupportedCommand(ordinal: 0, kind: .upwardDylib)
            ),
            (
                Self.dylibCommand(
                    cmd: Self.lcLazyLoadDylib,
                    name: dependencyName
                ),
                .unsupportedCommand(ordinal: 0, kind: .lazyDylib)
            ),
            (
                Self.pathCommand(
                    cmd: Self.lcRpath,
                    path: "/usr/lib"
                ),
                .unsupportedCommand(ordinal: 0, kind: .rpath)
            ),
            (
                Self.pathCommand(
                    cmd: Self.lcDyldEnvironment,
                    path: "DYLD_LIBRARY_PATH=/tmp"
                ),
                .unsupportedCommand(
                    ordinal: 0,
                    kind: .dyldEnvironment
                )
            ),
        ]

        for (bytes, expected) in cases {
            let snapshot = try Self.sharedCacheSnapshot(
                loadCommandBytes: bytes,
                installName: "/usr/lib/libUnsupportedParent.dylib"
            )
            XCTAssertThrowsError(
                try SyntheticAcceptedDependencyCommandInventoryVerifier
                    .sharedCacheMember(snapshot)
            ) {
                XCTAssertEqual(
                    $0 as?
                        SyntheticAcceptedDependencyCommandInventoryFailure,
                    expected
                )
            }
            XCTAssertEqual(effects, .zero)
        }
    }

    func testFailureClassesWinBeforeLaterParsingClassesAcrossOrdinals()
        throws
    {
        let effects = FakeOperationalEffects()
        let unsupportedAfterMalformedLayout =
            try Self.sharedCacheSnapshot(
                loadCommands: [
                    Self.framedCommand(
                        cmd: Self.lcLoadDylib,
                        size: 16
                    ),
                    Self.dylibCommand(
                        cmd: Self.lcLoadWeakDylib,
                        name: "/usr/lib/libWeak.dylib"
                    ),
                ],
                installName: "/usr/lib/libOrderingParentA.dylib"
            )
        XCTAssertThrowsError(
            try SyntheticAcceptedDependencyCommandInventoryVerifier
                .sharedCacheMember(unsupportedAfterMalformedLayout)
        ) {
            XCTAssertEqual(
                $0 as? SyntheticAcceptedDependencyCommandInventoryFailure,
                .unsupportedCommand(ordinal: 1, kind: .weakDylib)
            )
        }

        let malformedLayoutAfterInvalidName =
            try Self.sharedCacheSnapshot(
                loadCommands: [
                    Self.dylibCommand(
                        cmd: Self.lcLoadDylib,
                        name: "relative-name.dylib"
                    ),
                    Self.framedCommand(
                        cmd: Self.lcLoadDylib,
                        size: 16
                    ),
                ],
                installName: "/usr/lib/libOrderingParentB.dylib"
            )
        XCTAssertThrowsError(
            try SyntheticAcceptedDependencyCommandInventoryVerifier
                .sharedCacheMember(malformedLayoutAfterInvalidName)
        ) {
            XCTAssertEqual(
                $0 as? SyntheticAcceptedDependencyCommandInventoryFailure,
                .dependencyCommandLayout(
                    ordinal: 1,
                    field: .dylibCommandSize
                )
            )
        }
        XCTAssertEqual(effects, .zero)
    }

    func testSnapshotsAndInventoriesDoNotRetainMutableCallerBytes()
        throws
    {
        let effects = FakeOperationalEffects()
        var bytes = Self.commandRegion([
            Self.dylibCommand(
                cmd: Self.lcLoadDylib,
                name: "/usr/lib/libImmutable.dylib"
            ),
        ])
        let snapshot =
            try SyntheticSharedCacheImageLoadCommandSnapshotVerifier
                .derive(
                    imageEvidence: try Self.sharedCacheImageEvidence(
                        installName: "/usr/lib/libImmutableParent.dylib",
                        loadCommandsSHA256: Self.sha256Hex(bytes)
                    ),
                    loadCommandBytes: bytes
                )
        let inventory =
            try SyntheticAcceptedDependencyCommandInventoryVerifier
                .sharedCacheMember(snapshot)
        let originalSnapshotBytes = snapshot.loadCommandBytes
        let originalEntries = inventory.entries

        bytes[0] = 0xff

        XCTAssertEqual(snapshot.loadCommandBytes, originalSnapshotBytes)
        XCTAssertEqual(inventory.entries, originalEntries)
        XCTAssertEqual(
            inventory.entries.single?.decodedInstallName,
            Data("/usr/lib/libImmutable.dylib".utf8)
        )
        Self.assertInert(snapshot)
        Self.assertInert(inventory)
        XCTAssertEqual(effects, .zero)
    }
}

private extension SyntheticAcceptedDependencyCommandInventoryTests {
    struct FakeOperationalEffects: Equatable {
        var fileSystemCount: UInt64 = 0
        var networkCount: UInt64 = 0
        var processCount: UInt64 = 0
        var packCount: UInt64 = 0
        var objectDatabaseCount: UInt64 = 0
        var capsuleCount: UInt64 = 0
        var buildCount: UInt64 = 0
        var modelCount: UInt64 = 0
        var reservationCount: UInt64 = 0
        var publicationCount: UInt64 = 0

        static let zero = Self()
    }

    static let lcSegment64: UInt32 = 0x19
    static let lcIdDylib: UInt32 = 0x0d
    static let lcLoadDylib: UInt32 = 0x0c
    static let lcLoadWeakDylib: UInt32 = 0x80000018
    static let lcRpath: UInt32 = 0x8000001c
    static let lcReexportDylib: UInt32 = 0x8000001f
    static let lcLazyLoadDylib: UInt32 = 0x20
    static let lcLoadUpwardDylib: UInt32 = 0x80000023
    static let lcDyldEnvironment: UInt32 = 0x27

    static func sharedCacheSnapshot(
        loadCommands: [Data],
        installName: String
    ) throws -> SyntheticSharedCacheImageLoadCommandSnapshot {
        try sharedCacheSnapshot(
            loadCommandBytes: commandRegion(loadCommands),
            installName: installName
        )
    }

    static func sharedCacheSnapshot(
        loadCommandBytes: Data,
        installName: String
    ) throws -> SyntheticSharedCacheImageLoadCommandSnapshot {
        try SyntheticSharedCacheImageLoadCommandSnapshotVerifier
            .derive(
                imageEvidence: sharedCacheImageEvidence(
                    installName: installName,
                    loadCommandsSHA256: sha256Hex(loadCommandBytes)
                ),
                loadCommandBytes: loadCommandBytes
            )
    }

    static func sharedCacheImageEvidence(
        installName: String,
        loadCommandsSHA256: String
    ) throws -> SyntheticSharedCacheImageContentIdentityEvidence {
        let set = try SyntheticSharedCacheSetIdentityVerifier.derive(
            records: [
                SyntheticSharedCacheFileRecord(
                    suffixBytes: 0,
                    suffixBase64URL: "",
                    fileSHA256: String(repeating: "1", count: 64),
                    fileBytes: 4_096,
                    headerUUID: String(repeating: "2", count: 32)
                ),
            ]
        )
        let name = Data(installName.utf8)
        return try SyntheticSharedCacheImageContentIdentityVerifier
            .derive(
                cacheSetEvidence: set,
                facts: SyntheticSharedCacheImageContentFacts(
                    installNameBytes: UInt64(name.count),
                    installNameBase64URL: base64URL(name),
                    machOUUID: String(repeating: "3", count: 32),
                    primaryCodeDirectory: .absent,
                    loadCommandsSHA256: loadCommandsSHA256
                )
            )
    }

    static func rootEvidence(
        role: ExecutableContentArtifactRole,
        loadCommands: [Data]
    ) throws -> ExecutableContentIdentityEvidence {
        let primary = codeDirectory(
            signingIdentifier: Data(
                "com.example.inventory.\(role.rawValue).\(loadCommands.count)"
                    .utf8
            )
        )
        let comparison = try SyntheticMachOIdentityParser.parse(
            machO(
                loadCommands: loadCommands,
                signatureRegion: superBlob(entries: [(0, primary)])
            )
        )
        return try ExecutableContentIdentityVerifier.derive(
            artifactRole: role,
            comparison: comparison
        )
    }

    static func machO(
        loadCommands extraCommands: [Data],
        signatureRegion: Data
    ) -> Data {
        var loadCommands = Data()
        appendUInt32LE(&loadCommands, 0x1b)
        appendUInt32LE(&loadCommands, 24)
        loadCommands.append(contentsOf: [
            0x00, 0x11, 0x22, 0x33,
            0x44, 0x55, 0x66, 0x77,
            0x88, 0x99, 0xaa, 0xbb,
            0xcc, 0xdd, 0xee, 0xff,
        ])
        for command in extraCommands {
            loadCommands.append(command)
        }

        let signatureCommandOffset = loadCommands.count
        appendUInt32LE(&loadCommands, 0x1d)
        appendUInt32LE(&loadCommands, 16)
        appendUInt32LE(&loadCommands, 0)
        appendUInt32LE(&loadCommands, UInt32(signatureRegion.count))
        writeUInt32LE(
            &loadCommands,
            at: signatureCommandOffset + 8,
            value: UInt32(32 + loadCommands.count)
        )

        var result = Data()
        appendUInt32LE(&result, 0xfeedfacf)
        appendUInt32LE(&result, 0x0100000c)
        appendUInt32LE(&result, 2)
        appendUInt32LE(&result, 2)
        appendUInt32LE(&result, UInt32(extraCommands.count + 2))
        appendUInt32LE(&result, UInt32(loadCommands.count))
        appendUInt32LE(&result, 0x00200085)
        appendUInt32LE(&result, 0)
        result.append(loadCommands)
        result.append(signatureRegion)
        return result
    }

    static func commandRegion(_ commands: [Data]) -> Data {
        var result = Data()
        for command in commands {
            result.append(command)
        }
        return result
    }

    static func opaqueCommand(cmd: UInt32) -> Data {
        headerOnlyCommand(cmd: cmd, size: 8)
    }

    static func commandWithTrailingByte() -> Data {
        var result = opaqueCommand(cmd: lcSegment64)
        result.append(0xff)
        return result
    }

    static func headerOnlyCommand(
        cmd: UInt32,
        size: UInt32
    ) -> Data {
        var result = Data()
        appendUInt32LE(&result, cmd)
        appendUInt32LE(&result, size)
        return result
    }

    static func framedCommand(
        cmd: UInt32,
        size: UInt32
    ) -> Data {
        var result = headerOnlyCommand(cmd: cmd, size: size)
        if size > 8 {
            result.append(
                Data(repeating: 0, count: Int(size) - 8)
            )
        }
        return result
    }

    static func dylibCommand(
        cmd: UInt32,
        name: String,
        nameOffset: UInt32 = 24,
        includeNUL: Bool = true,
        paddingByte: UInt8 = 0
    ) -> Data {
        var result = Data()
        appendUInt32LE(&result, cmd)
        appendUInt32LE(&result, 0)
        appendUInt32LE(&result, nameOffset)
        appendUInt32LE(&result, 0)
        appendUInt32LE(&result, 0)
        appendUInt32LE(&result, 0)
        result.append(Data(name.utf8))
        if includeNUL {
            result.append(0)
        }
        while !result.count.isMultiple(of: 8) {
            result.append(paddingByte)
        }
        writeUInt32LE(&result, at: 4, value: UInt32(result.count))
        return result
    }

    static func pathCommand(cmd: UInt32, path: String) -> Data {
        var result = Data()
        appendUInt32LE(&result, cmd)
        appendUInt32LE(&result, 0)
        appendUInt32LE(&result, 12)
        result.append(Data(path.utf8))
        result.append(0)
        while !result.count.isMultiple(of: 8) {
            result.append(0)
        }
        writeUInt32LE(&result, at: 4, value: UInt32(result.count))
        return result
    }

    static func installName(
        _ value: String
    ) -> SyntheticRuntimeClosureInstallName {
        let data = Data(value.utf8)
        return SyntheticRuntimeClosureInstallName(
            bytes: UInt64(data.count),
            base64URL: base64URL(data)
        )
    }

    static func superBlob(entries: [(UInt32, Data)]) -> Data {
        var nextOffset = 12 + entries.count * 8
        var result = Data()
        appendUInt32BE(&result, 0xfade0cc0)
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
        writeUInt32BE(&result, at: 4, value: UInt32(result.count))
        return result
    }

    static func codeDirectory(signingIdentifier: Data) -> Data {
        var result = Data(repeating: 0, count: 52)
        let identifierOffset = result.count
        result.append(signingIdentifier)
        result.append(0)
        let hashOffset = result.count

        writeUInt32BE(&result, at: 0, value: 0xfade0c02)
        writeUInt32BE(&result, at: 4, value: UInt32(result.count))
        writeUInt32BE(&result, at: 8, value: 0x20200)
        writeUInt32BE(&result, at: 12, value: 0x2)
        writeUInt32BE(&result, at: 16, value: UInt32(hashOffset))
        writeUInt32BE(&result, at: 20, value: UInt32(identifierOffset))
        result[36] = 32
        result[37] = 2
        writeUInt32BE(&result, at: 48, value: 0)
        return result
    }

    static func assertInert(
        _ snapshot: SyntheticSharedCacheImageLoadCommandSnapshot,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertFalse(snapshot.canExecute, file: file, line: line)
        XCTAssertFalse(snapshot.canSpawn, file: file, line: line)
        XCTAssertFalse(
            snapshot.canAccessNetwork,
            file: file,
            line: line
        )
        XCTAssertFalse(snapshot.canConsumePack, file: file, line: line)
        XCTAssertFalse(
            snapshot.canMutateFileSystem,
            file: file,
            line: line
        )
        XCTAssertFalse(
            snapshot.canImportGitObjects,
            file: file,
            line: line
        )
        XCTAssertFalse(snapshot.canBuild, file: file, line: line)
        XCTAssertFalse(snapshot.canLoadModel, file: file, line: line)
        XCTAssertFalse(
            snapshot.canReserveOutput,
            file: file,
            line: line
        )
        XCTAssertFalse(snapshot.canPublish, file: file, line: line)
    }

    static func assertInert(
        _ inventory:
            SyntheticAcceptedDependencyCommandInventoryComparison,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertFalse(inventory.canExecute, file: file, line: line)
        XCTAssertFalse(inventory.canSpawn, file: file, line: line)
        XCTAssertFalse(
            inventory.canAccessNetwork,
            file: file,
            line: line
        )
        XCTAssertFalse(inventory.canConsumePack, file: file, line: line)
        XCTAssertFalse(
            inventory.canMutateFileSystem,
            file: file,
            line: line
        )
        XCTAssertFalse(
            inventory.canImportGitObjects,
            file: file,
            line: line
        )
        XCTAssertFalse(inventory.canBuild, file: file, line: line)
        XCTAssertFalse(inventory.canLoadModel, file: file, line: line)
        XCTAssertFalse(
            inventory.canReserveOutput,
            file: file,
            line: line
        )
        XCTAssertFalse(inventory.canPublish, file: file, line: line)
    }

    static func base64URL(_ data: Data) -> String {
        data.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
    }

    static func appendUInt32LE(
        _ data: inout Data,
        _ value: UInt32
    ) {
        data.append(UInt8(truncatingIfNeeded: value))
        data.append(UInt8(truncatingIfNeeded: value >> 8))
        data.append(UInt8(truncatingIfNeeded: value >> 16))
        data.append(UInt8(truncatingIfNeeded: value >> 24))
    }

    static func appendUInt32BE(
        _ data: inout Data,
        _ value: UInt32
    ) {
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
}

extension SyntheticAcceptedDependencyCommandInventoryTests {
    static func fmARootEvidence(
        role: ExecutableContentArtifactRole,
        loadCommands: [Data]
    ) throws -> ExecutableContentIdentityEvidence {
        try rootEvidence(role: role, loadCommands: loadCommands)
    }
}

private extension Array {
    var single: Element? {
        count == 1 ? self[0] : nil
    }
}
