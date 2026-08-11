import CryptoKit
import Foundation
import XCTest
@testable import ProofControl

final class SyntheticFileImageRuntimeClosureMemberSnapshotTests:
    XCTestCase
{
    func testCanonicalSnapshotRowAndInventoryRetainExactD2FactsInertly()
        throws
    {
        var fixture = FMAFileImageFixture.machO(
            identityName: Data(FMAFileImageFixture.memberName.utf8),
            commands: [
                FMAFileImageFixture.dylibCommand(
                    command: FMAFileImageFixture.lcLoadDylib,
                    name: FMAFileImageFixture.loadName
                ),
                FMAFileImageFixture.dylibCommand(
                    command: FMAFileImageFixture.lcReexportDylib,
                    name: FMAFileImageFixture.reexportName
                ),
            ]
        )
        let effects = Effects()
        let comparison = try SyntheticFileImageMachOIdentityParser
            .parse(fixture)
        let evidence = try FileImageContentIdentityVerifier.derive(
            comparison: comparison
        )
        let snapshot = try
            SyntheticFileImageRuntimeClosureMemberSnapshotVerifier
            .derive(fileImageEvidence: evidence)

        XCTAssertEqual(snapshot.fileImageEvidence, evidence)
        XCTAssertEqual(
            snapshot.installName,
            FMAFileImageFixture.installName(
                FMAFileImageFixture.memberName
            )
        )
        XCTAssertEqual(
            snapshot.decodedInstallName,
            Data(FMAFileImageFixture.memberName.utf8)
        )
        XCTAssertEqual(
            try SyntheticFileImageRuntimeClosureMemberSnapshotVerifier
                .derive(fileImageEvidence: evidence),
            snapshot
        )
        Self.assertInert(snapshot)

        let row = try SyntheticRuntimeClosureRecordSchemaVerifier.member(
            index: 7,
            source: .fileImage(snapshot),
            installName: snapshot.installName
        )
        XCTAssertEqual(row.source, .fileImage(snapshot))
        XCTAssertEqual(row.storage, .file)
        XCTAssertEqual(
            row.contentEvidenceID,
            evidence.contentEvidenceID.sha256
        )
        XCTAssertEqual(row.machOUUID, Self.hex(comparison.machOUUID))
        XCTAssertEqual(
            row.primaryCodeDirectoryBlobSHA256,
            evidence.primaryCodeDirectoryBlobSHA256
        )
        XCTAssertEqual(
            row.loadCommandsSHA256,
            comparison.loadCommandsSHA256
        )
        XCTAssertEqual(
            row.canonicalRecordBytes,
            Data(
                (
                    "member_0007_content_evidence_id=" +
                        "2e58401402ee183219e1ce2b7e2ee113" +
                        "a9297aafba067153abf6d9299bbc6073\n" +
                        "member_0007_storage=file\n" +
                        "member_0007_install_name_bytes=30\n" +
                        "member_0007_install_name_base64url=" +
                        "L3Vzci9saWIvbGliRmFzdE1MWFByb29mLmR5bGli\n" +
                        "member_0007_macho_uuid=" +
                        "0123456789abcdef1032547698badcfe\n" +
                        "member_0007_primary_code_directory_blob_sha256=" +
                        "2dcc134ef85a3fc06f6d547c30443d2f" +
                        "554dc50ae837bb985929b3f0edb56906\n" +
                        "member_0007_load_commands_sha256=" +
                        "7b0f03fbdb71b4b2a770e4a3b85a77ec" +
                        "1e2a9c28283307f8e498c4308f241ccb\n"
                ).utf8
            )
        )
        Self.assertInert(row)

        let inventory = try
            SyntheticAcceptedDependencyCommandInventoryVerifier
            .fileImageMember(snapshot)
        XCTAssertEqual(inventory.source, .fileImageMember(snapshot))
        XCTAssertEqual(
            inventory.parentContentEvidenceID,
            evidence.contentEvidenceID.sha256
        )
        XCTAssertEqual(
            inventory.sourceLoadCommandsSHA256,
            comparison.loadCommandsSHA256
        )
        XCTAssertEqual(inventory.commandCount, 5)
        XCTAssertEqual(
            inventory.entries.map(\.loadCommandOrdinal),
            [2, 3]
        )
        XCTAssertEqual(inventory.entries.map(\.kind), [.load, .reexport])
        XCTAssertEqual(
            inventory.entries.map(\.decodedInstallName),
            [
                Data(FMAFileImageFixture.loadName.utf8),
                Data(FMAFileImageFixture.reexportName.utf8),
            ]
        )
        Self.assertInert(inventory)

        fixture[0] = 0
        XCTAssertEqual(
            snapshot.fileImageEvidence.comparison.retainedFileBytes[0],
            UInt8(0xcf)
        )
        XCTAssertEqual(effects, .zero)
    }

    func testSnapshotRejectsEveryNoncanonicalD2IdentityName()
        throws
    {
        let effects = Effects()
        let cases: [
            (Data, SyntheticRuntimeClosureInstallNameFailure)
        ] = [
            (Data("/".utf8), .bytes),
            (Data("usr/lib/libRelative.dylib".utf8), .syntax),
            (Data("@rpath/libToken.dylib".utf8), .syntax),
            (Data("/usr/@loader_path/libToken.dylib".utf8), .syntax),
            (Data([0x2f, 0x75, 0x73, 0x72, 0x2f, 0xff]), .syntax),
            (Data("/usr\\libBackslash.dylib".utf8), .syntax),
            (Data("/usr//libEmpty.dylib".utf8), .syntax),
            (Data("/usr/./libDot.dylib".utf8), .syntax),
            (Data("/usr/../libDotDot.dylib".utf8), .syntax),
        ]

        for (identityName, expected) in cases {
            let evidence = try FMAFileImageFixture.evidence(
                identityName: identityName
            )
            XCTAssertThrowsError(
                try SyntheticFileImageRuntimeClosureMemberSnapshotVerifier
                    .derive(fileImageEvidence: evidence)
            ) {
                XCTAssertEqual(
                    $0 as?
                        SyntheticFileImageRuntimeClosureMemberSnapshotFailure,
                    .identityName(expected)
                )
            }
            XCTAssertEqual(effects, .zero)
        }
    }

    func testMaximumCanonicalIdentityNameRemainsExact() throws {
        let identityName = "/" + String(repeating: "a", count: 4_095)
        let snapshot = try FMAFileImageFixture.snapshot(
            identityName: Data(identityName.utf8)
        )

        XCTAssertEqual(snapshot.installName.bytes, 4_096)
        XCTAssertEqual(snapshot.decodedInstallName.count, 4_096)
        XCTAssertEqual(
            snapshot.decodedInstallName,
            Data(identityName.utf8)
        )
        Self.assertInert(snapshot)
    }

    func testFileImageRowUsesSnapshotLabelOnlyAndKeepsValidationOrder()
        throws
    {
        let effects = Effects()
        let snapshot = try FMAFileImageFixture.snapshot()
        let alternate = FMAFileImageFixture.installName(
            "/usr/lib/libAlternate.dylib"
        )

        XCTAssertThrowsError(
            try SyntheticRuntimeClosureRecordSchemaVerifier.member(
                index: 0,
                source: .fileImage(snapshot),
                installName: alternate
            )
        ) {
            XCTAssertEqual(
                $0 as? SyntheticRuntimeClosureRecordSchemaFailure,
                .fileImageInstallNameMismatch
            )
        }

        XCTAssertThrowsError(
            try SyntheticRuntimeClosureRecordSchemaVerifier.member(
                index: 0,
                source: .fileImage(snapshot),
                installName: SyntheticRuntimeClosureInstallName(
                    bytes: 1,
                    base64URL: "Lw"
                )
            )
        ) {
            XCTAssertEqual(
                $0 as? SyntheticRuntimeClosureRecordSchemaFailure,
                .invalidMember(index: 0, field: .installNameBytes)
            )
        }
        XCTAssertEqual(effects, .zero)
    }

    func testFileImageInventoryPreservesExistingDependencyRefusals()
        throws
    {
        let effects = Effects()
        let cases: [
            (
                Data,
                SyntheticAcceptedDependencyCommandInventoryFailure
            )
        ] = [
            (
                FMAFileImageFixture.dylibCommand(
                    command: FMAFileImageFixture.lcLoadWeakDylib,
                    name: FMAFileImageFixture.loadName
                ),
                .unsupportedCommand(ordinal: 2, kind: .weakDylib)
            ),
            (
                FMAFileImageFixture.dylibCommand(
                    command: FMAFileImageFixture.lcLoadUpwardDylib,
                    name: FMAFileImageFixture.loadName
                ),
                .unsupportedCommand(ordinal: 2, kind: .upwardDylib)
            ),
            (
                FMAFileImageFixture.dylibCommand(
                    command: FMAFileImageFixture.lcLazyLoadDylib,
                    name: FMAFileImageFixture.loadName
                ),
                .unsupportedCommand(ordinal: 2, kind: .lazyDylib)
            ),
            (
                FMAFileImageFixture.headerOnlyCommand(
                    command: FMAFileImageFixture.lcRpath
                ),
                .unsupportedCommand(ordinal: 2, kind: .rpath)
            ),
            (
                FMAFileImageFixture.headerOnlyCommand(
                    command: FMAFileImageFixture.lcLoadDylib
                ),
                .dependencyCommandLayout(
                    ordinal: 2,
                    field: .dylibCommandSize
                )
            ),
        ]

        for (command, expected) in cases {
            let snapshot = try FMAFileImageFixture.snapshot(
                commands: [command]
            )
            XCTAssertThrowsError(
                try SyntheticAcceptedDependencyCommandInventoryVerifier
                    .fileImageMember(snapshot)
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

    func testV1StageBRefusesSealedFileImageBeforeInventoryMatching()
        throws
    {
        let effects = Effects()
        let snapshot = try FMAFileImageFixture.snapshot()
        let member = try SyntheticRuntimeClosureRecordSchemaVerifier.member(
            index: 0,
            source: .fileImage(snapshot),
            installName: snapshot.installName
        )
        let rootCommand = FMAFileImageFixture.dylibCommand(
            command: FMAFileImageFixture.lcLoadDylib,
            name: FMAFileImageFixture.memberName
        )
        let root = try
            SyntheticAcceptedDependencyCommandInventoryTests
            .fmARootEvidence(role: .git, loadCommands: [rootCommand])
        let rootInventory = try
            SyntheticAcceptedDependencyCommandInventoryVerifier.root(root)
        let rootEntry = try XCTUnwrap(rootInventory.entries.first)
        let edge = try SyntheticRuntimeClosureRecordSchemaVerifier.edge(
            index: 0,
            parent: .root(root),
            loadCommandOrdinal: rootEntry.loadCommandOrdinal,
            kind: rootEntry.kind,
            installName: member.installName,
            resolved: member
        )
        let collection = try
            SyntheticRuntimeClosureRecordCollectionVerifier.derive(
                members: [member],
                edges: [edge]
            )

        XCTAssertThrowsError(
            try SyntheticRuntimeClosureGraphVerifier.compare(
                root: root,
                members: [member],
                edges: [edge],
                collection: collection,
                rootInventory: rootInventory,
                memberInventories: []
            )
        ) {
            XCTAssertEqual(
                $0 as? SyntheticRuntimeClosureGraphFailure,
                .unsupportedFileMemberIdentity(memberIndex: 0)
            )
        }
        XCTAssertEqual(effects, .zero)
    }
}

enum FMAFileImageFixture {
    static let memberName = "/usr/lib/libFastMLXProof.dylib"
    static let loadName = "/usr/lib/libAcceptedLoad.dylib"
    static let reexportName = "/usr/lib/libAcceptedReexport.dylib"

    static let lcLoadDylib: UInt32 = 0x0c
    static let lcLoadWeakDylib: UInt32 = 0x80000018
    static let lcRpath: UInt32 = 0x8000001c
    static let lcReexportDylib: UInt32 = 0x8000001f
    static let lcLazyLoadDylib: UInt32 = 0x20
    static let lcLoadUpwardDylib: UInt32 = 0x80000023

    static func evidence(
        identityName: Data = Data(memberName.utf8),
        commands: [Data] = []
    ) throws -> FileImageContentIdentityEvidence {
        try FileImageContentIdentityVerifier.derive(
            comparison: SyntheticFileImageMachOIdentityParser.parse(
                machO(identityName: identityName, commands: commands)
            )
        )
    }

    static func snapshot(
        identityName: Data = Data(memberName.utf8),
        commands: [Data] = []
    ) throws -> SyntheticFileImageRuntimeClosureMemberSnapshot {
        try SyntheticFileImageRuntimeClosureMemberSnapshotVerifier
            .derive(
                fileImageEvidence: evidence(
                    identityName: identityName,
                    commands: commands
                )
            )
    }

    static func machO(
        identityName: Data,
        commands: [Data]
    ) -> Data {
        SyntheticFileImageMachOIdentityTests.fmAFileImageMachO(
            identityName: identityName,
            extraCommands: commands
        )
    }

    static func installName(
        _ value: String
    ) -> SyntheticRuntimeClosureInstallName {
        let bytes = Data(value.utf8)
        return SyntheticRuntimeClosureInstallName(
            bytes: UInt64(bytes.count),
            base64URL: base64URL(bytes)
        )
    }

    static func dylibCommand(command: UInt32, name: String) -> Data {
        var result = Data()
        appendUInt32LE(&result, command)
        appendUInt32LE(&result, 0)
        appendUInt32LE(&result, 24)
        appendUInt32LE(&result, 0)
        appendUInt32LE(&result, 0)
        appendUInt32LE(&result, 0)
        result.append(Data(name.utf8))
        result.append(0)
        while !result.count.isMultiple(of: 8) {
            result.append(0)
        }
        writeUInt32LE(
            &result,
            at: 4,
            value: UInt32(result.count)
        )
        return result
    }

    static func headerOnlyCommand(command: UInt32) -> Data {
        var result = Data()
        appendUInt32LE(&result, command)
        appendUInt32LE(&result, 8)
        return result
    }

    private static func base64URL(_ data: Data) -> String {
        data.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    private static func appendUInt32LE(
        _ data: inout Data,
        _ value: UInt32
    ) {
        data.append(UInt8(truncatingIfNeeded: value))
        data.append(UInt8(truncatingIfNeeded: value >> 8))
        data.append(UInt8(truncatingIfNeeded: value >> 16))
        data.append(UInt8(truncatingIfNeeded: value >> 24))
    }

    private static func writeUInt32LE(
        _ data: inout Data,
        at offset: Int,
        value: UInt32
    ) {
        data[offset] = UInt8(truncatingIfNeeded: value)
        data[offset + 1] = UInt8(truncatingIfNeeded: value >> 8)
        data[offset + 2] = UInt8(truncatingIfNeeded: value >> 16)
        data[offset + 3] = UInt8(truncatingIfNeeded: value >> 24)
    }
}

private extension SyntheticFileImageRuntimeClosureMemberSnapshotTests {
    struct Effects: Equatable {
        var spawnCount = 0
        var networkCount = 0
        var fileSystemCount = 0
        var packCount = 0
        var objectDatabaseCount = 0
        var sourceMutationCount = 0
        var buildCount = 0
        var modelCount = 0
        var reservationCount = 0
        var publicationCount = 0

        static let zero = Self()
    }

    static func hex(_ data: Data) -> String {
        data.map { String(format: "%02x", $0) }.joined()
    }

    static func assertInert(
        _ value: SyntheticFileImageRuntimeClosureMemberSnapshot,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertFalse(value.canExecute, file: file, line: line)
        XCTAssertFalse(value.canSpawn, file: file, line: line)
        XCTAssertFalse(value.canAccessNetwork, file: file, line: line)
        XCTAssertFalse(value.canConsumePack, file: file, line: line)
        XCTAssertFalse(value.canMutateFileSystem, file: file, line: line)
        XCTAssertFalse(value.canImportGitObjects, file: file, line: line)
        XCTAssertFalse(value.canBuild, file: file, line: line)
        XCTAssertFalse(value.canLoadModel, file: file, line: line)
        XCTAssertFalse(value.canReserveOutput, file: file, line: line)
        XCTAssertFalse(value.canPublish, file: file, line: line)
    }

    static func assertInert(
        _ value: SyntheticRuntimeClosureMemberRecordComparison,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertFalse(value.canExecute, file: file, line: line)
        XCTAssertFalse(value.canSpawn, file: file, line: line)
        XCTAssertFalse(value.canAccessNetwork, file: file, line: line)
        XCTAssertFalse(value.canConsumePack, file: file, line: line)
        XCTAssertFalse(value.canMutateFileSystem, file: file, line: line)
        XCTAssertFalse(value.canImportGitObjects, file: file, line: line)
        XCTAssertFalse(value.canBuild, file: file, line: line)
        XCTAssertFalse(value.canLoadModel, file: file, line: line)
        XCTAssertFalse(value.canReserveOutput, file: file, line: line)
        XCTAssertFalse(value.canPublish, file: file, line: line)
    }

    static func assertInert(
        _ value: SyntheticAcceptedDependencyCommandInventoryComparison,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertFalse(value.canExecute, file: file, line: line)
        XCTAssertFalse(value.canSpawn, file: file, line: line)
        XCTAssertFalse(value.canAccessNetwork, file: file, line: line)
        XCTAssertFalse(value.canConsumePack, file: file, line: line)
        XCTAssertFalse(value.canMutateFileSystem, file: file, line: line)
        XCTAssertFalse(value.canImportGitObjects, file: file, line: line)
        XCTAssertFalse(value.canBuild, file: file, line: line)
        XCTAssertFalse(value.canLoadModel, file: file, line: line)
        XCTAssertFalse(value.canReserveOutput, file: file, line: line)
        XCTAssertFalse(value.canPublish, file: file, line: line)
    }
}
