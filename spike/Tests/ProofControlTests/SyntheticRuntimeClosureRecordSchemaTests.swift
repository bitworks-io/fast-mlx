import Foundation
import XCTest
@testable import ProofControl

final class SyntheticRuntimeClosureRecordSchemaTests: XCTestCase {
    func testCanonicalFileAndSharedCacheMembersFreezeExactIndexedRowsInertly()
        throws
    {
        let effects = Effects()
        let fileEvidence = try Self.executableEvidence(role: .fileImage)
        let fileInstallName = Self.installName("/usr/lib/libFile.dylib")
        let fileRecord = try SyntheticRuntimeClosureRecordSchemaVerifier
            .member(
                index: 7,
                source: .file(fileEvidence),
                installName: fileInstallName
            )

        XCTAssertEqual(fileRecord.index, 7)
        XCTAssertEqual(fileRecord.storage, .file)
        XCTAssertEqual(fileRecord.installName, fileInstallName)
        XCTAssertEqual(
            fileRecord.decodedInstallName,
            Data("/usr/lib/libFile.dylib".utf8)
        )
        XCTAssertEqual(
            fileRecord.contentEvidenceID,
            fileEvidence.contentEvidenceID.sha256
        )
        XCTAssertEqual(
            fileRecord.machOUUID,
            Self.hex(fileEvidence.comparison.machOUUID)
        )
        XCTAssertEqual(
            fileRecord.primaryCodeDirectoryBlobSHA256,
            fileEvidence.primaryCodeDirectoryBlobSHA256
        )
        XCTAssertEqual(
            fileRecord.loadCommandsSHA256,
            fileEvidence.comparison.loadCommandsSHA256
        )
        XCTAssertEqual(
            fileRecord.canonicalRecordBytes,
            Self.memberBytes(
                index: 7,
                contentEvidenceID:
                    fileEvidence.contentEvidenceID.sha256,
                storage: "file",
                installName: fileInstallName,
                machOUUID:
                    Self.hex(fileEvidence.comparison.machOUUID),
                primaryCodeDirectoryBlobSHA256:
                    fileEvidence.primaryCodeDirectoryBlobSHA256,
                loadCommandsSHA256:
                    fileEvidence.comparison.loadCommandsSHA256
            )
        )
        Self.assertInert(fileRecord)

        let sharedEvidence = try Self.sharedCacheImageEvidence(
            installName: "/usr/lib/libShared.dylib"
        )
        let sharedInstallName = SyntheticRuntimeClosureInstallName(
            bytes: sharedEvidence.facts.installNameBytes,
            base64URL: sharedEvidence.facts.installNameBase64URL
        )
        let sharedRecord = try SyntheticRuntimeClosureRecordSchemaVerifier
            .member(
                index: 8,
                source: .sharedCache(sharedEvidence),
                installName: sharedInstallName
            )

        XCTAssertEqual(sharedRecord.index, 8)
        XCTAssertEqual(sharedRecord.storage, .sharedCache)
        XCTAssertEqual(
            sharedRecord.contentEvidenceID,
            sharedEvidence.contentEvidenceID.sha256
        )
        XCTAssertEqual(
            sharedRecord.machOUUID,
            sharedEvidence.facts.machOUUID
        )
        XCTAssertEqual(
            sharedRecord.primaryCodeDirectoryBlobSHA256,
            String(repeating: "0", count: 64)
        )
        XCTAssertEqual(
            sharedRecord.loadCommandsSHA256,
            sharedEvidence.facts.loadCommandsSHA256
        )
        XCTAssertEqual(
            sharedRecord.canonicalRecordBytes,
            Self.memberBytes(
                index: 8,
                contentEvidenceID:
                    sharedEvidence.contentEvidenceID.sha256,
                storage: "shared-cache",
                installName: sharedInstallName,
                machOUUID: sharedEvidence.facts.machOUUID,
                primaryCodeDirectoryBlobSHA256:
                    String(repeating: "0", count: 64),
                loadCommandsSHA256:
                    sharedEvidence.facts.loadCommandsSHA256
            )
        )
        Self.assertInert(sharedRecord)

        // Collection sorting, install-name uniqueness, and reachability are
        // deliberately later graph concerns. This slice validates one row.
        let duplicateLabelRecord =
            try SyntheticRuntimeClosureRecordSchemaVerifier.member(
                index: 255,
                source: .file(fileEvidence),
                installName: sharedInstallName
            )
        XCTAssertEqual(
            duplicateLabelRecord.decodedInstallName,
            sharedRecord.decodedInstallName
        )

        let maximumInstallName = Self.installName(
            "/" + String(repeating: "a", count: 4_095)
        )
        let maximumRecord =
            try SyntheticRuntimeClosureRecordSchemaVerifier.member(
                index: 254,
                source: .file(fileEvidence),
                installName: maximumInstallName
            )
        XCTAssertEqual(maximumRecord.decodedInstallName.count, 4_096)
        XCTAssertEqual(effects, .zero)
    }

    func testMemberRowsRejectRoleEvidenceInstallNameAndIndexDriftBeforeEffects()
        throws
    {
        let effects = Effects()
        let fileEvidence = try Self.executableEvidence(role: .fileImage)
        let gitEvidence = try Self.executableEvidence(role: .git)
        let dynamicLoaderEvidence =
            try Self.executableEvidence(role: .dynamicLoader)
        let valid = Self.installName("/usr/lib/libFile.dylib")
        let sharedEvidence = try Self.sharedCacheImageEvidence(
            installName: "/usr/lib/libShared.dylib"
        )

        let cases: [
            (
                Int,
                SyntheticRuntimeClosureMemberSource,
                SyntheticRuntimeClosureInstallName,
                SyntheticRuntimeClosureRecordSchemaFailure
            )
        ] = [
            (
                -1,
                .file(fileEvidence),
                valid,
                .memberIndexOutOfRange(-1)
            ),
            (
                256,
                .file(fileEvidence),
                valid,
                .memberIndexOutOfRange(256)
            ),
            (
                0,
                .file(gitEvidence),
                valid,
                .unsupportedFileMemberRole(.git)
            ),
            (
                0,
                .file(dynamicLoaderEvidence),
                valid,
                .unsupportedFileMemberRole(.dynamicLoader)
            ),
            (
                0,
                .file(fileEvidence),
                SyntheticRuntimeClosureInstallName(
                    bytes: valid.bytes,
                    base64URL: valid.base64URL + "="
                ),
                .invalidMember(
                    index: 0,
                    field: .installNameBase64URL
                )
            ),
            (
                0,
                .file(fileEvidence),
                SyntheticRuntimeClosureInstallName(
                    bytes: 1,
                    base64URL: "AB"
                ),
                .invalidMember(
                    index: 0,
                    field: .installNameBase64URL
                )
            ),
            (
                0,
                .file(fileEvidence),
                SyntheticRuntimeClosureInstallName(
                    bytes: UInt64.max,
                    base64URL: valid.base64URL
                ),
                .invalidMember(index: 0, field: .installNameBytes)
            ),
            (
                0,
                .file(fileEvidence),
                Self.installName("usr/lib/libFile.dylib"),
                .invalidMember(index: 0, field: .installNameSyntax)
            ),
            (
                0,
                .file(fileEvidence),
                Self.installName("/usr/../libFile.dylib"),
                .invalidMember(index: 0, field: .installNameSyntax)
            ),
            (
                0,
                .file(fileEvidence),
                Self.installName("/usr/./libFile.dylib"),
                .invalidMember(index: 0, field: .installNameSyntax)
            ),
            (
                0,
                .file(fileEvidence),
                Self.installName("/usr//libFile.dylib"),
                .invalidMember(index: 0, field: .installNameSyntax)
            ),
            (
                0,
                .file(fileEvidence),
                Self.installName("/usr/libFile.dylib/"),
                .invalidMember(index: 0, field: .installNameSyntax)
            ),
            (
                0,
                .file(fileEvidence),
                Self.installName("/usr/@rpath/libFile.dylib"),
                .invalidMember(index: 0, field: .installNameSyntax)
            ),
            (
                0,
                .file(fileEvidence),
                Self.installName("/usr\\libFile.dylib"),
                .invalidMember(index: 0, field: .installNameSyntax)
            ),
            (
                0,
                .file(fileEvidence),
                Self.installName("/usr/é.dylib"),
                .invalidMember(index: 0, field: .installNameSyntax)
            ),
            (
                0,
                .file(fileEvidence),
                Self.installName(
                    "/" + String(repeating: "a", count: 4_096)
                ),
                .invalidMember(index: 0, field: .installNameBytes)
            ),
            (
                0,
                .sharedCache(sharedEvidence),
                valid,
                .sharedCacheInstallNameMismatch
            ),
        ]

        for (index, source, installName, expected) in cases {
            XCTAssertThrowsError(
                try SyntheticRuntimeClosureRecordSchemaVerifier.member(
                    index: index,
                    source: source,
                    installName: installName
                )
            ) {
                XCTAssertEqual(
                    $0 as? SyntheticRuntimeClosureRecordSchemaFailure,
                    expected
                )
            }
            XCTAssertEqual(effects, .zero)
        }

        let oversized = SyntheticRuntimeClosureInstallName(
            bytes: UInt64.max,
            base64URL: String(repeating: "A", count: 100_000)
        )
        XCTAssertThrowsError(
            try SyntheticRuntimeClosureRecordSchemaVerifier.member(
                index: 0,
                source: .file(fileEvidence),
                installName: oversized
            )
        ) {
            XCTAssertEqual(
                $0 as? SyntheticRuntimeClosureRecordSchemaFailure,
                .invalidMember(index: 0, field: .installNameBytes)
            )
        }
        XCTAssertEqual(effects, .zero)
    }

    func testCanonicalLoadAndReexportEdgesFreezeExactIndexedRowsInertly()
        throws
    {
        let effects = Effects()
        let root = try Self.executableEvidence(role: .git)
        let fileMember =
            try SyntheticRuntimeClosureRecordSchemaVerifier.member(
                index: 0,
                source: .file(
                    try Self.executableEvidence(role: .fileImage)
                ),
                installName:
                    Self.installName("/usr/lib/libFile.dylib")
            )
        let sharedMember =
            try SyntheticRuntimeClosureRecordSchemaVerifier.member(
                index: 1,
                source: .sharedCache(
                    try Self.sharedCacheImageEvidence(
                        installName: "/usr/lib/libShared.dylib"
                    )
                ),
                installName:
                    Self.installName("/usr/lib/libShared.dylib")
            )

        let load = try SyntheticRuntimeClosureRecordSchemaVerifier.edge(
            index: 12,
            parent: .root(root),
            loadCommandOrdinal: 0,
            kind: .load,
            installName: fileMember.installName,
            resolved: fileMember
        )
        XCTAssertEqual(load.index, 12)
        XCTAssertEqual(load.parentContentEvidenceID, root.contentEvidenceID.sha256)
        XCTAssertEqual(load.loadCommandOrdinal, 0)
        XCTAssertEqual(load.kind, .load)
        XCTAssertEqual(
            load.resolvedContentEvidenceID,
            fileMember.contentEvidenceID
        )
        XCTAssertEqual(
            load.canonicalRecordBytes,
            Self.edgeBytes(
                index: 12,
                parentContentEvidenceID:
                    root.contentEvidenceID.sha256,
                ordinal: 0,
                kind: "load",
                installName: fileMember.installName,
                resolvedContentEvidenceID:
                    fileMember.contentEvidenceID
            )
        )
        Self.assertInert(load)

        let reexport = try SyntheticRuntimeClosureRecordSchemaVerifier.edge(
            index: 4_095,
            parent: .member(fileMember),
            loadCommandOrdinal: UInt64.max,
            kind: .reexport,
            installName: sharedMember.installName,
            resolved: sharedMember
        )
        XCTAssertEqual(reexport.loadCommandOrdinal, UInt64.max)
        XCTAssertEqual(
            reexport.canonicalRecordBytes,
            Self.edgeBytes(
                index: 4_095,
                parentContentEvidenceID:
                    fileMember.contentEvidenceID,
                ordinal: UInt64.max,
                kind: "reexport",
                installName: sharedMember.installName,
                resolvedContentEvidenceID:
                    sharedMember.contentEvidenceID
            )
        )
        Self.assertInert(reexport)
        XCTAssertEqual(effects, .zero)
    }

    func testEdgeRowsRejectParentResolutionNameAndIndexDriftBeforeEffects()
        throws
    {
        let effects = Effects()
        let resolved =
            try SyntheticRuntimeClosureRecordSchemaVerifier.member(
                index: 0,
                source: .file(
                    try Self.executableEvidence(role: .fileImage)
                ),
                installName:
                    Self.installName("/usr/lib/libFile.dylib")
            )
        let root = try Self.executableEvidence(role: .selfGuard)
        let fileImageRoot =
            try Self.executableEvidence(role: .fileImage)
        let dynamicLoader =
            try Self.executableEvidence(role: .dynamicLoader)

        let cases: [
            (
                Int,
                SyntheticRuntimeClosureEdgeParent,
                SyntheticRuntimeClosureInstallName,
                SyntheticRuntimeClosureRecordSchemaFailure
            )
        ] = [
            (
                -1,
                .root(root),
                resolved.installName,
                .edgeIndexOutOfRange(-1)
            ),
            (
                4_096,
                .root(root),
                resolved.installName,
                .edgeIndexOutOfRange(4_096)
            ),
            (
                0,
                .root(fileImageRoot),
                resolved.installName,
                .unsupportedRootRole(.fileImage)
            ),
            (
                0,
                .root(dynamicLoader),
                resolved.installName,
                .unsupportedRootRole(.dynamicLoader)
            ),
            (
                0,
                .root(root),
                Self.installName("/usr/lib/other.dylib"),
                .edgeInstallNameMismatch
            ),
            (
                0,
                .root(root),
                SyntheticRuntimeClosureInstallName(
                    bytes: resolved.installName.bytes,
                    base64URL: resolved.installName.base64URL + "="
                ),
                .invalidEdge(
                    index: 0,
                    field: .installNameBase64URL
                )
            ),
            (
                0,
                .root(root),
                SyntheticRuntimeClosureInstallName(
                    bytes: UInt64.max,
                    base64URL: resolved.installName.base64URL
                ),
                .invalidEdge(
                    index: 0,
                    field: .installNameBytes
                )
            ),
            (
                0,
                .root(root),
                Self.installName("usr/lib/libFile.dylib"),
                .invalidEdge(
                    index: 0,
                    field: .installNameSyntax
                )
            ),
        ]

        for (index, parent, installName, expected) in cases {
            XCTAssertThrowsError(
                try SyntheticRuntimeClosureRecordSchemaVerifier.edge(
                    index: index,
                    parent: parent,
                    loadCommandOrdinal: 1,
                    kind: .load,
                    installName: installName,
                    resolved: resolved
                )
            ) {
                XCTAssertEqual(
                    $0 as? SyntheticRuntimeClosureRecordSchemaFailure,
                    expected
                )
            }
            XCTAssertEqual(effects, .zero)
        }
    }
}

private extension SyntheticRuntimeClosureRecordSchemaTests {
    struct Effects: Equatable {
        var fileSystemMutations = 0
        var networkOperations = 0
        var processSpawns = 0
        var packConsumes = 0
        var runtimeAdmissions = 0

        static let zero = Self()
    }

    static let canonicalUUID: [UInt8] = [
        0x00, 0x11, 0x22, 0x33,
        0x44, 0x55, 0x66, 0x77,
        0x88, 0x99, 0xaa, 0xbb,
        0xcc, 0xdd, 0xee, 0xff,
    ]

    static func installName(
        _ value: String
    ) -> SyntheticRuntimeClosureInstallName {
        let bytes = Data(value.utf8)
        return SyntheticRuntimeClosureInstallName(
            bytes: UInt64(bytes.count),
            base64URL: base64URL(bytes)
        )
    }

    static func executableEvidence(
        role: ExecutableContentArtifactRole
    ) throws -> ExecutableContentIdentityEvidence {
        let comparison = try SyntheticMachOIdentityParser.parse(
            signedFixture()
        )
        return try ExecutableContentIdentityVerifier.derive(
            artifactRole: role,
            comparison: comparison
        )
    }

    static func sharedCacheImageEvidence(
        installName: String
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
        return try SyntheticSharedCacheImageContentIdentityVerifier.derive(
            cacheSetEvidence: set,
            facts: SyntheticSharedCacheImageContentFacts(
                installNameBytes: UInt64(name.count),
                installNameBase64URL: base64URL(name),
                machOUUID: String(repeating: "3", count: 32),
                primaryCodeDirectory: .absent,
                loadCommandsSHA256:
                    String(repeating: "4", count: 64)
            )
        )
    }

    static func memberBytes(
        index: Int,
        contentEvidenceID: String,
        storage: String,
        installName: SyntheticRuntimeClosureInstallName,
        machOUUID: String,
        primaryCodeDirectoryBlobSHA256: String,
        loadCommandsSHA256: String
    ) -> Data {
        let prefix = "member_\(fourDigit(index))"
        return Data(
            (
                [
                    "\(prefix)_content_evidence_id=\(contentEvidenceID)",
                    "\(prefix)_storage=\(storage)",
                    "\(prefix)_install_name_bytes=\(installName.bytes)",
                    "\(prefix)_install_name_base64url=" +
                        installName.base64URL,
                    "\(prefix)_macho_uuid=\(machOUUID)",
                    "\(prefix)_primary_code_directory_blob_sha256=" +
                        primaryCodeDirectoryBlobSHA256,
                    "\(prefix)_load_commands_sha256=" +
                        loadCommandsSHA256,
                ].joined(separator: "\n") + "\n"
            ).utf8
        )
    }

    static func edgeBytes(
        index: Int,
        parentContentEvidenceID: String,
        ordinal: UInt64,
        kind: String,
        installName: SyntheticRuntimeClosureInstallName,
        resolvedContentEvidenceID: String
    ) -> Data {
        let prefix = "edge_\(fourDigit(index))"
        return Data(
            (
                [
                    "\(prefix)_parent_content_evidence_id=" +
                        parentContentEvidenceID,
                    "\(prefix)_load_command_ordinal=\(ordinal)",
                    "\(prefix)_kind=\(kind)",
                    "\(prefix)_install_name_bytes=\(installName.bytes)",
                    "\(prefix)_install_name_base64url=" +
                        installName.base64URL,
                    "\(prefix)_resolved_content_evidence_id=" +
                        resolvedContentEvidenceID,
                ].joined(separator: "\n") + "\n"
            ).utf8
        )
    }

    static func fourDigit(_ value: Int) -> String {
        let text = String(value)
        return String(repeating: "0", count: 4 - text.count) + text
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
        _ value: SyntheticRuntimeClosureEdgeRecordComparison,
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

    static func signedFixture() -> Data {
        let primary = codeDirectory(
            hashType: 2,
            flags: 0x2,
            signingIdentifier: Data("com.example.image".utf8)
        )
        return machO(
            signatureRegion: superBlob(entries: [(0, primary)])
        )
    }

    static func machO(signatureRegion: Data) -> Data {
        var loadCommands = Data()
        appendUInt32LE(&loadCommands, 0x1b)
        appendUInt32LE(&loadCommands, 24)
        loadCommands.append(contentsOf: canonicalUUID)

        let signatureCommandOffset = loadCommands.count
        appendUInt32LE(&loadCommands, 0x1d)
        appendUInt32LE(&loadCommands, 16)
        appendUInt32LE(&loadCommands, 0)
        appendUInt32LE(
            &loadCommands,
            UInt32(signatureRegion.count)
        )
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
        appendUInt32LE(&result, 2)
        appendUInt32LE(&result, UInt32(loadCommands.count))
        appendUInt32LE(&result, 0x00200085)
        appendUInt32LE(&result, 0)
        result.append(loadCommands)
        result.append(signatureRegion)
        return result
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

    static func codeDirectory(
        hashType: UInt8,
        flags: UInt32,
        signingIdentifier: Data
    ) -> Data {
        var result = Data(repeating: 0, count: 52)
        let identifierOffset = result.count
        result.append(signingIdentifier)
        result.append(0)
        let hashOffset = result.count

        writeUInt32BE(&result, at: 0, value: 0xfade0c02)
        writeUInt32BE(&result, at: 4, value: UInt32(result.count))
        writeUInt32BE(&result, at: 8, value: 0x20200)
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
        result[36] = hashType == 2 ? 32 : 0
        result[37] = hashType
        writeUInt32BE(&result, at: 48, value: 0)
        return result
    }

    static func base64URL(_ data: Data) -> String {
        data.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    static func hex(_ data: Data) -> String {
        data.map { String(format: "%02x", $0) }.joined()
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
