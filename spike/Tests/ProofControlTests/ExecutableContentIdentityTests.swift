import CryptoKit
import Darwin
import Foundation
import XCTest
@testable import ProofControl

final class ExecutableContentIdentityTests: XCTestCase {
    private var caseRoot: URL!

    override func setUpWithError() throws {
        let canonicalTemporaryPath = try XCTUnwrap(
            Darwin.realpath(NSTemporaryDirectory(), nil)
        )
        defer { Darwin.free(canonicalTemporaryPath) }

        caseRoot = URL(
            fileURLWithPath: String(cString: canonicalTemporaryPath),
            isDirectory: true
        )
        .appendingPathComponent(
            "fast-mlx-executable-content-\(UUID().uuidString)"
        )
        try FileManager.default.createDirectory(
            at: caseRoot,
            withIntermediateDirectories: false
        )
    }

    override func tearDownWithError() throws {
        if let caseRoot {
            try? FileManager.default.removeItem(at: caseRoot)
        }
    }

    func testCanonicalContentIdentityMatchesExactExpectationOnlyInertly()
        throws
    {
        var fixture = Self.signedFixture()
        let comparison = try SyntheticMachOIdentityParser.parse(fixture)
        let expectation = try anchoredExpectation(for: comparison)
        let effects = FakeOperationalEffects()

        let evidence = try ExecutableContentIdentityVerifier.derive(
            artifactRole: .git,
            comparison: comparison
        )
        let expectedPreimage = Self.expectedPreimage(
            role: .git,
            comparison: comparison
        )

        XCTAssertEqual(evidence.artifactRole, .git)
        XCTAssertEqual(evidence.identityPreimage, expectedPreimage)
        XCTAssertEqual(
            evidence.contentEvidenceID.sha256,
            Self.sha256Hex(expectedPreimage)
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

        let matched = try ExecutableContentIdentityVerifier.match(
            evidence: evidence,
            expectation: expectation
        )
        XCTAssertEqual(matched.contentEvidence, evidence)
        XCTAssertEqual(matched.expectation, expectation)
        Self.assertInert(matched)

        fixture[0] = 0
        XCTAssertEqual(
            evidence.comparison.retainedFileBytes,
            comparison.retainedFileBytes
        )
        XCTAssertEqual(evidence.identityPreimage, expectedPreimage)
        XCTAssertEqual(effects, .zero)
    }

    func testEveryDomainRoleHasDistinctCanonicalNoncircularIdentity()
        throws
    {
        let comparison = try SyntheticMachOIdentityParser.parse(
            Self.signedFixture()
        )
        let roles: [ExecutableContentArtifactRole] = [
            .git,
            .selfGuard,
            .fileImage,
            .dynamicLoader,
        ]
        let effects = FakeOperationalEffects()

        let evidence = try roles.map {
            try ExecutableContentIdentityVerifier.derive(
                artifactRole: $0,
                comparison: comparison
            )
        }

        XCTAssertEqual(
            Set(evidence.map(\.contentEvidenceID.sha256)).count,
            roles.count
        )
        for (role, value) in zip(roles, evidence) {
            XCTAssertEqual(value.artifactRole, role)
            XCTAssertEqual(
                value.identityPreimage,
                Self.expectedPreimage(
                    role: role,
                    comparison: comparison
                )
            )
            let text = String(
                decoding: value.identityPreimage,
                as: UTF8.self
            )
            XCTAssertTrue(
                text.contains("artifact_role=\(role.rawValue)\n")
            )
            for forbidden in [
                "expectation",
                "claim",
                "tool_policy",
                "runtime",
                "filesystem",
                "path",
                "content_evidence_id=",
                "transcript",
                "closure",
                "cache",
                "member",
                "edge",
            ] {
                XCTAssertFalse(text.contains(forbidden))
            }
            Self.assertInert(value)
        }

        let firstExpectation = try anchoredExpectation(
            for: comparison,
            evidenceGeneration: 9
        )
        let secondExpectation = try anchoredExpectation(
            for: comparison,
            evidenceGeneration: 10
        )
        let firstMatch = try ExecutableContentIdentityVerifier.match(
            evidence: evidence[0],
            expectation: firstExpectation
        )
        let secondMatch = try ExecutableContentIdentityVerifier.match(
            evidence: evidence[0],
            expectation: secondExpectation
        )
        XCTAssertEqual(
            firstMatch.contentEvidence.contentEvidenceID,
            secondMatch.contentEvidence.contentEvidenceID
        )
        XCTAssertNotEqual(
            firstMatch.expectation.documentSHA256,
            secondMatch.expectation.documentSHA256
        )

        for value in evidence.suffix(2) {
            XCTAssertThrowsError(
                try ExecutableContentIdentityVerifier.match(
                    evidence: value,
                    expectation: firstExpectation
                )
            ) { error in
                XCTAssertEqual(
                    error as? ExecutableContentIdentityFailure,
                    .expectationRoleUnsupported(value.artifactRole)
                )
            }
            XCTAssertEqual(effects, .zero)
        }
    }

    func testEveryExpectationDriftRejectsBeforeEffects() throws {
        let comparison = try SyntheticMachOIdentityParser.parse(
            Self.signedFixture()
        )
        let evidence = try ExecutableContentIdentityVerifier.derive(
            artifactRole: .git,
            comparison: comparison
        )
        let effects = FakeOperationalEffects()
        let cases: [
            (
                ExpectationMutation,
                ExecutableContentExpectationField
            )
        ] = [
            (.role, .artifactRole),
            (.fileSHA256, .fileSHA256),
            (.fileBytes, .fileBytes),
            (.cpuSubtype, .cpuSubtype),
            (.headerFlags, .headerFlags),
            (.loadCommandCount, .loadCommandCount),
            (.loadCommandBytes, .loadCommandBytes),
            (.loadCommandsSHA256, .loadCommandsSHA256),
            (.machOUUID, .machOUUID),
            (
                .codeSignatureRegionSHA256,
                .codeSignatureRegionSHA256
            ),
            (
                .codeSignatureRegionBytes,
                .codeSignatureRegionBytes
            ),
            (.codeDirectoryCount, .codeDirectoryCount),
            (
                .codeDirectorySlot,
                .codeDirectory(index: 1, field: .slot)
            ),
            (
                .codeDirectoryBlobSHA256,
                .codeDirectory(index: 0, field: .blobSHA256)
            ),
            (
                .codeDirectoryBlobBytes,
                .codeDirectory(index: 0, field: .blobBytes)
            ),
            (
                .codeDirectoryHashType,
                .codeDirectory(index: 0, field: .hashType)
            ),
            (
                .codeDirectoryFlags,
                .codeDirectory(index: 0, field: .flags)
            ),
            (
                .signingIdentifierBytes,
                .codeDirectory(
                    index: 0,
                    field: .signingIdentifierBytes
                )
            ),
            (
                .signingIdentifierBase64URL,
                .codeDirectory(
                    index: 0,
                    field: .signingIdentifierBase64URL
                )
            ),
            (
                .teamIdentifierBytes,
                .codeDirectory(
                    index: 0,
                    field: .teamIdentifierBytes
                )
            ),
            (
                .teamIdentifierBase64URL,
                .codeDirectory(
                    index: 0,
                    field: .teamIdentifierBase64URL
                )
            ),
            (.cmsBlobSHA256, .cmsBlobSHA256),
            (.cmsBlobBytes, .cmsBlobBytes),
        ]

        for (mutation, expectedField) in cases {
            let expectation = try anchoredExpectation(
                for: comparison,
                mutation: mutation
            )
            XCTAssertThrowsError(
                try ExecutableContentIdentityVerifier.match(
                    evidence: evidence,
                    expectation: expectation
                ),
                "mutation \(mutation) unexpectedly matched"
            ) { error in
                XCTAssertEqual(
                    error as? ExecutableContentIdentityFailure,
                    .expectationMismatch(expectedField)
                )
            }
            XCTAssertEqual(effects, .zero)
        }
    }

    func testMissingPrimaryAndForgedParserFactsRejectBeforeEffects()
        throws
    {
        let comparison = try SyntheticMachOIdentityParser.parse(
            Self.signedFixture()
        )
        let effects = FakeOperationalEffects()

        let missingPrimary = Self.replacing(
            comparison,
            codeDirectories:
                Array(comparison.codeDirectories.dropFirst())
        )
        XCTAssertThrowsError(
            try ExecutableContentIdentityVerifier.derive(
                artifactRole: .git,
                comparison: missingPrimary
            )
        ) { error in
            XCTAssertEqual(
                error as? ExecutableContentIdentityFailure,
                .missingPrimaryCodeDirectory
            )
        }

        let forgedDigest = Self.replacing(
            comparison,
            fileSHA256: Self.otherSHA256
        )
        XCTAssertThrowsError(
            try ExecutableContentIdentityVerifier.derive(
                artifactRole: .git,
                comparison: forgedDigest
            )
        ) { error in
            XCTAssertEqual(
                error as? ExecutableContentIdentityFailure,
                .parserFactMismatch
            )
        }

        let selfGuard = try ExecutableContentIdentityVerifier.derive(
            artifactRole: .selfGuard,
            comparison: comparison
        )
        let gitExpectation = try anchoredExpectation(for: comparison)
        XCTAssertThrowsError(
            try ExecutableContentIdentityVerifier.match(
                evidence: selfGuard,
                expectation: gitExpectation
            )
        ) { error in
            XCTAssertEqual(
                error as? ExecutableContentIdentityFailure,
                .expectationMismatch(.artifactRole)
            )
        }
        XCTAssertEqual(effects, .zero)
    }
}

private extension ExecutableContentIdentityTests {
    enum ExpectationMutation: String {
        case role
        case fileSHA256
        case fileBytes
        case cpuSubtype
        case headerFlags
        case loadCommandCount
        case loadCommandBytes
        case loadCommandsSHA256
        case machOUUID
        case codeSignatureRegionSHA256
        case codeSignatureRegionBytes
        case codeDirectoryCount
        case codeDirectorySlot
        case codeDirectoryBlobSHA256
        case codeDirectoryBlobBytes
        case codeDirectoryHashType
        case codeDirectoryFlags
        case signingIdentifierBytes
        case signingIdentifierBase64URL
        case teamIdentifierBytes
        case teamIdentifierBase64URL
        case cmsBlobSHA256
        case cmsBlobBytes
    }

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
    static let csMagicCodeDirectory: UInt32 = 0xfade0c02
    static let csMagicEmbeddedSignature: UInt32 = 0xfade0cc0
    static let csMagicBlobWrapper: UInt32 = 0xfade0b01
    static let otherSHA256 = String(repeating: "9", count: 64)

    static func signedFixture() -> Data {
        let primary = codeDirectory(
            hashType: 2,
            flags: 0,
            signingIdentifier: Data("com.example.git".utf8),
            teamIdentifier: Data("TEAM123456".utf8)
        )
        let alternate = codeDirectory(
            hashType: 4,
            flags: 0,
            signingIdentifier: Data("com.example.git.alt".utf8),
            teamIdentifier: Data()
        )
        let cms = genericBlob(
            magic: csMagicBlobWrapper,
            payload: Data([0x30, 0x03, 0x02, 0x01, 0x01])
        )
        return machO(
            signatureRegion: superBlob(entries: [
                (0, primary),
                (0x1000, alternate),
                (0x10000, cms),
            ])
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
        let teamOffset: Int
        if teamIdentifier.isEmpty {
            teamOffset = 0
        } else {
            teamOffset = result.count
            result.append(teamIdentifier)
            result.append(0)
        }
        let hashOffset = result.count

        writeUInt32BE(&result, at: 0, value: csMagicCodeDirectory)
        writeUInt32BE(
            &result,
            at: 4,
            value: UInt32(result.count)
        )
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
        writeUInt32BE(&result, at: 24, value: 0)
        writeUInt32BE(&result, at: 28, value: 0)
        writeUInt32BE(&result, at: 32, value: 0)
        result[36] = expectedHashSize(hashType)
        result[37] = hashType
        result[38] = 0
        result[39] = 0
        writeUInt32BE(&result, at: 40, value: 0)
        writeUInt32BE(&result, at: 44, value: 0)
        writeUInt32BE(
            &result,
            at: 48,
            value: UInt32(teamOffset)
        )
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
            preconditionFailure("unsupported test hash type")
        }
    }

    static func fields(
        for comparison: SyntheticMachOIdentityComparison,
        evidenceGeneration: UInt64,
        mutation: ExpectationMutation?
    ) -> ExecutableIdentityExpectationFields {
        var role: ExecutableIdentityArtifactRole = .git
        var fileSHA256 = comparison.fileSHA256
        var fileBytes = UInt64(comparison.retainedFileBytes.count)
        var cpuSubtype = hex8(comparison.cpuSubtype)
        var headerFlags = hex8(comparison.headerFlags)
        var loadCommandCount = UInt64(comparison.loadCommandCount)
        var loadCommandBytes = UInt64(
            comparison.loadCommandBytes.count
        )
        var loadCommandsSHA256 = comparison.loadCommandsSHA256
        var machOUUID = hex(comparison.machOUUID)
        var codeSignatureRegionSHA256 =
            comparison.codeSignatureRegionSHA256
        var codeSignatureRegionBytes = UInt64(
            comparison.codeSignatureRegion.count
        )
        var directories = comparison.codeDirectories.map {
            ExecutableIdentityCodeDirectoryExpectation(
                slot: UInt64($0.slot),
                blobSHA256: $0.blobSHA256,
                blobBytes: UInt64($0.blob.count),
                hashType: UInt64($0.hashType),
                flags: hex8($0.flags),
                signingIdentifierBytes:
                    UInt64($0.signingIdentifier.count),
                signingIdentifierBase64URL:
                    base64URL($0.signingIdentifier),
                teamIdentifierBytes:
                    UInt64($0.teamIdentifier.count),
                teamIdentifierBase64URL:
                    base64URL($0.teamIdentifier)
            )
        }
        var cmsBlobSHA256 = comparison.cmsBlobSHA256
        var cmsBlobBytes = UInt64(comparison.cmsBlob?.count ?? 0)

        switch mutation {
        case nil:
            break
        case .role:
            role = .selfGuard
        case .fileSHA256:
            fileSHA256 = otherSHA256
        case .fileBytes:
            fileBytes += 1
        case .cpuSubtype:
            cpuSubtype = "00000003"
        case .headerFlags:
            headerFlags = "00000000"
        case .loadCommandCount:
            loadCommandCount += 1
        case .loadCommandBytes:
            loadCommandBytes += 1
        case .loadCommandsSHA256:
            loadCommandsSHA256 = otherSHA256
        case .machOUUID:
            machOUUID = String(repeating: "1", count: 32)
        case .codeSignatureRegionSHA256:
            codeSignatureRegionSHA256 = otherSHA256
        case .codeSignatureRegionBytes:
            codeSignatureRegionBytes += 1
        case .codeDirectoryCount:
            directories.removeLast()
        case .codeDirectorySlot:
            let value = directories[1]
            directories[1] = ExecutableIdentityCodeDirectoryExpectation(
                slot: 4_097,
                blobSHA256: value.blobSHA256,
                blobBytes: value.blobBytes,
                hashType: value.hashType,
                flags: value.flags,
                signingIdentifierBytes:
                    value.signingIdentifierBytes,
                signingIdentifierBase64URL:
                    value.signingIdentifierBase64URL,
                teamIdentifierBytes: value.teamIdentifierBytes,
                teamIdentifierBase64URL:
                    value.teamIdentifierBase64URL
            )
        case .codeDirectoryBlobSHA256:
            directories[0] = replacing(
                directories[0],
                blobSHA256: otherSHA256
            )
        case .codeDirectoryBlobBytes:
            directories[0] = replacing(
                directories[0],
                blobBytes: directories[0].blobBytes + 1
            )
        case .codeDirectoryHashType:
            directories[0] = replacing(
                directories[0],
                hashType: 4
            )
        case .codeDirectoryFlags:
            directories[0] = replacing(
                directories[0],
                flags: "00000002"
            )
        case .signingIdentifierBytes:
            let value = Data("different-length".utf8)
            directories[0] = replacing(
                directories[0],
                signingIdentifier: value
            )
        case .signingIdentifierBase64URL:
            let value = Data("dom.example.git".utf8)
            precondition(
                value.count ==
                    comparison.codeDirectories[0]
                        .signingIdentifier.count
            )
            directories[0] = replacing(
                directories[0],
                signingIdentifier: value
            )
        case .teamIdentifierBytes:
            let value = Data("TEAM-DIFFERENT".utf8)
            directories[0] = replacing(
                directories[0],
                teamIdentifier: value
            )
        case .teamIdentifierBase64URL:
            let value = Data("TEAM654321".utf8)
            precondition(
                value.count ==
                    comparison.codeDirectories[0]
                        .teamIdentifier.count
            )
            directories[0] = replacing(
                directories[0],
                teamIdentifier: value
            )
        case .cmsBlobSHA256:
            cmsBlobSHA256 = otherSHA256
        case .cmsBlobBytes:
            cmsBlobBytes += 1
        }

        return ExecutableIdentityExpectationFields(
            evidenceGeneration: evidenceGeneration,
            validFromUnixSeconds: 1_900_000_000,
            validUntilUnixSeconds: 2_100_000_000,
            artifactRole: role,
            fileSHA256: fileSHA256,
            fileBytes: fileBytes,
            cpuSubtype: cpuSubtype,
            headerFlags: headerFlags,
            loadCommandCount: loadCommandCount,
            loadCommandBytes: loadCommandBytes,
            loadCommandsSHA256: loadCommandsSHA256,
            machOUUID: machOUUID,
            codeSignatureRegionSHA256:
                codeSignatureRegionSHA256,
            codeSignatureRegionBytes:
                codeSignatureRegionBytes,
            codeDirectories: directories,
            cmsBlobSHA256: cmsBlobSHA256,
            cmsBlobBytes: cmsBlobBytes
        )
    }

    func anchoredExpectation(
        for comparison: SyntheticMachOIdentityComparison,
        evidenceGeneration: UInt64 = 9,
        mutation: ExpectationMutation? = nil
    ) throws -> AnchoredExecutableIdentityExpectationDocument {
        let fields = Self.fields(
            for: comparison,
            evidenceGeneration: evidenceGeneration,
            mutation: mutation
        )
        let bytes = try ExecutableIdentityExpectationVerifier
            .documentBytes(fields: fields)
        let url = caseRoot.appendingPathComponent(
            "\(UUID().uuidString).executable-expectation"
        )
        try bytes.write(to: url)
        let file = try AdmittedFile.capture(
            absolutePath: url.path,
            maximumBytes: 32 * 1024
        )
        return try ExecutableIdentityExpectationVerifier.anchor(
            expectationFile: file,
            trustAnchor: ExecutableIdentityExpectationTrustAnchor(
                expectedCurrentDocumentSHA256: file.sha256,
                expectedCurrentDocumentBytes:
                    UInt64(file.bytes.count),
                minimumEvidenceGeneration: evidenceGeneration,
                verificationUnixSeconds: 2_000_000_000,
                expectedArtifactRole: fields.artifactRole
            )
        )
    }

    static func replacing(
        _ value: ExecutableIdentityCodeDirectoryExpectation,
        blobSHA256: String? = nil,
        blobBytes: UInt64? = nil,
        hashType: UInt64? = nil,
        flags: String? = nil,
        signingIdentifier: Data? = nil,
        teamIdentifier: Data? = nil
    ) -> ExecutableIdentityCodeDirectoryExpectation {
        let signingIdentifierBytes =
            signingIdentifier.map { UInt64($0.count) }
                ?? value.signingIdentifierBytes
        let signingIdentifierBase64URL =
            signingIdentifier.map(base64URL)
                ?? value.signingIdentifierBase64URL
        let teamIdentifierBytes =
            teamIdentifier.map { UInt64($0.count) }
                ?? value.teamIdentifierBytes
        let teamIdentifierBase64URL =
            teamIdentifier.map(base64URL)
                ?? value.teamIdentifierBase64URL
        return ExecutableIdentityCodeDirectoryExpectation(
            slot: value.slot,
            blobSHA256: blobSHA256 ?? value.blobSHA256,
            blobBytes: blobBytes ?? value.blobBytes,
            hashType: hashType ?? value.hashType,
            flags: flags ?? value.flags,
            signingIdentifierBytes: signingIdentifierBytes,
            signingIdentifierBase64URL:
                signingIdentifierBase64URL,
            teamIdentifierBytes: teamIdentifierBytes,
            teamIdentifierBase64URL:
                teamIdentifierBase64URL
        )
    }

    static func replacing(
        _ value: SyntheticMachOIdentityComparison,
        fileSHA256: String? = nil,
        codeDirectories: [SyntheticCodeDirectoryComparison]? = nil
    ) -> SyntheticMachOIdentityComparison {
        SyntheticMachOIdentityComparison(
            retainedFileBytes: value.retainedFileBytes,
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
            codeSignatureRegion: value.codeSignatureRegion,
            codeSignatureRegionSHA256:
                value.codeSignatureRegionSHA256,
            codeDirectories: codeDirectories ?? value.codeDirectories,
            cmsBlob: value.cmsBlob,
            cmsBlobSHA256: value.cmsBlobSHA256,
            isAdHoc: value.isAdHoc
        )
    }

    static func expectedPreimage(
        role: ExecutableContentArtifactRole,
        comparison: SyntheticMachOIdentityComparison
    ) -> Data {
        let primary = comparison.codeDirectories[0]
        let lines = [
            "fast-mlx-proof-control-executable-content-evidence-id-v1",
            "artifact_role=\(role.rawValue)",
            "file_sha256=\(comparison.fileSHA256)",
            "file_bytes=\(comparison.retainedFileBytes.count)",
            "mach_header_sha256=" +
                sha256Hex(
                    Data(comparison.retainedFileBytes.prefix(32))
                ),
            "load_commands_sha256=" +
                comparison.loadCommandsSHA256,
            "macho_uuid=\(hex(comparison.machOUUID))",
            "primary_code_directory_blob_sha256=" +
                primary.blobSHA256,
            "code_signature_region_sha256=" +
                comparison.codeSignatureRegionSHA256,
        ]
        return Data((lines.joined(separator: "\n") + "\n").utf8)
    }

    static func assertInert(
        _ evidence: ExecutableContentIdentityEvidence,
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

    static func assertInert(
        _ comparison: ExecutableContentExpectationComparison,
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
        XCTAssertFalse(
            comparison.canLoadModel,
            file: file,
            line: line
        )
        XCTAssertFalse(
            comparison.canReserveOutput,
            file: file,
            line: line
        )
        XCTAssertFalse(comparison.canPublish, file: file, line: line)
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

    static func hex8(_ value: UInt32) -> String {
        String(format: "%08x", value)
    }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
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
}
