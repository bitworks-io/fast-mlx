import Darwin
import Foundation
import XCTest
@testable import ProofControl

final class ExecutableIdentityExpectationDocumentTests: XCTestCase {
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
            "fast-mlx-executable-expectation-\(UUID().uuidString)"
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

    func testCanonicalExecutableExpectationAnchorsOnlyExactInertDocument()
        throws
    {
        let effects = FakeOperationalEffects()
        let expectedBytes = Self.expectedDocumentBytes
        XCTAssertEqual(
            try ExecutableIdentityExpectationVerifier.documentBytes(
                fields: Self.fields()
            ),
            expectedBytes
        )
        XCTAssertEqual(
            String(decoding: expectedBytes, as: UTF8.self)
                .split(
                    separator: "\n",
                    omittingEmptySubsequences: false
                )
                .count,
            46
        )

        let expectationFile = try captureDocument(expectedBytes)
        XCTAssertEqual(
            expectationFile.sha256,
            Self.expectedDocumentSHA256
        )
        XCTAssertEqual(
            UInt64(expectationFile.bytes.count),
            Self.expectedDocumentByteCount
        )
        let trustAnchor = trustAnchorMatching(expectationFile)
        let document = try ExecutableIdentityExpectationVerifier.anchor(
            expectationFile: expectationFile,
            trustAnchor: trustAnchor
        )

        XCTAssertEqual(document.expectationFile, expectationFile)
        XCTAssertEqual(document.documentSHA256, expectationFile.sha256)
        XCTAssertEqual(
            document.documentBytes,
            UInt64(expectedBytes.count)
        )
        XCTAssertEqual(document.fields, Self.fields())
        XCTAssertEqual(document.trustAnchor, trustAnchor)
        XCTAssertFalse(document.canExecute)
        XCTAssertFalse(document.canSpawn)
        XCTAssertFalse(document.canAccessNetwork)
        XCTAssertFalse(document.canConsumePack)
        XCTAssertFalse(document.canMutateFileSystem)
        XCTAssertFalse(document.canImportGitObjects)
        XCTAssertFalse(document.canBuild)
        XCTAssertFalse(document.canLoadModel)
        XCTAssertFalse(document.canReserveOutput)
        XCTAssertFalse(document.canPublish)
        XCTAssertEqual(effects, .zero)
    }

    func testAnchoredExpectationRetainsCapturedBytesAfterBackingRewrite()
        throws
    {
        let url = caseRoot.appendingPathComponent(
            "\(UUID().uuidString).executable-expectation"
        )
        try Self.expectedDocumentBytes.write(to: url)
        let expectationFile = try AdmittedFile.capture(
            absolutePath: url.path,
            maximumBytes: 32 * 1024
        )
        let trustAnchor = trustAnchorMatching(expectationFile)
        let document = try ExecutableIdentityExpectationVerifier.anchor(
            expectationFile: expectationFile,
            trustAnchor: trustAnchor
        )

        try Data("rewritten\n".utf8).write(to: url)

        XCTAssertEqual(
            document.expectationFile.bytes,
            Self.expectedDocumentBytes
        )
        XCTAssertEqual(document.documentSHA256, expectationFile.sha256)
        XCTAssertEqual(document.trustAnchor, trustAnchor)
        XCTAssertFalse(document.canExecute)
        XCTAssertFalse(document.canMutateFileSystem)
    }

    func testAnchorRejectsEveryExternalTrustMismatchBeforeEffects()
        throws
    {
        let expectationFile = try captureDocument(
            Self.expectedDocumentBytes
        )
        let effects = FakeOperationalEffects()
        let cases: [
            (
                ExecutableIdentityExpectationTrustAnchor,
                ExecutableIdentityExpectationError
            )
        ] = [
            (
                trustAnchorMatching(
                    expectationFile,
                    expectedSHA256: Self.uppercaseSHA256
                ),
                .invalidTrustAnchor(
                    .expectedCurrentDocumentSHA256
                )
            ),
            (
                trustAnchorMatching(
                    expectationFile,
                    expectedBytes: 0
                ),
                .invalidTrustAnchor(
                    .expectedCurrentDocumentBytes
                )
            ),
            (
                trustAnchorMatching(
                    expectationFile,
                    minimumGeneration: 0
                ),
                .invalidTrustAnchor(
                    .minimumEvidenceGeneration
                )
            ),
            (
                trustAnchorMatching(
                    expectationFile,
                    expectedSHA256: Self.otherSHA256
                ),
                .documentDigestMismatch
            ),
            (
                trustAnchorMatching(
                    expectationFile,
                    expectedBytes:
                        UInt64(expectationFile.bytes.count) + 1
                ),
                .documentByteCountMismatch(
                    expected:
                        UInt64(expectationFile.bytes.count) + 1,
                    actual: UInt64(expectationFile.bytes.count)
                )
            ),
            (
                trustAnchorMatching(
                    expectationFile,
                    minimumGeneration: 10
                ),
                .evidenceGenerationRollback(
                    minimum: 10,
                    actual: 9
                )
            ),
            (
                trustAnchorMatching(
                    expectationFile,
                    verificationUnixSeconds: 1_899_999_999
                ),
                .documentNotYetValid
            ),
            (
                trustAnchorMatching(
                    expectationFile,
                    verificationUnixSeconds: 2_100_000_001
                ),
                .documentExpired
            ),
            (
                trustAnchorMatching(
                    expectationFile,
                    expectedArtifactRole: .selfGuard
                ),
                .artifactRoleMismatch(
                    expected: .selfGuard,
                    actual: .git
                )
            ),
        ]

        for (anchor, expectedError) in cases {
            XCTAssertThrowsError(
                try ExecutableIdentityExpectationVerifier.anchor(
                    expectationFile: expectationFile,
                    trustAnchor: anchor
                )
            ) { error in
                XCTAssertEqual(
                    error as? ExecutableIdentityExpectationError,
                    expectedError
                )
            }
            XCTAssertEqual(effects, .zero)
        }

        for boundary in [1_900_000_000, 2_100_000_000] {
            XCTAssertNoThrow(
                try ExecutableIdentityExpectationVerifier.anchor(
                    expectationFile: expectationFile,
                    trustAnchor: trustAnchorMatching(
                        expectationFile,
                        verificationUnixSeconds: UInt64(boundary)
                    )
                )
            )
        }
        XCTAssertEqual(effects, .zero)
    }

    func testRejectsMalformedOrderDuplicateUnknownEncodingAndRanges()
        throws
    {
        var reordered = Self.expectedDocumentString
            .split(separator: "\n", omittingEmptySubsequences: false)
        reordered.swapAt(2, 3)

        let zeroDigest = String(repeating: "0", count: 64)
        let malformedCases: [Data] = [
            Data(Self.expectedDocumentBytes.dropLast()),
            Data(
                (
                    Self.expectedDocumentString
                        .replacingOccurrences(of: "\n", with: "\r\n") +
                        "\r\n"
                ).utf8
            ),
            Data([0xef, 0xbb, 0xbf]) + Self.expectedDocumentBytes,
            Data([0xff, 0xfe, 0xfd, 0x0a]),
            Data((reordered.joined(separator: "\n") + "\n").utf8),
            replacing(
                "evidence_generation=9\n",
                with: "evidence_generation=9\n" +
                    "evidence_generation=9\n"
            ),
            replacing(
                "runtime_authority=none",
                with: "runtime_authority=none\nunknown=true"
            ),
            replacing(
                "file_sha256=\(Self.fileSHA256)",
                with: "file_sha256=\(Self.uppercaseSHA256)"
            ),
            replacing("artifact_role=git", with: "artifact_role=Git"),
            replacing("platform_architecture=arm64", with:
                "platform_architecture=arm64 "),
            replacing("evidence_generation=9", with:
                "evidence_generation=09"),
            replacing("evidence_generation=9", with:
                "evidence_generation=0"),
            replacing("valid_from_unix_seconds=1900000000", with:
                "valid_from_unix_seconds=2100000001"),
            replacing("file_bytes=123456", with: "file_bytes=0"),
            replacing("macho_uuid=\(Self.machOUUID)", with:
                "macho_uuid=\(String(repeating: "0", count: 32))"),
            replacing("code_directory_count=2", with:
                "code_directory_count=0"),
            replacing("code_directory_count=2", with:
                "code_directory_count=7"),
            replacing("code_directory_0000_slot=0", with:
                "code_directory_0001_slot=0"),
            replacing("code_directory_0001_slot=4096", with:
                "code_directory_0001_slot=0"),
            replacing("code_directory_0001_slot=4096", with:
                "code_directory_0001_slot=4101"),
            replacing(
                "code_directory_0000_blob_bytes=2048",
                with: "code_directory_0000_blob_bytes=0"
            ),
            replacing(
                "code_directory_0000_flags=00010000",
                with: "code_directory_0000_flags=0010000"
            ),
            replacing(
                "code_directory_0000_signing_identifier_bytes=15",
                with:
                    "code_directory_0000_signing_identifier_bytes=14"
            ),
            replacing(
                "code_directory_0000_signing_identifier_base64url=" +
                    Self.signingIdentifierBase64URL,
                with:
                    "code_directory_0000_signing_identifier_base64url=" +
                    Self.signingIdentifierBase64URL + "="
            ),
            replacing(
                "code_directory_0000_signing_identifier_base64url=" +
                    Self.signingIdentifierBase64URL,
                with:
                    "code_directory_0000_signing_identifier_base64url=A"
            ),
            replacing(
                "code_directory_0000_team_identifier_base64url=" +
                    Self.teamIdentifierBase64URL,
                with:
                    "code_directory_0000_team_identifier_base64url=" +
                    "VEVBTTEyMzQ1N/"
            ),
            replacing(
                "code_directory_0001_team_identifier_bytes=0\n" +
                    "code_directory_0001_team_identifier_base64url=\n",
                with:
                    "code_directory_0001_team_identifier_bytes=1\n" +
                    "code_directory_0001_team_identifier_base64url=\n"
            ),
            replacing(
                "cms_blob_sha256=\(Self.cmsSHA256)\n" +
                    "cms_blob_bytes=4096\n",
                with:
                    "cms_blob_sha256=\(zeroDigest)\n" +
                    "cms_blob_bytes=4096\n"
            ),
            replacing("cms_blob_bytes=4096", with: "cms_blob_bytes=0"),
            replacing(
                "signer_semantics=metadata-only-no-trust-v1",
                with: "signer_semantics=trusted-v1"
            ),
            replacing("runtime_authority=none", with:
                "runtime_authority=execute"),
        ]

        for (caseIndex, malformed) in malformedCases.enumerated() {
            let file = try captureDocument(malformed)
            let effects = FakeOperationalEffects()
            XCTAssertThrowsError(
                try ExecutableIdentityExpectationVerifier.anchor(
                    expectationFile: file,
                    trustAnchor: trustAnchorMatching(file)
                ),
                "malformed case \(caseIndex) unexpectedly anchored"
            ) { error in
                XCTAssertEqual(
                    error as? ExecutableIdentityExpectationError,
                    .nonCanonicalDocument
                )
            }
            XCTAssertEqual(effects, .zero)
        }
    }

    private func replacing(
        _ target: String,
        with replacement: String
    ) -> Data {
        Data(
            (
                Self.expectedDocumentString
                    .replacingOccurrences(
                        of: target,
                        with: replacement
                    ) + "\n"
            ).utf8
        )
    }

    private func captureDocument(_ bytes: Data) throws -> AdmittedFile {
        let url = caseRoot.appendingPathComponent(
            "\(UUID().uuidString).executable-expectation"
        )
        try bytes.write(to: url)
        return try AdmittedFile.capture(
            absolutePath: url.path,
            maximumBytes: 32 * 1024
        )
    }

    private func trustAnchorMatching(
        _ file: AdmittedFile,
        expectedSHA256: String? = nil,
        expectedBytes: UInt64? = nil,
        minimumGeneration: UInt64 = 9,
        verificationUnixSeconds: UInt64 = 2_000_000_000,
        expectedArtifactRole:
            ExecutableIdentityArtifactRole = .git
    ) -> ExecutableIdentityExpectationTrustAnchor {
        ExecutableIdentityExpectationTrustAnchor(
            expectedCurrentDocumentSHA256:
                expectedSHA256 ?? file.sha256,
            expectedCurrentDocumentBytes:
                expectedBytes ?? UInt64(file.bytes.count),
            minimumEvidenceGeneration: minimumGeneration,
            verificationUnixSeconds: verificationUnixSeconds,
            expectedArtifactRole: expectedArtifactRole
        )
    }
}

private extension ExecutableIdentityExpectationDocumentTests {
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

    static func fields(
        role: ExecutableIdentityArtifactRole = .git
    ) -> ExecutableIdentityExpectationFields {
        ExecutableIdentityExpectationFields(
            evidenceGeneration: 9,
            validFromUnixSeconds: 1_900_000_000,
            validUntilUnixSeconds: 2_100_000_000,
            artifactRole: role,
            fileSHA256: fileSHA256,
            fileBytes: 123_456,
            cpuSubtype: "00000002",
            headerFlags: "00200085",
            loadCommandCount: 28,
            loadCommandBytes: 4_096,
            loadCommandsSHA256: loadCommandsSHA256,
            machOUUID: machOUUID,
            codeSignatureRegionSHA256: codeSignatureRegionSHA256,
            codeSignatureRegionBytes: 8_192,
            codeDirectories: [
                ExecutableIdentityCodeDirectoryExpectation(
                    slot: 0,
                    blobSHA256: primaryCodeDirectorySHA256,
                    blobBytes: 2_048,
                    hashType: 2,
                    flags: "00010000",
                    signingIdentifierBytes: 15,
                    signingIdentifierBase64URL:
                        signingIdentifierBase64URL,
                    teamIdentifierBytes: 10,
                    teamIdentifierBase64URL:
                        teamIdentifierBase64URL
                ),
                ExecutableIdentityCodeDirectoryExpectation(
                    slot: 4_096,
                    blobSHA256: alternateCodeDirectorySHA256,
                    blobBytes: 1_536,
                    hashType: 2,
                    flags: "00000000",
                    signingIdentifierBytes: 0,
                    signingIdentifierBase64URL: "",
                    teamIdentifierBytes: 0,
                    teamIdentifierBase64URL: ""
                ),
            ],
            cmsBlobSHA256: cmsSHA256,
            cmsBlobBytes: 4_096
        )
    }

    static let fileSHA256 = String(repeating: "a", count: 64)
    static let loadCommandsSHA256 = String(repeating: "b", count: 64)
    static let machOUUID = "0123456789abcdef0123456789abcdef"
    static let codeSignatureRegionSHA256 =
        String(repeating: "c", count: 64)
    static let primaryCodeDirectorySHA256 =
        String(repeating: "d", count: 64)
    static let alternateCodeDirectorySHA256 =
        String(repeating: "e", count: 64)
    static let cmsSHA256 = String(repeating: "f", count: 64)
    static let signingIdentifierBase64URL =
        "Y29tLmV4YW1wbGUuZ2l0"
    static let teamIdentifierBase64URL = "VEVBTTEyMzQ1Ng"
    static let uppercaseSHA256 = String(repeating: "A", count: 64)
    static let otherSHA256 = String(repeating: "1", count: 64)

    static let expectedDocumentString = """
        fast-mlx-proof-control-executable-identity-expectation-v1
        subject=absorbed-mla-source-import-executable-identity
        evidence_generation=9
        valid_from_unix_seconds=1900000000
        valid_until_unix_seconds=2100000000
        artifact_role=git
        platform_architecture=arm64
        platform_os_build=25F84
        container_format=thin-macho64-little-endian-v1
        file_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        file_bytes=123456
        mach_header_magic=feedfacf
        cpu_type=0100000c
        cpu_subtype=00000002
        file_type=00000002
        header_flags=00200085
        load_command_count=28
        load_command_bytes=4096
        load_commands_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
        macho_uuid=0123456789abcdef0123456789abcdef
        code_signature_region_sha256=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        code_signature_region_bytes=8192
        code_directory_count=2
        code_directory_0000_slot=0
        code_directory_0000_blob_sha256=dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
        code_directory_0000_blob_bytes=2048
        code_directory_0000_hash_type=2
        code_directory_0000_flags=00010000
        code_directory_0000_signing_identifier_bytes=15
        code_directory_0000_signing_identifier_base64url=Y29tLmV4YW1wbGUuZ2l0
        code_directory_0000_team_identifier_bytes=10
        code_directory_0000_team_identifier_base64url=VEVBTTEyMzQ1Ng
        code_directory_0001_slot=4096
        code_directory_0001_blob_sha256=eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
        code_directory_0001_blob_bytes=1536
        code_directory_0001_hash_type=2
        code_directory_0001_flags=00000000
        code_directory_0001_signing_identifier_bytes=0
        code_directory_0001_signing_identifier_base64url=
        code_directory_0001_team_identifier_bytes=0
        code_directory_0001_team_identifier_base64url=
        cms_blob_sha256=ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
        cms_blob_bytes=4096
        signer_semantics=metadata-only-no-trust-v1
        runtime_authority=none
        """
    static let expectedDocumentBytes = Data(
        (expectedDocumentString + "\n").utf8
    )
    static let expectedDocumentSHA256 =
        "2ebf5086a29a04893b6c7f76c554940004f4140c0e03bf5695510c7da35d44be"
    static let expectedDocumentByteCount: UInt64 = 1_887
}
