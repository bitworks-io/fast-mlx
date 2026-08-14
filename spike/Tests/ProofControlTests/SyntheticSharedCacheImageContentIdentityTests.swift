import CryptoKit
import Foundation
import XCTest
@testable import ProofControl

final class SyntheticSharedCacheImageContentIdentityTests:
    XCTestCase
{
    func testCanonicalImageFactsDeriveExactNoncircularIdentityInertly()
        throws
    {
        let cacheSet = try Self.cacheSetEvidence()
        var installName = Data("/usr/lib/libSystem.B.dylib".utf8)
        let facts = Self.facts(
            installName: installName,
            primaryCodeDirectory: .present(
                blobSHA256: String(repeating: "c", count: 64)
            )
        )
        let effects = FakeOperationalEffects()

        let evidence =
            try SyntheticSharedCacheImageContentIdentityVerifier.derive(
                cacheSetEvidence: cacheSet,
                facts: facts
            )
        let expectedPreimage = Self.expectedPreimage(
            cacheSetID: cacheSet.sharedCacheSetID.sha256,
            facts: facts
        )

        XCTAssertEqual(evidence.cacheSetEvidence, cacheSet)
        XCTAssertEqual(evidence.facts, facts)
        XCTAssertEqual(evidence.decodedInstallName, installName)
        XCTAssertEqual(
            evidence.primaryCodeDirectoryBlobSHA256,
            String(repeating: "c", count: 64)
        )
        XCTAssertEqual(evidence.identityPreimage, expectedPreimage)
        XCTAssertEqual(
            evidence.contentEvidenceID.sha256,
            Self.sha256Hex(expectedPreimage)
        )
        XCTAssertEqual(
            evidence.contentEvidenceID.sha256,
            "f9ce35b0ba5c0079f15704c7f3c7a121" +
                "9f471a2a351a1b78fc9d6423f5fff6e0"
        )
        XCTAssertEqual(
            String(decoding: expectedPreimage, as: UTF8.self)
                .split(
                    separator: "\n",
                    omittingEmptySubsequences: false
                )
                .count,
            8
        )

        let text = String(
            decoding: expectedPreimage,
            as: UTF8.self
        )
        XCTAssertTrue(
            text.hasPrefix(
                "fast-mlx-proof-control-" +
                    "shared-cache-image-content-evidence-id-v1\n" +
                    "shared_cache_set_id=" +
                    cacheSet.sharedCacheSetID.sha256 +
                    "\n"
            )
        )
        for forbidden in [
            "content_evidence_id=",
            "runtime",
            "closure",
            "member",
            "edge",
            "transcript",
            "expectation",
            "claim",
            "tool_policy",
            "filesystem",
        ] {
            XCTAssertFalse(text.contains(forbidden))
        }
        Self.assertInert(evidence)

        installName[1] = 0x7a
        XCTAssertNotEqual(evidence.decodedInstallName, installName)
        XCTAssertEqual(evidence.identityPreimage, expectedPreimage)
        XCTAssertEqual(effects, .zero)
    }

    func testAbsentCodeDirectoryAndCacheSetBindingAreExact()
        throws
    {
        let firstCacheSet = try Self.cacheSetEvidence()
        let secondCacheSet = try Self.cacheSetEvidence(
            fileSHA256: String(repeating: "9", count: 64)
        )
        let absentFacts = Self.facts(
            primaryCodeDirectory: .absent
        )
        let presentFacts = Self.facts(
            primaryCodeDirectory: .present(
                blobSHA256: String(repeating: "1", count: 64)
            )
        )
        let effects = FakeOperationalEffects()

        let first =
            try SyntheticSharedCacheImageContentIdentityVerifier.derive(
                cacheSetEvidence: firstCacheSet,
                facts: absentFacts
            )
        let second =
            try SyntheticSharedCacheImageContentIdentityVerifier.derive(
                cacheSetEvidence: secondCacheSet,
                facts: absentFacts
            )
        let present =
            try SyntheticSharedCacheImageContentIdentityVerifier.derive(
                cacheSetEvidence: firstCacheSet,
                facts: presentFacts
            )

        XCTAssertEqual(
            first.primaryCodeDirectoryBlobSHA256,
            String(repeating: "0", count: 64)
        )
        XCTAssertNotEqual(
            first.contentEvidenceID,
            second.contentEvidenceID
        )
        XCTAssertNotEqual(
            first.contentEvidenceID,
            present.contentEvidenceID
        )
        XCTAssertTrue(
            String(
                decoding: first.identityPreimage,
                as: UTF8.self
            ).contains(
                "primary_code_directory_blob_sha256=" +
                    String(repeating: "0", count: 64) +
                    "\n"
            )
        )
        Self.assertInert(first)
        Self.assertInert(second)
        Self.assertInert(present)
        XCTAssertEqual(effects, .zero)
    }

    func testInstallNameEncodingLengthAndSyntaxRejectExactly()
        throws
    {
        struct InvalidCase {
            let facts: SyntheticSharedCacheImageContentFacts
            let field: SyntheticSharedCacheImageContentField
        }

        let valid = Self.facts()
        let cases: [InvalidCase] = [
            InvalidCase(
                facts: Self.replacing(valid, installNameBytes: 1),
                field: .installNameBytes
            ),
            InvalidCase(
                facts: Self.replacing(
                    valid,
                    installNameBytes: UInt64.max
                ),
                field: .installNameBytes
            ),
            InvalidCase(
                facts: Self.replacing(
                    valid,
                    installNameBase64URL:
                        valid.installNameBase64URL + "=="
                ),
                field: .installNameBase64URL
            ),
            InvalidCase(
                facts: Self.replacing(
                    valid,
                    installNameBase64URL: "A"
                ),
                field: .installNameBase64URL
            ),
            InvalidCase(
                facts: Self.replacing(
                    valid,
                    installNameBase64URL:
                        valid.installNameBase64URL + "\n"
                ),
                field: .installNameBase64URL
            ),
            InvalidCase(
                facts: Self.replacing(
                    valid,
                    installNameBytes: 1,
                    installNameBase64URL: "AB"
                ),
                field: .installNameBase64URL
            ),
            InvalidCase(
                facts: Self.facts(installName: Data()),
                field: .installNameBytes
            ),
            InvalidCase(
                facts: Self.facts(installName: Data("/".utf8)),
                field: .installNameBytes
            ),
            InvalidCase(
                facts: Self.facts(
                    installName: Data("usr/lib".utf8)
                ),
                field: .installNameSyntax
            ),
            InvalidCase(
                facts: Self.facts(
                    installName: Data("//usr/lib".utf8)
                ),
                field: .installNameSyntax
            ),
            InvalidCase(
                facts: Self.facts(
                    installName: Data("/usr//lib".utf8)
                ),
                field: .installNameSyntax
            ),
            InvalidCase(
                facts: Self.facts(
                    installName: Data("/usr/lib/".utf8)
                ),
                field: .installNameSyntax
            ),
            InvalidCase(
                facts: Self.facts(
                    installName: Data("/usr/./lib".utf8)
                ),
                field: .installNameSyntax
            ),
            InvalidCase(
                facts: Self.facts(
                    installName: Data("/usr/../lib".utf8)
                ),
                field: .installNameSyntax
            ),
            InvalidCase(
                facts: Self.facts(
                    installName: Data("/usr/@rpath".utf8)
                ),
                field: .installNameSyntax
            ),
            InvalidCase(
                facts: Self.facts(
                    installName: Data("/usr\\lib".utf8)
                ),
                field: .installNameSyntax
            ),
            InvalidCase(
                facts: Self.facts(
                    installName: Data("/usr lib".utf8)
                ),
                field: .installNameSyntax
            ),
            InvalidCase(
                facts: Self.facts(
                    installName: Data([0x2f, 0x75, 0x7f])
                ),
                field: .installNameSyntax
            ),
            InvalidCase(
                facts: Self.facts(
                    installName: Data([0x2f, 0xc3, 0xa9])
                ),
                field: .installNameSyntax
            ),
            InvalidCase(
                facts: Self.facts(
                    installName: Data(
                        [0x2f] +
                            Array(repeating: 0x61, count: 4_096)
                    )
                ),
                field: .installNameBytes
            ),
            InvalidCase(
                facts: Self.replacing(
                    valid,
                    installNameBytes: UInt64.max,
                    installNameBase64URL:
                        String(repeating: "A", count: 100_000)
                ),
                field: .installNameBytes
            ),
        ]
        let cacheSet = try Self.cacheSetEvidence()
        let effects = FakeOperationalEffects()

        for testCase in cases {
            XCTAssertThrowsError(
                try SyntheticSharedCacheImageContentIdentityVerifier
                    .derive(
                        cacheSetEvidence: cacheSet,
                        facts: testCase.facts
                    )
            ) { error in
                XCTAssertEqual(
                    error as?
                        SyntheticSharedCacheImageContentIdentityFailure,
                    .invalidField(testCase.field)
                )
            }
            XCTAssertEqual(effects, .zero)
        }

        let maximumInstallName = Data(
            [0x2f] + Array(repeating: 0x61, count: 4_095)
        )
        let maximum =
            try SyntheticSharedCacheImageContentIdentityVerifier.derive(
                cacheSetEvidence: cacheSet,
                facts: Self.facts(
                    installName: maximumInstallName
                )
            )
        XCTAssertEqual(
            maximum.decodedInstallName.count,
            4_096
        )
        Self.assertInert(maximum)
        XCTAssertEqual(effects, .zero)
    }

    func testMetadataAndCodeDirectoryPresenceRejectExactly()
        throws
    {
        struct InvalidCase {
            let facts: SyntheticSharedCacheImageContentFacts
            let field: SyntheticSharedCacheImageContentField
        }

        let valid = Self.facts()
        let cases: [InvalidCase] = [
            InvalidCase(
                facts: Self.replacing(
                    valid,
                    machOUUID: String(repeating: "A", count: 32)
                ),
                field: .machOUUID
            ),
            InvalidCase(
                facts: Self.replacing(
                    valid,
                    machOUUID: String(repeating: "a", count: 31)
                ),
                field: .machOUUID
            ),
            InvalidCase(
                facts: Self.replacing(
                    valid,
                    machOUUID: String(repeating: "g", count: 32)
                ),
                field: .machOUUID
            ),
            InvalidCase(
                facts: Self.replacing(
                    valid,
                    primaryCodeDirectory: .present(
                        blobSHA256:
                            String(repeating: "A", count: 64)
                    )
                ),
                field: .primaryCodeDirectoryBlobSHA256
            ),
            InvalidCase(
                facts: Self.replacing(
                    valid,
                    primaryCodeDirectory: .present(
                        blobSHA256:
                            String(repeating: "a", count: 63)
                    )
                ),
                field: .primaryCodeDirectoryBlobSHA256
            ),
            InvalidCase(
                facts: Self.replacing(
                    valid,
                    primaryCodeDirectory: .present(
                        blobSHA256:
                            String(repeating: "g", count: 64)
                    )
                ),
                field: .primaryCodeDirectoryBlobSHA256
            ),
            InvalidCase(
                facts: Self.replacing(
                    valid,
                    primaryCodeDirectory: .present(
                        blobSHA256:
                            String(repeating: "0", count: 64)
                    )
                ),
                field: .primaryCodeDirectoryBlobSHA256
            ),
            InvalidCase(
                facts: Self.replacing(
                    valid,
                    loadCommandsSHA256:
                        String(repeating: "A", count: 64)
                ),
                field: .loadCommandsSHA256
            ),
            InvalidCase(
                facts: Self.replacing(
                    valid,
                    loadCommandsSHA256:
                        String(repeating: "a", count: 63)
                ),
                field: .loadCommandsSHA256
            ),
            InvalidCase(
                facts: Self.replacing(
                    valid,
                    loadCommandsSHA256:
                        String(repeating: "g", count: 64)
                ),
                field: .loadCommandsSHA256
            ),
        ]
        let cacheSet = try Self.cacheSetEvidence()
        let effects = FakeOperationalEffects()

        for testCase in cases {
            XCTAssertThrowsError(
                try SyntheticSharedCacheImageContentIdentityVerifier
                    .derive(
                        cacheSetEvidence: cacheSet,
                        facts: testCase.facts
                    )
            ) { error in
                XCTAssertEqual(
                    error as?
                        SyntheticSharedCacheImageContentIdentityFailure,
                    .invalidField(testCase.field)
                )
            }
            XCTAssertEqual(effects, .zero)
        }
    }
}

private extension SyntheticSharedCacheImageContentIdentityTests {
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

    static func cacheSetEvidence(
        fileSHA256: String = String(repeating: "1", count: 64)
    ) throws -> SyntheticSharedCacheSetIdentityEvidence {
        try SyntheticSharedCacheSetIdentityVerifier.derive(records: [
            SyntheticSharedCacheFileRecord(
                suffixBytes: 0,
                suffixBase64URL: "",
                fileSHA256: fileSHA256,
                fileBytes: 4_096,
                headerUUID: String(repeating: "a", count: 32)
            ),
        ])
    }

    static func facts(
        installName: Data = Data(
            "/usr/lib/libSystem.B.dylib".utf8
        ),
        machOUUID: String = String(repeating: "b", count: 32),
        primaryCodeDirectory:
            SyntheticSharedCacheImagePrimaryCodeDirectory =
                .present(
                    blobSHA256:
                        String(repeating: "c", count: 64)
                ),
        loadCommandsSHA256: String =
            String(repeating: "d", count: 64)
    ) -> SyntheticSharedCacheImageContentFacts {
        SyntheticSharedCacheImageContentFacts(
            installNameBytes: UInt64(installName.count),
            installNameBase64URL: base64URL(installName),
            machOUUID: machOUUID,
            primaryCodeDirectory: primaryCodeDirectory,
            loadCommandsSHA256: loadCommandsSHA256
        )
    }

    static func replacing(
        _ facts: SyntheticSharedCacheImageContentFacts,
        installNameBytes: UInt64? = nil,
        installNameBase64URL: String? = nil,
        machOUUID: String? = nil,
        primaryCodeDirectory:
            SyntheticSharedCacheImagePrimaryCodeDirectory? = nil,
        loadCommandsSHA256: String? = nil
    ) -> SyntheticSharedCacheImageContentFacts {
        SyntheticSharedCacheImageContentFacts(
            installNameBytes:
                installNameBytes ?? facts.installNameBytes,
            installNameBase64URL:
                installNameBase64URL ??
                    facts.installNameBase64URL,
            machOUUID: machOUUID ?? facts.machOUUID,
            primaryCodeDirectory:
                primaryCodeDirectory ??
                    facts.primaryCodeDirectory,
            loadCommandsSHA256:
                loadCommandsSHA256 ??
                    facts.loadCommandsSHA256
        )
    }

    static func expectedPreimage(
        cacheSetID: String,
        facts: SyntheticSharedCacheImageContentFacts
    ) -> Data {
        let codeDirectorySHA256: String
        switch facts.primaryCodeDirectory {
        case .absent:
            codeDirectorySHA256 = String(repeating: "0", count: 64)
        case let .present(blobSHA256):
            codeDirectorySHA256 = blobSHA256
        }

        return Data(
            (
                [
                    "fast-mlx-proof-control-" +
                        "shared-cache-image-content-evidence-id-v1",
                    "shared_cache_set_id=\(cacheSetID)",
                    "install_name_bytes=\(facts.installNameBytes)",
                    "install_name_base64url=" +
                        facts.installNameBase64URL,
                    "macho_uuid=\(facts.machOUUID)",
                    "primary_code_directory_blob_sha256=" +
                        codeDirectorySHA256,
                    "load_commands_sha256=" +
                        facts.loadCommandsSHA256,
                ].joined(separator: "\n") + "\n"
            ).utf8
        )
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

    static func assertInert(
        _ evidence:
            SyntheticSharedCacheImageContentIdentityEvidence,
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
}
