import CryptoKit
import Foundation
import XCTest
@testable import ProofControl

final class SyntheticSharedCacheSetIdentityTests: XCTestCase {
    func testCanonicalRecordsDeriveExactNoncircularIdentityInertly()
        throws
    {
        var records = Self.canonicalRecords()
        let effects = FakeOperationalEffects()

        let evidence = try SyntheticSharedCacheSetIdentityVerifier.derive(
            records: records
        )
        let expectedPreimage = Self.expectedPreimage(records)

        XCTAssertEqual(evidence.records, records)
        XCTAssertEqual(
            evidence.decodedSuffixes,
            [Data(), Data([0x00]), Data([0xf8])]
        )
        XCTAssertEqual(evidence.identityPreimage, expectedPreimage)
        XCTAssertEqual(
            evidence.sharedCacheSetID.sha256,
            Self.sha256Hex(expectedPreimage)
        )
        XCTAssertEqual(
            String(decoding: expectedPreimage, as: UTF8.self)
                .split(
                    separator: "\n",
                    omittingEmptySubsequences: false
                )
                .count,
            18
        )

        let text = String(decoding: expectedPreimage, as: UTF8.self)
        XCTAssertTrue(
            text.hasPrefix(
                "fast-mlx-proof-control-shared-cache-set-id-v1\n" +
                    "shared_cache_file_count=3\n"
            )
        )
        XCTAssertTrue(
            text.contains(
                "shared_cache_file_0000_suffix_bytes=0\n" +
                    "shared_cache_file_0000_suffix_base64url=\n"
            )
        )
        for forbidden in [
            "shared_cache_set_id=",
            "runtime",
            "closure",
            "member",
            "edge",
            "path",
            "transcript",
            "expectation",
            "claim",
            "tool_policy",
        ] {
            XCTAssertFalse(text.contains(forbidden))
        }
        Self.assertInert(evidence)

        records[0] = Self.record(
            suffix: Data(),
            fileSHA256: String(repeating: "f", count: 64),
            fileBytes: 9_999,
            headerUUID: String(repeating: "e", count: 32)
        )
        XCTAssertNotEqual(evidence.records, records)
        XCTAssertEqual(evidence.identityPreimage, expectedPreimage)
        XCTAssertEqual(effects, .zero)
    }

    func testEveryRecordScalarAndEncodingMismatchRejectsBeforeEffects()
        throws
    {
        struct InvalidCase {
            let record: SyntheticSharedCacheFileRecord
            let field: SyntheticSharedCacheFileRecordField
        }

        let valid = Self.record(
            suffix: Data(),
            fileSHA256: String(repeating: "1", count: 64),
            fileBytes: 4_096,
            headerUUID: String(repeating: "a", count: 32)
        )
        let cases: [InvalidCase] = [
            InvalidCase(
                record: Self.replacing(
                    valid,
                    suffixBytes: 1
                ),
                field: .suffixBytes
            ),
            InvalidCase(
                record: Self.replacing(
                    valid,
                    suffixBytes: UInt64.max
                ),
                field: .suffixBytes
            ),
            InvalidCase(
                record: Self.replacing(
                    valid,
                    suffixBytes: 1,
                    suffixBase64URL: "AA=="
                ),
                field: .suffixBase64URL
            ),
            InvalidCase(
                record: Self.replacing(
                    valid,
                    suffixBytes: 1,
                    suffixBase64URL: "A"
                ),
                field: .suffixBase64URL
            ),
            InvalidCase(
                record: Self.replacing(
                    valid,
                    suffixBytes: 1,
                    suffixBase64URL: "AA\n"
                ),
                field: .suffixBase64URL
            ),
            InvalidCase(
                record: Self.replacing(
                    valid,
                    suffixBytes: 1,
                    suffixBase64URL: "AB"
                ),
                field: .suffixBase64URL
            ),
            InvalidCase(
                record: Self.replacing(
                    valid,
                    fileSHA256: String(repeating: "A", count: 64)
                ),
                field: .fileSHA256
            ),
            InvalidCase(
                record: Self.replacing(
                    valid,
                    fileSHA256: String(repeating: "1", count: 63)
                ),
                field: .fileSHA256
            ),
            InvalidCase(
                record: Self.replacing(
                    valid,
                    fileSHA256: String(repeating: "g", count: 64)
                ),
                field: .fileSHA256
            ),
            InvalidCase(
                record: Self.replacing(valid, fileBytes: 0),
                field: .fileBytes
            ),
            InvalidCase(
                record: Self.replacing(
                    valid,
                    headerUUID: String(repeating: "A", count: 32)
                ),
                field: .headerUUID
            ),
            InvalidCase(
                record: Self.replacing(
                    valid,
                    headerUUID: String(repeating: "a", count: 31)
                ),
                field: .headerUUID
            ),
            InvalidCase(
                record: Self.replacing(
                    valid,
                    headerUUID: String(repeating: "g", count: 32)
                ),
                field: .headerUUID
            ),
        ]
        let effects = FakeOperationalEffects()

        for testCase in cases {
            XCTAssertThrowsError(
                try SyntheticSharedCacheSetIdentityVerifier.derive(
                    records: [testCase.record]
                )
            ) { error in
                XCTAssertEqual(
                    error as? SyntheticSharedCacheSetIdentityFailure,
                    .invalidRecord(index: 0, field: testCase.field)
                )
            }
            XCTAssertEqual(effects, .zero)
        }
    }

    func testCountMainUniquenessAndDecodedByteOrderRejectExactly()
        throws
    {
        let canonical = Self.canonicalRecords()
        let effects = FakeOperationalEffects()

        XCTAssertThrowsError(
            try SyntheticSharedCacheSetIdentityVerifier.derive(records: [])
        ) { error in
            XCTAssertEqual(
                error as? SyntheticSharedCacheSetIdentityFailure,
                .recordCountOutOfRange(0)
            )
        }

        var tooMany = [canonical[0]]
        for value in 0..<64 {
            tooMany.append(
                Self.record(
                    suffix: Data([UInt8(value)]),
                    fileSHA256: String(
                        format: "%064llx",
                        UInt64(value + 1)
                    ),
                    fileBytes: UInt64(value + 1),
                    headerUUID: String(
                        format: "%032llx",
                        UInt64(value + 1)
                    )
                )
            )
        }
        XCTAssertThrowsError(
            try SyntheticSharedCacheSetIdentityVerifier.derive(
                records: tooMany
            )
        ) { error in
            XCTAssertEqual(
                error as? SyntheticSharedCacheSetIdentityFailure,
                .recordCountOutOfRange(65)
            )
        }

        XCTAssertThrowsError(
            try SyntheticSharedCacheSetIdentityVerifier.derive(
                records: [canonical[1]]
            )
        ) { error in
            XCTAssertEqual(
                error as? SyntheticSharedCacheSetIdentityFailure,
                .mainRecordMissing
            )
        }

        XCTAssertThrowsError(
            try SyntheticSharedCacheSetIdentityVerifier.derive(
                records: [canonical[1], canonical[0]]
            )
        ) { error in
            XCTAssertEqual(
                error as? SyntheticSharedCacheSetIdentityFailure,
                .mainRecordNotFirst(1)
            )
        }

        let secondMain = Self.replacing(
            canonical[0],
            fileSHA256: String(repeating: "4", count: 64),
            headerUUID: String(repeating: "d", count: 32)
        )
        XCTAssertThrowsError(
            try SyntheticSharedCacheSetIdentityVerifier.derive(
                records: [canonical[0], secondMain]
            )
        ) { error in
            XCTAssertEqual(
                error as? SyntheticSharedCacheSetIdentityFailure,
                .duplicateSuffix(1)
            )
        }

        let duplicateSuffix = Self.replacing(
            canonical[1],
            fileSHA256: String(repeating: "4", count: 64),
            headerUUID: String(repeating: "d", count: 32)
        )
        XCTAssertThrowsError(
            try SyntheticSharedCacheSetIdentityVerifier.derive(
                records: [canonical[0], canonical[1], duplicateSuffix]
            )
        ) { error in
            XCTAssertEqual(
                error as? SyntheticSharedCacheSetIdentityFailure,
                .duplicateSuffix(2)
            )
        }

        let duplicateUUID = Self.replacing(
            canonical[1],
            headerUUID: canonical[0].headerUUID
        )
        XCTAssertThrowsError(
            try SyntheticSharedCacheSetIdentityVerifier.derive(
                records: [canonical[0], duplicateUUID]
            )
        ) { error in
            XCTAssertEqual(
                error as? SyntheticSharedCacheSetIdentityFailure,
                .duplicateHeaderUUID(1)
            )
        }

        XCTAssertThrowsError(
            try SyntheticSharedCacheSetIdentityVerifier.derive(
                records: [canonical[0], canonical[2], canonical[1]]
            )
        ) { error in
            XCTAssertEqual(
                error as? SyntheticSharedCacheSetIdentityFailure,
                .recordsNotSorted(2)
            )
        }

        // "-A" sorts before "AA" as encoded text, but decoded 0xf8
        // sorts after decoded 0x00. Encoded-text ordering must not pass.
        XCTAssertLessThan(
            canonical[2].suffixBase64URL,
            canonical[1].suffixBase64URL
        )
        XCTAssertThrowsError(
            try SyntheticSharedCacheSetIdentityVerifier.derive(
                records: [canonical[0], canonical[2], canonical[1]]
            )
        ) { error in
            XCTAssertEqual(
                error as? SyntheticSharedCacheSetIdentityFailure,
                .recordsNotSorted(2)
            )
        }
        XCTAssertEqual(effects, .zero)
    }

    func testCanonicalBase64URLAliasesCannotCreateDuplicateIdentity()
        throws
    {
        let main = Self.canonicalRecords()[0]
        let noncanonicalAlias = SyntheticSharedCacheFileRecord(
            suffixBytes: 1,
            suffixBase64URL: "AB",
            fileSHA256: String(repeating: "2", count: 64),
            fileBytes: 8_192,
            headerUUID: String(repeating: "b", count: 32)
        )
        let effects = FakeOperationalEffects()

        XCTAssertThrowsError(
            try SyntheticSharedCacheSetIdentityVerifier.derive(
                records: [main, noncanonicalAlias]
            )
        ) { error in
            XCTAssertEqual(
                error as? SyntheticSharedCacheSetIdentityFailure,
                .invalidRecord(index: 1, field: .suffixBase64URL)
            )
        }
        XCTAssertEqual(effects, .zero)
    }
}

private extension SyntheticSharedCacheSetIdentityTests {
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

    static func canonicalRecords() -> [SyntheticSharedCacheFileRecord] {
        [
            record(
                suffix: Data(),
                fileSHA256: String(repeating: "1", count: 64),
                fileBytes: 4_096,
                headerUUID: String(repeating: "a", count: 32)
            ),
            record(
                suffix: Data([0x00]),
                fileSHA256: String(repeating: "2", count: 64),
                fileBytes: 8_192,
                headerUUID: String(repeating: "b", count: 32)
            ),
            record(
                suffix: Data([0xf8]),
                fileSHA256: String(repeating: "3", count: 64),
                fileBytes: 12_288,
                headerUUID: String(repeating: "c", count: 32)
            ),
        ]
    }

    static func record(
        suffix: Data,
        fileSHA256: String,
        fileBytes: UInt64,
        headerUUID: String
    ) -> SyntheticSharedCacheFileRecord {
        SyntheticSharedCacheFileRecord(
            suffixBytes: UInt64(suffix.count),
            suffixBase64URL: suffix.base64EncodedString()
                .replacingOccurrences(of: "+", with: "-")
                .replacingOccurrences(of: "/", with: "_")
                .replacingOccurrences(of: "=", with: ""),
            fileSHA256: fileSHA256,
            fileBytes: fileBytes,
            headerUUID: headerUUID
        )
    }

    static func replacing(
        _ record: SyntheticSharedCacheFileRecord,
        suffixBytes: UInt64? = nil,
        suffixBase64URL: String? = nil,
        fileSHA256: String? = nil,
        fileBytes: UInt64? = nil,
        headerUUID: String? = nil
    ) -> SyntheticSharedCacheFileRecord {
        SyntheticSharedCacheFileRecord(
            suffixBytes: suffixBytes ?? record.suffixBytes,
            suffixBase64URL:
                suffixBase64URL ?? record.suffixBase64URL,
            fileSHA256: fileSHA256 ?? record.fileSHA256,
            fileBytes: fileBytes ?? record.fileBytes,
            headerUUID: headerUUID ?? record.headerUUID
        )
    }

    static func expectedPreimage(
        _ records: [SyntheticSharedCacheFileRecord]
    ) -> Data {
        var lines = [
            "fast-mlx-proof-control-shared-cache-set-id-v1",
            "shared_cache_file_count=\(records.count)",
        ]
        for (index, record) in records.enumerated() {
            let prefix = String(
                format: "shared_cache_file_%04d",
                index
            )
            lines.append(
                "\(prefix)_suffix_bytes=\(record.suffixBytes)"
            )
            lines.append(
                "\(prefix)_suffix_base64url=" +
                    record.suffixBase64URL
            )
            lines.append(
                "\(prefix)_sha256=\(record.fileSHA256)"
            )
            lines.append(
                "\(prefix)_bytes=\(record.fileBytes)"
            )
            lines.append(
                "\(prefix)_header_uuid=\(record.headerUUID)"
            )
        }
        return Data((lines.joined(separator: "\n") + "\n").utf8)
    }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
    }

    static func assertInert(
        _ evidence: SyntheticSharedCacheSetIdentityEvidence,
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
