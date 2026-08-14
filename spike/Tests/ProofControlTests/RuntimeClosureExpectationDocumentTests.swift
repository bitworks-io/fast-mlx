import CryptoKit
import Foundation
import XCTest
@testable import ProofControl

final class RuntimeClosureExpectationDocumentTests: XCTestCase {
    func testFileImageExpectationReferenceCanonicalRolesStayInert()
        throws
    {
        let effects = ReferenceEffects()
        for role in [
            RuntimeClosureExpectationArtifactRole.git,
            .selfGuard,
        ] {
            let fixture = try Self.fileFixture(role: role, count: 1)
            let file = Self.admittedFile(fixture.documentBytes)
            let anchor = Self.matchingAnchor(file, role: role)
            let anchored = try RuntimeClosureExpectationVerifier.anchor(
                expectationFile: file,
                trustAnchor: anchor
            )
            let first = try FileImageRuntimeClosureExpectationVerifier
                .reference(
                    anchoredExpectation: anchored,
                    currentExpectationAnchor: anchor
                )
            let second = try FileImageRuntimeClosureExpectationVerifier
                .reference(
                    anchoredExpectation: anchored,
                    currentExpectationAnchor: anchor
                )

            XCTAssertEqual(first, second)
            XCTAssertEqual(first.anchoredExpectation, anchored)
            XCTAssertEqual(first.declaredFileImageMemberCount, 1)
            Self.assertFileImageReferenceInert(first)
            XCTAssertEqual(effects, .zero)
        }
    }

    func testFileImageExpectationReferenceCountsMixedAndAllFileRows()
        throws
    {
        let mixedBase = try Self.fixture(
            memberNames: [
                "/usr/lib/libMixedFile.dylib",
                "/usr/lib/libMixedCache.dylib",
            ]
        )
        let mixedMembers = [
            Self.fileMember(mixedBase.members[0]),
            mixedBase.members[1],
        ]
        let mixed = try Self.reference(
            role: .git,
            cacheSet: mixedBase.cacheSet,
            members: mixedMembers,
            edges: mixedBase.edges
        )
        XCTAssertEqual(mixed.declaredFileImageMemberCount, 1)
        XCTAssertEqual(mixed.anchoredExpectation.fields.members, mixedMembers)

        for count in [3, 256] {
            let fixture = try Self.fileFixture(count: count)
            let result = try Self.reference(fixture)
            XCTAssertEqual(result.declaredFileImageMemberCount, count)
            XCTAssertEqual(result.anchoredExpectation.fields.members.count, count)
            XCTAssertTrue(
                result.anchoredExpectation.fields.members.allSatisfy {
                    $0.storage == .file
                }
            )
            Self.assertFileImageReferenceInert(result)
        }
    }

    func testFileImageExpectationReferenceReanchorOrderIsExact()
        throws
    {
        let fixture = try Self.fileFixture(count: 1)
        let file = Self.admittedFile(fixture.documentBytes)
        let anchor = Self.matchingAnchor(file)
        let anchored = try RuntimeClosureExpectationVerifier.anchor(
            expectationFile: file,
            trustAnchor: anchor
        )
        let otherSHA = String(repeating: "f", count: 64)
        let cases: [
            (
                RuntimeClosureExpectationTrustAnchor,
                RuntimeClosureExpectationError
            )
        ] = [
            (
                Self.matchingAnchor(
                    file,
                    expectedSHA256: file.sha256.uppercased()
                ),
                .invalidTrustAnchor(.expectedCurrentDocumentSHA256)
            ),
            (
                Self.matchingAnchor(file, expectedSHA256: otherSHA),
                .documentDigestMismatch
            ),
            (
                Self.matchingAnchor(
                    file,
                    expectedBytes: UInt64(file.bytes.count) + 1
                ),
                .documentByteCountMismatch(
                    expected: UInt64(file.bytes.count) + 1,
                    actual: UInt64(file.bytes.count)
                )
            ),
            (
                Self.matchingAnchor(file, minimumGeneration: 10),
                .generationRollback(minimum: 10, actual: 9)
            ),
            (
                Self.matchingAnchor(
                    file,
                    verificationUnixSeconds: 1_899_999_999
                ),
                .documentNotYetValid
            ),
            (
                Self.matchingAnchor(
                    file,
                    verificationUnixSeconds: 2_100_000_001
                ),
                .documentExpired
            ),
            (
                Self.matchingAnchor(file, role: .selfGuard),
                .artifactRoleMismatch(expected: .selfGuard, actual: .git)
            ),
        ]
        for (current, expected) in cases {
            Self.assertReferenceFailure(
                anchored: anchored,
                currentAnchor: current,
                expected: .expectationReanchor(expected)
            )
        }

        for changedContext in [
            Self.matchingAnchor(
                file,
                verificationUnixSeconds: 2_000_000_001
            ),
            Self.matchingAnchor(file, minimumGeneration: 8),
        ] {
            Self.assertReferenceFailure(
                anchored: anchored,
                currentAnchor: changedContext,
                expected: .expectationEvidenceMismatch
            )
        }
    }

    func testFileImageExpectationReferenceRefusesZeroAndIsImmutable()
        throws
    {
        let sharedCacheOnly = try Self.fixture()
        let sharedFile = Self.admittedFile(sharedCacheOnly.documentBytes)
        let sharedAnchor = Self.matchingAnchor(sharedFile)
        let sharedAnchored = try RuntimeClosureExpectationVerifier.anchor(
            expectationFile: sharedFile,
            trustAnchor: sharedAnchor
        )
        Self.assertReferenceFailure(
            anchored: sharedAnchored,
            currentAnchor: sharedAnchor,
            expected: .declaredFileImageMemberCount
        )

        var fixture = try Self.fileFixture(count: 1)
        let padded = Data([0xff]) + fixture.documentBytes
        var callerBytes = padded.dropFirst()
        XCTAssertGreaterThan(callerBytes.startIndex, 0)
        let file = Self.admittedFile(callerBytes)
        let anchor = Self.matchingAnchor(file)
        let anchored = try RuntimeClosureExpectationVerifier.anchor(
            expectationFile: file,
            trustAnchor: anchor
        )
        let result = try FileImageRuntimeClosureExpectationVerifier
            .reference(
                anchoredExpectation: anchored,
                currentExpectationAnchor: anchor
            )
        let retainedBytes = result.anchoredExpectation.expectationFile.bytes

        callerBytes[callerBytes.startIndex] = 0x58
        fixture.members.removeAll()
        fixture.edges.removeAll()
        fixture.documentBytes.removeAll()
        XCTAssertEqual(
            result.anchoredExpectation.expectationFile.bytes,
            retainedBytes
        )
        XCTAssertEqual(result.anchoredExpectation.fields.members.count, 1)
        XCTAssertEqual(result.declaredFileImageMemberCount, 1)
        Self.assertFileImageReferenceInert(result)
    }

    func testCanonicalRolesAnchorExactGraphBytesAndStayInert()
        throws
    {
        let effects = Effects()
        for role in [
            RuntimeClosureExpectationArtifactRole.git,
            .selfGuard,
        ] {
            var fixture = try Self.fixture(role: role)
            let file = Self.admittedFile(fixture.documentBytes)
            let anchor = Self.matchingAnchor(file, role: role)
            let result = try RuntimeClosureExpectationVerifier.anchor(
                expectationFile: file,
                trustAnchor: anchor
            )

            XCTAssertEqual(result.expectationFile, file)
            XCTAssertEqual(result.documentSHA256, file.sha256)
            XCTAssertEqual(
                result.documentBytes,
                UInt64(fixture.documentBytes.count)
            )
            XCTAssertEqual(result.fields.artifactRole, role)
            XCTAssertEqual(
                result.fields.platformArchitecture,
                "arm64"
            )
            XCTAssertEqual(
                result.fields.platformHardwareModel,
                "Mac15,14"
            )
            XCTAssertEqual(result.fields.platformOSVersion, "26.5.2")
            XCTAssertEqual(result.fields.platformOSBuild, "25F84")
            XCTAssertEqual(
                result.fields.resolutionProfile,
                "absolute-static-graph-v1"
            )
            XCTAssertEqual(
                result.fields.environmentProfile,
                "no-dyld-environment-v1"
            )
            XCTAssertEqual(
                result.sharedCacheSetEvidence,
                fixture.cacheSet
            )
            XCTAssertEqual(result.fields.members, fixture.members)
            XCTAssertEqual(result.fields.edges, fixture.edges)
            XCTAssertEqual(result.trustAnchor, anchor)
            XCTAssertEqual(result.runtimeDecision, .noGo)

            let graphBytes = result.expectationFile.bytes[
                result.canonicalGraphSectionRange
            ]
            XCTAssertEqual(
                Data(graphBytes),
                fixture.collection.canonicalSectionBytes
            )
            XCTAssertEqual(
                Self.lineCount(fixture.documentBytes),
                20 + 5 * fixture.cacheSet.records.count +
                    7 * fixture.members.count +
                    6 * fixture.edges.count
            )
            Self.assertInert(result)
            Self.assertNoPrematureIdentity(result)

            fixture.members.removeAll()
            fixture.edges.removeAll()
            fixture.documentBytes.removeAll()
            XCTAssertEqual(
                Data(result.expectationFile.bytes[
                    result.canonicalGraphSectionRange
                ]),
                fixture.collection.canonicalSectionBytes
            )
            XCTAssertEqual(effects, .zero)
        }
    }

    func testAnchorOrderAndInclusiveBoundariesAreExact() throws {
        let effects = Effects()
        let fixture = try Self.fixture()
        let file = Self.admittedFile(fixture.documentBytes)
        let upperSHA = file.sha256.uppercased()
        let otherSHA = String(repeating: "f", count: 64)

        let cases: [
            (
                RuntimeClosureExpectationTrustAnchor,
                RuntimeClosureExpectationError
            )
        ] = [
            (
                Self.matchingAnchor(file, expectedSHA256: upperSHA),
                .invalidTrustAnchor(
                    .expectedCurrentDocumentSHA256
                )
            ),
            (
                Self.matchingAnchor(file, expectedBytes: 0),
                .invalidTrustAnchor(
                    .expectedCurrentDocumentBytes
                )
            ),
            (
                Self.matchingAnchor(file, minimumGeneration: 0),
                .invalidTrustAnchor(.minimumEvidenceGeneration)
            ),
            (
                Self.matchingAnchor(file, expectedSHA256: otherSHA),
                .documentDigestMismatch
            ),
            (
                Self.matchingAnchor(
                    file,
                    expectedBytes: UInt64(file.bytes.count) + 1
                ),
                .documentByteCountMismatch(
                    expected: UInt64(file.bytes.count) + 1,
                    actual: UInt64(file.bytes.count)
                )
            ),
            (
                Self.matchingAnchor(file, minimumGeneration: 10),
                .generationRollback(minimum: 10, actual: 9)
            ),
            (
                Self.matchingAnchor(
                    file,
                    verificationUnixSeconds: 1_899_999_999
                ),
                .documentNotYetValid
            ),
            (
                Self.matchingAnchor(
                    file,
                    verificationUnixSeconds: 2_100_000_001
                ),
                .documentExpired
            ),
            (
                Self.matchingAnchor(file, role: .selfGuard),
                .artifactRoleMismatch(
                    expected: .selfGuard,
                    actual: .git
                )
            ),
        ]
        for (anchor, expected) in cases {
            Self.assertFailure(
                file: file,
                anchor: anchor,
                expected: expected
            )
            XCTAssertEqual(effects, .zero)
        }

        for boundary in [1_900_000_000, 2_100_000_000] {
            XCTAssertNoThrow(
                try RuntimeClosureExpectationVerifier.anchor(
                    expectationFile: file,
                    trustAnchor: Self.matchingAnchor(
                        file,
                        verificationUnixSeconds: UInt64(boundary)
                    )
                )
            )
        }

        let oversizedBytes = Data(
            repeating: 0x61,
            count:
                RuntimeClosureExpectationVerifier.maximumDocumentBytes + 1
        )
        let oversized = Self.admittedFile(oversizedBytes)
        Self.assertFailure(
            file: oversized,
            anchor: Self.matchingAnchor(
                oversized,
                expectedSHA256: otherSHA
            ),
            expected: .documentSize
        )
        XCTAssertEqual(
            RuntimeClosureExpectationVerifier.maximumDocumentBytes,
            25_687_913
        )
        XCTAssertEqual(
            RuntimeClosureExpectationVerifier.maximumLineBytes,
            5_502
        )
        XCTAssertEqual(effects, .zero)
    }

    func testCanonicalDocumentGrammarRejectsEveryTextualDrift()
        throws
    {
        let fixture = try Self.fixture()
        let text = String(
            decoding: fixture.documentBytes,
            as: UTF8.self
        )
        var reordered = Self.lines(fixture.documentBytes)
        reordered.swapAt(2, 3)
        var unknown = Self.lines(fixture.documentBytes)
        unknown.insert("unknown=value", at: 2)
        var duplicated = Self.lines(fixture.documentBytes)
        duplicated.insert(duplicated[2], at: 3)
        let tooLongLine = Data(
            (
                String(
                    repeating: "a",
                    count:
                        RuntimeClosureExpectationVerifier
                        .maximumLineBytes + 1
                ) + "\n"
            ).utf8
        )
        let malformed: [Data] = [
            Data(fixture.documentBytes.dropLast()),
            fixture.documentBytes + Data("\n".utf8),
            Data(text.replacingOccurrences(of: "\n", with: "\r\n").utf8),
            Data([0xef, 0xbb, 0xbf]) + fixture.documentBytes,
            Self.replacingFirst(
                fixture.documentBytes,
                "subject=",
                with: "subject=\t"
            ),
            Self.replacingFirst(
                fixture.documentBytes,
                "artifact_role=git",
                with: "artifact_role=Git"
            ),
            Self.replacingFirst(
                fixture.documentBytes,
                "evidence_generation=9",
                with: "evidence_generation=09"
            ),
            Self.replacingFirst(
                fixture.documentBytes,
                "valid_from_unix_seconds=1900000000",
                with: "valid_from_unix_seconds=+1900000000"
            ),
            Self.replacingFirst(
                fixture.documentBytes,
                "runtime_authority=none",
                with: "runtime_authority=None"
            ),
            Self.replacingFirst(
                fixture.documentBytes,
                "\nmember_count=",
                with: "\n\nmember_count="
            ),
            Self.data(reordered),
            Self.data(unknown),
            Self.data(duplicated),
            tooLongLine,
            Data([0xff, 0x0a]),
            Self.replacingFirst(
                fixture.documentBytes,
                "platform_architecture=arm64",
                with: "platform_architecture=arm64\0"
            ),
        ]

        for bytes in malformed {
            let file = Self.admittedFile(bytes)
            Self.assertFailure(
                file: file,
                anchor: Self.matchingAnchor(file),
                expected: .nonCanonicalDocument
            )
        }
    }

    func testCacheBoundsCanonicalizationSetIdentityAndOrderingReject()
        throws
    {
        let fixture = try Self.fixture()
        let base = fixture.documentBytes
        let malformed: [(Data, RuntimeClosureExpectationError)] = [
            (
                Self.replacingFirst(
                    base,
                    "shared_cache_file_count=1",
                    with: "shared_cache_file_count=0"
                ),
                .nonCanonicalDocument
            ),
            (
                Self.replacingFirst(
                    base,
                    "shared_cache_file_count=1",
                    with: "shared_cache_file_count=65"
                ),
                .nonCanonicalDocument
            ),
            (
                Self.replacingFirst(
                    base,
                    "shared_cache_file_0000_suffix_bytes=0",
                    with: "shared_cache_file_0000_suffix_bytes=4097"
                ),
                .cacheRecord(index: 0, field: .suffixBytes)
            ),
            (
                Self.replacingFirst(
                    base,
                    "shared_cache_file_0000_suffix_base64url=",
                    with:
                        "shared_cache_file_0000_suffix_base64url=="
                ),
                .cacheRecord(index: 0, field: .suffixBase64URL)
            ),
            (
                Self.replacingFirst(
                    base,
                    "shared_cache_file_0000_sha256=" +
                        String(repeating: "3", count: 64),
                    with: "shared_cache_file_0000_sha256=" +
                        String(repeating: "A", count: 64)
                ),
                .cacheRecord(index: 0, field: .fileSHA256)
            ),
            (
                Self.replacingFirst(
                    base,
                    "shared_cache_file_0000_bytes=4096",
                    with: "shared_cache_file_0000_bytes=0"
                ),
                .cacheRecord(index: 0, field: .fileBytes)
            ),
            (
                Self.replacingFirst(
                    base,
                    "shared_cache_set_id=" +
                        fixture.cacheSet.sharedCacheSetID.sha256,
                    with: "shared_cache_set_id=" +
                        String(repeating: "a", count: 64)
                ),
                .cacheSet(.identityMismatch)
            ),
        ]
        for (bytes, expected) in malformed {
            let file = Self.admittedFile(bytes)
            Self.assertFailure(
                file: file,
                anchor: Self.matchingAnchor(file),
                expected: expected
            )
        }

        let split = try Self.fixture(cacheRecords: [
            Self.mainCacheRecord,
            Self.cacheRecord(suffix: ".1", seed: "5"),
        ])
        var duplicateMainLines = Self.lines(split.documentBytes)
        Self.replaceLine(
            &duplicateMainLines,
            prefix: "shared_cache_file_0001_suffix_bytes=",
            with: "shared_cache_file_0001_suffix_bytes=0"
        )
        Self.replaceLine(
            &duplicateMainLines,
            prefix: "shared_cache_file_0001_suffix_base64url=",
            with: "shared_cache_file_0001_suffix_base64url="
        )
        Self.assertFreshFailure(
            Self.data(duplicateMainLines),
            .cacheSet(.predecessor(.duplicateSuffix(1)))
        )

        var duplicateUUIDLines = Self.lines(split.documentBytes)
        Self.replaceLine(
            &duplicateUUIDLines,
            prefix: "shared_cache_file_0001_header_uuid=",
            with: "shared_cache_file_0001_header_uuid=" +
                Self.mainCacheRecord.headerUUID
        )
        Self.assertFreshFailure(
            Self.data(duplicateUUIDLines),
            .cacheSet(.predecessor(.duplicateHeaderUUID(1)))
        )

        let nonemptyMain = Self.cacheRecord(suffix: ".0", seed: "7")
        var missingMainLines = Self.lines(split.documentBytes)
        Self.replaceLine(
            &missingMainLines,
            prefix: "shared_cache_file_0000_suffix_bytes=",
            with:
                "shared_cache_file_0000_suffix_bytes=" +
                String(nonemptyMain.suffixBytes)
        )
        Self.replaceLine(
            &missingMainLines,
            prefix: "shared_cache_file_0000_suffix_base64url=",
            with: "shared_cache_file_0000_suffix_base64url=" +
                nonemptyMain.suffixBase64URL
        )
        Self.assertFreshFailure(
            Self.data(missingMainLines),
            .cacheSet(.predecessor(.mainRecordMissing))
        )

        var mainNotFirstLines = Self.lines(split.documentBytes)
        let secondSuffix = Self.cacheRecord(suffix: ".1", seed: "5")
        Self.replaceLine(
            &mainNotFirstLines,
            prefix: "shared_cache_file_0000_suffix_bytes=",
            with: "shared_cache_file_0000_suffix_bytes=" +
                String(secondSuffix.suffixBytes)
        )
        Self.replaceLine(
            &mainNotFirstLines,
            prefix: "shared_cache_file_0000_suffix_base64url=",
            with: "shared_cache_file_0000_suffix_base64url=" +
                secondSuffix.suffixBase64URL
        )
        Self.replaceLine(
            &mainNotFirstLines,
            prefix: "shared_cache_file_0001_suffix_bytes=",
            with: "shared_cache_file_0001_suffix_bytes=0"
        )
        Self.replaceLine(
            &mainNotFirstLines,
            prefix: "shared_cache_file_0001_suffix_base64url=",
            with: "shared_cache_file_0001_suffix_base64url="
        )
        Self.assertFreshFailure(
            Self.data(mainNotFirstLines),
            .cacheSet(.predecessor(.mainRecordNotFirst(1)))
        )

        let three = try Self.fixture(cacheRecords: [
            Self.mainCacheRecord,
            Self.cacheRecord(suffix: ".1", seed: "5"),
            Self.cacheRecord(suffix: ".2", seed: "6"),
        ])
        var unsortedLines = Self.lines(three.documentBytes)
        let firstSuffixLine = Self.line(
            prefix: "shared_cache_file_0001_suffix_base64url=",
            in: three.documentBytes
        )
        let secondSuffixLine = Self.line(
            prefix: "shared_cache_file_0002_suffix_base64url=",
            in: three.documentBytes
        )
        Self.replaceLine(
            &unsortedLines,
            prefix: "shared_cache_file_0001_suffix_base64url=",
            with: secondSuffixLine.replacingOccurrences(
                of: "shared_cache_file_0002_",
                with: "shared_cache_file_0001_"
            )
        )
        Self.replaceLine(
            &unsortedLines,
            prefix: "shared_cache_file_0002_suffix_base64url=",
            with: firstSuffixLine.replacingOccurrences(
                of: "shared_cache_file_0001_",
                with: "shared_cache_file_0002_"
            )
        )
        Self.assertFreshFailure(
            Self.data(unsortedLines),
            .cacheSet(.predecessor(.recordsNotSorted(2)))
        )

        let maximumSuffix = Data(repeating: 0x7a, count: 4_096)
        let maximumSuffixFixture = try Self.fixture(cacheRecords: [
            Self.mainCacheRecord,
            SyntheticSharedCacheFileRecord(
                suffixBytes: 4_096,
                suffixBase64URL: Self.base64URL(maximumSuffix),
                fileSHA256: String(repeating: "5", count: 64),
                fileBytes: 8_192,
                headerUUID: String(repeating: "5", count: 32)
            ),
        ])
        let maximumSuffixFile = Self.admittedFile(
            maximumSuffixFixture.documentBytes
        )
        XCTAssertNoThrow(
            try RuntimeClosureExpectationVerifier.anchor(
                expectationFile: maximumSuffixFile,
                trustAnchor: Self.matchingAnchor(maximumSuffixFile)
            )
        )
        XCTAssertEqual(
            Self.longestLine(maximumSuffixFixture.documentBytes),
            RuntimeClosureExpectationVerifier.maximumLineBytes
        )
    }

    func testCountBoundsAndRecordScalarFailuresStayTyped() throws {
        let fixture = try Self.fixture()
        for (target, replacement) in [
            ("member_count=1", "member_count=0"),
            ("member_count=1", "member_count=257"),
            ("edge_count=1", "edge_count=0"),
            ("edge_count=1", "edge_count=4097"),
        ] {
            Self.assertFreshFailure(
                Self.replacingFirst(
                    fixture.documentBytes,
                    target,
                    with: replacement
                ),
                .nonCanonicalDocument
            )
        }
        Self.assertFreshFailure(
            Self.replacingFirst(
                fixture.documentBytes,
                "member_0000_install_name_bytes=" +
                    String(fixture.members[0].installName.bytes),
                with: "member_0000_install_name_bytes=+2"
            ),
            .memberRecord(index: 0, field: .installNameBytes)
        )
        Self.assertFreshFailure(
            Self.replacingFirst(
                fixture.documentBytes,
                "edge_0000_load_command_ordinal=0",
                with: "edge_0000_load_command_ordinal=00"
            ),
            .edgeRecord(index: 0, field: .loadCommandOrdinal)
        )
        Self.assertFreshFailure(
            Self.replacingFirst(
                fixture.documentBytes,
                "edge_0000_kind=load",
                with: "edge_0000_kind=weak"
            ),
            .edgeRecord(index: 0, field: .kind)
        )
        Self.assertFreshFailure(
            Self.replacingFirst(
                fixture.documentBytes,
                "member_0000_install_name_base64url=" +
                    fixture.members[0].installName.base64URL,
                with: "member_0000_install_name_base64url=A"
            ),
            .memberRecord(index: 0, field: .installNameBase64URL)
        )
    }

    func testCanonicalFileMemberRowIsExpectationOnly() throws {
        let base = try Self.fixture()
        let name = Self.installName("/usr/lib/libFileOnly.dylib")
        let decoded = try SyntheticRuntimeClosureInstallNameVerifier
            .validate(name)
        let fileMember = RuntimeClosureExpectationMemberFields(
            contentEvidenceID: String(repeating: "a", count: 64),
            storage: .file,
            installName: name,
            decodedInstallName: decoded,
            machOUUID: String(repeating: "b", count: 32),
            primaryCodeDirectoryBlobSHA256:
                String(repeating: "c", count: 64),
            loadCommandsSHA256: String(repeating: "d", count: 64)
        )
        let edge = RuntimeClosureExpectationEdgeFields(
            parentContentEvidenceID: fileMember.contentEvidenceID,
            loadCommandOrdinal: 0,
            kind: .load,
            installName: name,
            decodedInstallName: decoded,
            resolvedContentEvidenceID: fileMember.contentEvidenceID
        )
        let bytes = Self.render(
            role: .git,
            cacheSet: base.cacheSet,
            members: [fileMember],
            edges: [edge]
        )
        let file = Self.admittedFile(bytes)
        let result = try RuntimeClosureExpectationVerifier.anchor(
            expectationFile: file,
            trustAnchor: Self.matchingAnchor(file)
        )
        XCTAssertEqual(result.fields.members, [fileMember])
        Self.assertInert(result)
        Self.assertNoPrematureIdentity(result)
    }

    func testMemberRulesRederivationOrderingUniquenessAndCollisions()
        throws
    {
        let single = try Self.fixture()
        let member = single.members[0]
        Self.assertFreshFailure(
            Self.replacingFirst(
                single.documentBytes,
                "member_0000_content_evidence_id=" +
                    member.contentEvidenceID,
                with: "member_0000_content_evidence_id=" +
                    String(repeating: "a", count: 64)
            ),
            .sharedCacheMemberIdentityMismatch(index: 0)
        )
        Self.assertFreshFailure(
            Self.replacingFirst(
                single.documentBytes,
                "root_executable_content_evidence_id=" + Self.rootID,
                with: "root_executable_content_evidence_id=" +
                    member.contentEvidenceID
            ),
            .prefixIdentityCollision(.rootAndMember)
        )
        Self.assertFreshFailure(
            Self.replacingFirst(
                single.documentBytes,
                "dynamic_loader_content_evidence_id=" + Self.loaderID,
                with: "dynamic_loader_content_evidence_id=" + Self.rootID
            ),
            .prefixIdentityCollision(.rootAndDynamicLoader)
        )
        Self.assertFreshFailure(
            Self.replacingFirst(
                single.documentBytes,
                "member_0000_storage=shared-cache",
                with: "member_0000_storage=file"
            ).replacing(
                Self.line(
                    prefix:
                        "member_0000_primary_code_directory_blob_sha256=",
                    in: single.documentBytes
                ),
                with:
                    "member_0000_primary_code_directory_blob_sha256=" +
                    String(repeating: "0", count: 64)
            ),
            .memberRecord(
                index: 0,
                field: .primaryCodeDirectoryBlobSHA256
            )
        )

        let pair = try Self.fixture(
            memberNames: [
                "/usr/lib/libFirst.dylib",
                "/usr/lib/libSecond.dylib",
            ]
        )
        let duplicateIDMembers = [pair.members[0], Self.member(
            pair.members[1],
            contentEvidenceID: pair.members[0].contentEvidenceID
        )]
        Self.assertFreshFailure(
            Self.render(
                role: .git,
                cacheSet: pair.cacheSet,
                members: duplicateIDMembers,
                edges: pair.edges
            ),
            .memberUniqueness(
                index: 1,
                kind: .contentEvidenceID
            )
        )
        Self.assertFreshFailure(
            Self.render(
                role: .git,
                cacheSet: pair.cacheSet,
                members: Array(pair.members.reversed()),
                edges: pair.edges
            ),
            .memberOrdering(index: 1)
        )
        let duplicateNameMembers = [pair.members[0], Self.member(
            pair.members[1],
            installName: pair.members[0].installName,
            decodedInstallName: pair.members[0].decodedInstallName
        )]
        Self.assertFreshFailure(
            Self.render(
                role: .git,
                cacheSet: pair.cacheSet,
                members: duplicateNameMembers,
                edges: pair.edges
            ),
            .memberUniqueness(index: 1, kind: .installName)
        )
    }

    func testEdgeOrderMembershipResolutionAndLabelsRejectExactly()
        throws
    {
        let pair = try Self.fixture(
            memberNames: [
                "/usr/lib/libEdgeA.dylib",
                "/usr/lib/libEdgeB.dylib",
            ]
        )
        let duplicate = [pair.edges[0], Self.edge(
            pair.edges[1],
            parentContentEvidenceID:
                pair.edges[0].parentContentEvidenceID,
            loadCommandOrdinal: pair.edges[0].loadCommandOrdinal
        )]
        Self.assertFreshFailure(
            Self.render(
                role: .git,
                cacheSet: pair.cacheSet,
                members: pair.members,
                edges: duplicate
            ),
            .edgeUniqueness(index: 1)
        )
        Self.assertFreshFailure(
            Self.render(
                role: .git,
                cacheSet: pair.cacheSet,
                members: pair.members,
                edges: Array(pair.edges.reversed())
            ),
            .edgeOrdering(index: 1)
        )

        let one = try Self.fixture()
        Self.assertFreshFailure(
            Self.render(
                role: .git,
                cacheSet: one.cacheSet,
                members: one.members,
                edges: [Self.edge(
                    one.edges[0],
                    parentContentEvidenceID:
                        String(repeating: "a", count: 64)
                )]
            ),
            .edgeParentMissing(index: 0)
        )
        Self.assertFreshFailure(
            Self.render(
                role: .git,
                cacheSet: one.cacheSet,
                members: one.members,
                edges: [Self.edge(
                    one.edges[0],
                    resolvedContentEvidenceID:
                        String(repeating: "b", count: 64)
                )]
            ),
            .edgeResolvedMissing(index: 0)
        )
        let wrongName = Self.installName(
            "/usr/lib/libWrong.dylib"
        )
        Self.assertFreshFailure(
            Self.render(
                role: .git,
                cacheSet: one.cacheSet,
                members: one.members,
                edges: [Self.edge(
                    one.edges[0],
                    installName: wrongName,
                    decodedInstallName:
                        try SyntheticRuntimeClosureInstallNameVerifier
                        .validate(wrongName)
                )]
            ),
            .edgeInstallNameMismatch(index: 0)
        )
        Self.assertFreshFailure(
            Self.render(
                role: .git,
                cacheSet: one.cacheSet,
                members: one.members,
                edges: [Self.edge(
                    one.edges[0],
                    parentContentEvidenceID: Self.loaderID
                )]
            ),
            .edgeParentMissing(index: 0)
        )
    }

    func testMaximumInstallNameAndCallerMutationRemainBounded()
        throws
    {
        let maximumName = "/" + String(repeating: "a", count: 4_095)
        var fixture = try Self.fixture(memberNames: [maximumName])
        var callerBytes = fixture.documentBytes
        let file = Self.admittedFile(callerBytes)
        let result = try RuntimeClosureExpectationVerifier.anchor(
            expectationFile: file,
            trustAnchor: Self.matchingAnchor(file)
        )
        XCTAssertEqual(result.fields.members[0].installName.bytes, 4_096)
        XCTAssertLessThanOrEqual(
            Self.longestLine(fixture.documentBytes),
            RuntimeClosureExpectationVerifier.maximumLineBytes
        )

        callerBytes[callerBytes.startIndex] = 0x58
        fixture.members.removeAll()
        fixture.edges.removeAll()
        XCTAssertEqual(result.expectationFile.bytes, file.bytes)
        XCTAssertEqual(result.fields.members.count, 1)
        XCTAssertEqual(result.fields.edges.count, 1)
        Self.assertInert(result)
    }
}

private extension RuntimeClosureExpectationDocumentTests {
    struct Fixture {
        let cacheSet: SyntheticSharedCacheSetIdentityEvidence
        var members: [RuntimeClosureExpectationMemberFields]
        var edges: [RuntimeClosureExpectationEdgeFields]
        let collection: SyntheticRuntimeClosureRecordCollectionComparison
        var documentBytes: Data
    }

    struct FileReferenceFixture {
        let role: RuntimeClosureExpectationArtifactRole
        let cacheSet: SyntheticSharedCacheSetIdentityEvidence
        var members: [RuntimeClosureExpectationMemberFields]
        var edges: [RuntimeClosureExpectationEdgeFields]
        var documentBytes: Data
    }

    struct Effects: Equatable {
        var spawns = 0
        var fileSystemMutations = 0
        var networkOperations = 0
        var packConsumptions = 0

        static let zero = Self()
    }

    struct ReferenceEffects: Equatable {
        var spawns = 0
        var fileSystemMutations = 0
        var networkOperations = 0
        var packConsumptions = 0
        var gitObjectImports = 0
        var builds = 0
        var modelLoads = 0
        var outputReservations = 0
        var publications = 0
        var liveCaptures = 0

        static let zero = Self()
    }

    static let rootID = String(repeating: "1", count: 64)
    static let loaderID = String(repeating: "2", count: 64)
    static let mainCacheRecord = SyntheticSharedCacheFileRecord(
        suffixBytes: 0,
        suffixBase64URL: "",
        fileSHA256: String(repeating: "3", count: 64),
        fileBytes: 4_096,
        headerUUID: String(repeating: "4", count: 32)
    )

    static func fileFixture(
        role: RuntimeClosureExpectationArtifactRole = .git,
        count: Int
    ) throws -> FileReferenceFixture {
        let cacheSet = try SyntheticSharedCacheSetIdentityVerifier
            .derive(records: [mainCacheRecord])
        var members: [RuntimeClosureExpectationMemberFields] = []
        var edges: [RuntimeClosureExpectationEdgeFields] = []
        members.reserveCapacity(count)
        edges.reserveCapacity(count)
        for index in 0..<count {
            let suffix = fourDigit(index)
            let name = installName("/usr/lib/libFile\(suffix).dylib")
            let decoded = try SyntheticRuntimeClosureInstallNameVerifier
                .validate(name)
            let contentEvidenceID =
                String(repeating: "a", count: 60) + suffix
            let member = RuntimeClosureExpectationMemberFields(
                contentEvidenceID: contentEvidenceID,
                storage: .file,
                installName: name,
                decodedInstallName: decoded,
                machOUUID: String(repeating: "b", count: 32),
                primaryCodeDirectoryBlobSHA256:
                    String(repeating: "c", count: 64),
                loadCommandsSHA256:
                    String(repeating: "d", count: 64)
            )
            members.append(member)
            edges.append(
                RuntimeClosureExpectationEdgeFields(
                    parentContentEvidenceID: contentEvidenceID,
                    loadCommandOrdinal: 0,
                    kind: .load,
                    installName: name,
                    decodedInstallName: decoded,
                    resolvedContentEvidenceID: contentEvidenceID
                )
            )
        }
        return FileReferenceFixture(
            role: role,
            cacheSet: cacheSet,
            members: members,
            edges: edges,
            documentBytes: render(
                role: role,
                cacheSet: cacheSet,
                members: members,
                edges: edges
            )
        )
    }

    static func fileMember(
        _ value: RuntimeClosureExpectationMemberFields
    ) -> RuntimeClosureExpectationMemberFields {
        RuntimeClosureExpectationMemberFields(
            contentEvidenceID: value.contentEvidenceID,
            storage: .file,
            installName: value.installName,
            decodedInstallName: value.decodedInstallName,
            machOUUID: value.machOUUID,
            primaryCodeDirectoryBlobSHA256:
                value.primaryCodeDirectoryBlobSHA256,
            loadCommandsSHA256: value.loadCommandsSHA256
        )
    }

    static func reference(
        _ fixture: FileReferenceFixture
    ) throws -> FileImageRuntimeClosureExpectationReference {
        try reference(
            role: fixture.role,
            cacheSet: fixture.cacheSet,
            members: fixture.members,
            edges: fixture.edges
        )
    }

    static func reference(
        role: RuntimeClosureExpectationArtifactRole,
        cacheSet: SyntheticSharedCacheSetIdentityEvidence,
        members: [RuntimeClosureExpectationMemberFields],
        edges: [RuntimeClosureExpectationEdgeFields]
    ) throws -> FileImageRuntimeClosureExpectationReference {
        let bytes = render(
            role: role,
            cacheSet: cacheSet,
            members: members,
            edges: edges
        )
        let file = admittedFile(bytes)
        let anchor = matchingAnchor(file, role: role)
        let anchored = try RuntimeClosureExpectationVerifier.anchor(
            expectationFile: file,
            trustAnchor: anchor
        )
        return try FileImageRuntimeClosureExpectationVerifier.reference(
            anchoredExpectation: anchored,
            currentExpectationAnchor: anchor
        )
    }

    static func fixture(
        role: RuntimeClosureExpectationArtifactRole = .git,
        cacheRecords: [SyntheticSharedCacheFileRecord] = [
            mainCacheRecord
        ],
        memberNames: [String] = [
            "/usr/lib/libFixture.dylib"
        ]
    ) throws -> Fixture {
        let cacheSet = try SyntheticSharedCacheSetIdentityVerifier
            .derive(records: cacheRecords)
        var sources: [
            SyntheticSharedCacheImageContentIdentityEvidence
        ] = []
        for (index, name) in memberNames.enumerated() {
            let installName = installName(name)
            sources.append(
                try SyntheticSharedCacheImageContentIdentityVerifier
                    .derive(
                        cacheSetEvidence: cacheSet,
                        facts: SyntheticSharedCacheImageContentFacts(
                            installNameBytes: installName.bytes,
                            installNameBase64URL:
                                installName.base64URL,
                            machOUUID: fixedHex(
                                seed: UInt8(0x70 + index),
                                count: 16
                            ),
                            primaryCodeDirectory: .present(
                                blobSHA256: fixedHex(
                                    seed: UInt8(0x80 + index),
                                    count: 32
                                )
                            ),
                            loadCommandsSHA256: fixedHex(
                                seed: UInt8(0x90 + index),
                                count: 32
                            )
                        )
                    )
            )
        }
        sources.sort {
            $0.contentEvidenceID.sha256.utf8
                .lexicographicallyPrecedes(
                    $1.contentEvidenceID.sha256.utf8
                )
        }

        var memberRecords: [
            SyntheticRuntimeClosureMemberRecordComparison
        ] = []
        for (index, source) in sources.enumerated() {
            memberRecords.append(
                try SyntheticRuntimeClosureRecordSchemaVerifier.member(
                    index: index,
                    source: .sharedCache(source),
                    installName: SyntheticRuntimeClosureInstallName(
                        bytes: source.facts.installNameBytes,
                        base64URL:
                            source.facts.installNameBase64URL
                    )
                )
            )
        }
        let members = memberRecords.map {
            RuntimeClosureExpectationMemberFields(
                contentEvidenceID: $0.contentEvidenceID,
                storage: $0.storage,
                installName: $0.installName,
                decodedInstallName: $0.decodedInstallName,
                machOUUID: $0.machOUUID,
                primaryCodeDirectoryBlobSHA256:
                    $0.primaryCodeDirectoryBlobSHA256,
                loadCommandsSHA256: $0.loadCommandsSHA256
            )
        }

        var edgeRecords: [
            SyntheticRuntimeClosureEdgeRecordComparison
        ] = []
        for (index, member) in memberRecords.enumerated() {
            edgeRecords.append(
                try SyntheticRuntimeClosureRecordSchemaVerifier.edge(
                    index: index,
                    parent: .member(member),
                    loadCommandOrdinal: UInt64(index),
                    kind: .load,
                    installName: member.installName,
                    resolved: member
                )
            )
        }
        let edges = edgeRecords.map {
            RuntimeClosureExpectationEdgeFields(
                parentContentEvidenceID:
                    $0.parentContentEvidenceID,
                loadCommandOrdinal: $0.loadCommandOrdinal,
                kind: $0.kind,
                installName: $0.installName,
                decodedInstallName: $0.decodedInstallName,
                resolvedContentEvidenceID:
                    $0.resolvedContentEvidenceID
            )
        }
        let collection =
            try SyntheticRuntimeClosureRecordCollectionVerifier.derive(
                members: memberRecords,
                edges: edgeRecords
            )
        let documentBytes = render(
            role: role,
            cacheSet: cacheSet,
            members: members,
            edges: edges
        )
        return Fixture(
            cacheSet: cacheSet,
            members: members,
            edges: edges,
            collection: collection,
            documentBytes: documentBytes
        )
    }

    static func render(
        role: RuntimeClosureExpectationArtifactRole,
        cacheSet: SyntheticSharedCacheSetIdentityEvidence,
        members: [RuntimeClosureExpectationMemberFields],
        edges: [RuntimeClosureExpectationEdgeFields]
    ) -> Data {
        var lines = [
            RuntimeClosureExpectationVerifier.documentDomain,
            "subject=" +
                "absorbed-mla-source-import-runtime-closure-identity",
            "evidence_generation=9",
            "valid_from_unix_seconds=1900000000",
            "valid_until_unix_seconds=2100000000",
            "artifact_role=\(role.rawValue)",
            "platform_architecture=arm64",
            "platform_hardware_model=Mac15,14",
            "platform_os_version=26.5.2",
            "platform_os_build=25F84",
            "resolution_profile=absolute-static-graph-v1",
            "environment_profile=no-dyld-environment-v1",
            "root_executable_content_evidence_id=\(rootID)",
            "dynamic_loader_content_evidence_id=\(loaderID)",
            "shared_cache_file_count=\(cacheSet.records.count)",
        ]
        for (index, record) in cacheSet.records.enumerated() {
            let prefix = "shared_cache_file_\(fourDigit(index))_"
            lines.append("\(prefix)suffix_bytes=\(record.suffixBytes)")
            lines.append(
                "\(prefix)suffix_base64url=" +
                    record.suffixBase64URL
            )
            lines.append("\(prefix)sha256=\(record.fileSHA256)")
            lines.append("\(prefix)bytes=\(record.fileBytes)")
            lines.append("\(prefix)header_uuid=\(record.headerUUID)")
        }
        lines.append(
            "shared_cache_set_id=" +
                cacheSet.sharedCacheSetID.sha256
        )
        lines.append("member_count=\(members.count)")
        for (index, member) in members.enumerated() {
            let prefix = "member_\(fourDigit(index))_"
            lines.append(
                "\(prefix)content_evidence_id=" +
                    member.contentEvidenceID
            )
            lines.append("\(prefix)storage=\(member.storage.rawValue)")
            lines.append(
                "\(prefix)install_name_bytes=" +
                    String(member.installName.bytes)
            )
            lines.append(
                "\(prefix)install_name_base64url=" +
                    member.installName.base64URL
            )
            lines.append("\(prefix)macho_uuid=\(member.machOUUID)")
            lines.append(
                "\(prefix)primary_code_directory_blob_sha256=" +
                    member.primaryCodeDirectoryBlobSHA256
            )
            lines.append(
                "\(prefix)load_commands_sha256=" +
                    member.loadCommandsSHA256
            )
        }
        lines.append("edge_count=\(edges.count)")
        for (index, edge) in edges.enumerated() {
            let prefix = "edge_\(fourDigit(index))_"
            lines.append(
                "\(prefix)parent_content_evidence_id=" +
                    edge.parentContentEvidenceID
            )
            lines.append(
                "\(prefix)load_command_ordinal=" +
                    String(edge.loadCommandOrdinal)
            )
            lines.append("\(prefix)kind=\(edge.kind.rawValue)")
            lines.append(
                "\(prefix)install_name_bytes=" +
                    String(edge.installName.bytes)
            )
            lines.append(
                "\(prefix)install_name_base64url=" +
                    edge.installName.base64URL
            )
            lines.append(
                "\(prefix)resolved_content_evidence_id=" +
                    edge.resolvedContentEvidenceID
            )
        }
        lines.append(
            "runtime_resolution_outcome=" +
                "unproved-static-comparison-only"
        )
        lines.append("runtime_authority=none")
        return data(lines)
    }

    static func cacheRecord(
        suffix: String,
        seed: String
    ) -> SyntheticSharedCacheFileRecord {
        let suffixData = Data(suffix.utf8)
        return SyntheticSharedCacheFileRecord(
            suffixBytes: UInt64(suffixData.count),
            suffixBase64URL: base64URL(suffixData),
            fileSHA256: String(repeating: seed, count: 64),
            fileBytes: 8_192,
            headerUUID: String(repeating: seed, count: 32)
        )
    }

    static func installName(_ value: String)
        -> SyntheticRuntimeClosureInstallName
    {
        let bytes = Data(value.utf8)
        return SyntheticRuntimeClosureInstallName(
            bytes: UInt64(bytes.count),
            base64URL: base64URL(bytes)
        )
    }

    static func member(
        _ value: RuntimeClosureExpectationMemberFields,
        contentEvidenceID: String? = nil,
        installName: SyntheticRuntimeClosureInstallName? = nil,
        decodedInstallName: Data? = nil
    ) -> RuntimeClosureExpectationMemberFields {
        RuntimeClosureExpectationMemberFields(
            contentEvidenceID:
                contentEvidenceID ?? value.contentEvidenceID,
            storage: value.storage,
            installName: installName ?? value.installName,
            decodedInstallName:
                decodedInstallName ?? value.decodedInstallName,
            machOUUID: value.machOUUID,
            primaryCodeDirectoryBlobSHA256:
                value.primaryCodeDirectoryBlobSHA256,
            loadCommandsSHA256: value.loadCommandsSHA256
        )
    }

    static func edge(
        _ value: RuntimeClosureExpectationEdgeFields,
        parentContentEvidenceID: String? = nil,
        loadCommandOrdinal: UInt64? = nil,
        installName: SyntheticRuntimeClosureInstallName? = nil,
        decodedInstallName: Data? = nil,
        resolvedContentEvidenceID: String? = nil
    ) -> RuntimeClosureExpectationEdgeFields {
        RuntimeClosureExpectationEdgeFields(
            parentContentEvidenceID:
                parentContentEvidenceID ??
                    value.parentContentEvidenceID,
            loadCommandOrdinal:
                loadCommandOrdinal ?? value.loadCommandOrdinal,
            kind: value.kind,
            installName: installName ?? value.installName,
            decodedInstallName:
                decodedInstallName ?? value.decodedInstallName,
            resolvedContentEvidenceID:
                resolvedContentEvidenceID ??
                    value.resolvedContentEvidenceID
        )
    }

    static func admittedFile(_ bytes: Data) -> AdmittedFile {
        AdmittedFile(
            bytes: bytes,
            sha256: SHA256.hash(data: bytes)
                .map { String(format: "%02x", $0) }
                .joined(),
            identity: AdmittedFileIdentity(
                device: 0,
                inode: 0,
                size: UInt64(bytes.count),
                linkCount: 1,
                mode: 0,
                modificationSeconds: 0,
                modificationNanoseconds: 0,
                changeSeconds: 0,
                changeNanoseconds: 0
            )
        )
    }

    static func matchingAnchor(
        _ file: AdmittedFile,
        expectedSHA256: String? = nil,
        expectedBytes: UInt64? = nil,
        minimumGeneration: UInt64 = 9,
        verificationUnixSeconds: UInt64 = 2_000_000_000,
        role: RuntimeClosureExpectationArtifactRole = .git
    ) -> RuntimeClosureExpectationTrustAnchor {
        RuntimeClosureExpectationTrustAnchor(
            expectedCurrentDocumentSHA256:
                expectedSHA256 ?? file.sha256,
            expectedCurrentDocumentBytes:
                expectedBytes ?? UInt64(file.bytes.count),
            minimumEvidenceGeneration: minimumGeneration,
            verificationUnixSeconds: verificationUnixSeconds,
            expectedArtifactRole: role
        )
    }

    static func assertFreshFailure(
        _ bytes: Data,
        _ expected: RuntimeClosureExpectationError,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let admitted = admittedFile(bytes)
        assertFailure(
            file: admitted,
            anchor: matchingAnchor(admitted),
            expected: expected,
            filePath: file,
            line: line
        )
    }

    static func assertFailure(
        file: AdmittedFile,
        anchor: RuntimeClosureExpectationTrustAnchor,
        expected: RuntimeClosureExpectationError,
        filePath: StaticString = #filePath,
        line: UInt = #line
    ) {
        let effects = Effects()
        XCTAssertThrowsError(
            try RuntimeClosureExpectationVerifier.anchor(
                expectationFile: file,
                trustAnchor: anchor
            ),
            file: filePath,
            line: line
        ) {
            XCTAssertEqual(
                $0 as? RuntimeClosureExpectationError,
                expected,
                file: filePath,
                line: line
            )
        }
        XCTAssertEqual(effects, .zero, file: filePath, line: line)
    }

    static func assertReferenceFailure(
        anchored: AnchoredRuntimeClosureExpectationDocument,
        currentAnchor: RuntimeClosureExpectationTrustAnchor,
        expected: FileImageRuntimeClosureExpectationFailure,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let effects = ReferenceEffects()
        XCTAssertThrowsError(
            try FileImageRuntimeClosureExpectationVerifier.reference(
                anchoredExpectation: anchored,
                currentExpectationAnchor: currentAnchor
            ),
            file: file,
            line: line
        ) {
            XCTAssertEqual(
                $0 as? FileImageRuntimeClosureExpectationFailure,
                expected,
                file: file,
                line: line
            )
        }
        XCTAssertEqual(effects, .zero, file: file, line: line)
    }

    static func assertFileImageReferenceInert(
        _ result: FileImageRuntimeClosureExpectationReference,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(result.runtimeDecision, .noGo, file: file, line: line)
        XCTAssertFalse(result.canExecute, file: file, line: line)
        XCTAssertFalse(result.canSpawn, file: file, line: line)
        XCTAssertFalse(result.canAccessNetwork, file: file, line: line)
        XCTAssertFalse(result.canConsumePack, file: file, line: line)
        XCTAssertFalse(result.canMutateFileSystem, file: file, line: line)
        XCTAssertFalse(result.canImportGitObjects, file: file, line: line)
        XCTAssertFalse(result.canBuild, file: file, line: line)
        XCTAssertFalse(result.canLoadModel, file: file, line: line)
        XCTAssertFalse(result.canReserveOutput, file: file, line: line)
        XCTAssertFalse(result.canPublish, file: file, line: line)

        let labels = Set(
            Mirror(reflecting: result).children.compactMap(\.label)
        )
        for forbidden in [
            "id",
            "expectationID",
            "graphID",
            "closureContentID",
            "finalComparisonID",
            "transcriptID",
            "path",
            "fileDescriptor",
            "argv",
            "process",
            "sandbox",
            "capability",
            "nonce",
        ] {
            XCTAssertFalse(
                labels.contains(forbidden),
                file: file,
                line: line
            )
        }
    }

    static func assertInert(
        _ result: AnchoredRuntimeClosureExpectationDocument,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertFalse(result.canExecute, file: file, line: line)
        XCTAssertFalse(result.canSpawn, file: file, line: line)
        XCTAssertFalse(result.canAccessNetwork, file: file, line: line)
        XCTAssertFalse(result.canConsumePack, file: file, line: line)
        XCTAssertFalse(result.canMutateFileSystem, file: file, line: line)
        XCTAssertFalse(result.canImportGitObjects, file: file, line: line)
        XCTAssertFalse(result.canBuild, file: file, line: line)
        XCTAssertFalse(result.canLoadModel, file: file, line: line)
        XCTAssertFalse(result.canReserveOutput, file: file, line: line)
        XCTAssertFalse(result.canPublish, file: file, line: line)
    }

    static func assertNoPrematureIdentity(
        _ result: AnchoredRuntimeClosureExpectationDocument,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let labels = Set(
            Mirror(reflecting: result).children.compactMap(\.label)
        )
        for forbidden in [
            "id",
            "closureContentID",
            "finalComparisonID",
            "transcriptID",
            "path",
            "fileDescriptor",
            "argv",
            "process",
            "sandbox",
            "nonce",
        ] {
            XCTAssertFalse(
                labels.contains(forbidden),
                file: file,
                line: line
            )
        }
    }

    static func data(_ lines: [String]) -> Data {
        Data((lines.joined(separator: "\n") + "\n").utf8)
    }

    static func lines(_ bytes: Data) -> [String] {
        String(decoding: bytes, as: UTF8.self)
            .split(separator: "\n", omittingEmptySubsequences: false)
            .dropLast()
            .map(String.init)
    }

    static func replacingFirst(
        _ data: Data,
        _ target: String,
        with replacement: String
    ) -> Data {
        Data(
            String(decoding: data, as: UTF8.self)
                .replacingOccurrences(
                    of: target,
                    with: replacement,
                    options: [],
                    range: String(decoding: data, as: UTF8.self)
                        .range(of: target)
                ).utf8
        )
    }

    static func line(prefix: String, in data: Data) -> String {
        lines(data).first { $0.hasPrefix(prefix) }!
    }

    static func replaceLine(
        _ lines: inout [String],
        prefix: String,
        with replacement: String
    ) {
        let index = lines.firstIndex { $0.hasPrefix(prefix) }!
        lines[index] = replacement
    }

    static func base64URL(_ bytes: Data) -> String {
        bytes.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    static func fixedHex(seed: UInt8, count: Int) -> String {
        Data((0..<count).map { seed &+ UInt8($0) })
            .map { String(format: "%02x", $0) }
            .joined()
    }

    static func fourDigit(_ value: Int) -> String {
        let text = String(value)
        return String(repeating: "0", count: 4 - text.count) + text
    }

    static func lineCount(_ data: Data) -> Int {
        data.reduce(into: 0) { count, byte in
            if byte == 0x0a { count += 1 }
        }
    }

    static func longestLine(_ data: Data) -> Int {
        var longest = 0
        var current = 0
        for byte in data {
            if byte == 0x0a {
                longest = max(longest, current)
                current = 0
            } else {
                current += 1
            }
        }
        return longest
    }
}

private extension Data {
    func replacing(_ target: String, with replacement: String) -> Data {
        let text = String(decoding: self, as: UTF8.self)
        guard let range = text.range(of: target) else {
            return self
        }
        return Data(text.replacingCharacters(in: range, with: replacement).utf8)
    }
}
