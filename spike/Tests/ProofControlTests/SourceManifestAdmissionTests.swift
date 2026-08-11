import CryptoKit
import Darwin
import Foundation
import XCTest
@testable import ProofControl

final class SourceManifestAdmissionTests: XCTestCase {
    private static let baselineCommit =
        "4400b8932df374945ebef2cc504782016297c0df"
    private static let baselineTree =
        "7c81bdf20225425e61efec24ce02835c9893fffa"
    private static let candidateCommit =
        "f8d86192e2c558605a8745c446598063aedaac36"
    private static let candidateTree =
        "f9ca3359d542ad621650fd968193e97051f56afb"
    private static let blobA = String(repeating: "1", count: 40)
    private static let blobB = String(repeating: "2", count: 40)
    private static let blobC = String(repeating: "4", count: 40)
    private static let shaA = String(repeating: "a", count: 64)
    private static let shaB = String(repeating: "b", count: 64)
    private static let shaC = String(repeating: "c", count: 64)
    private static let differentSHA1 = String(repeating: "3", count: 40)
    private static let differentSHA256 = String(repeating: "3", count: 64)

    private static let pinnedBaselineManifestBytes = Data(
        """
        fast-mlx-proof-control-source-manifest-v1
        subject=absorbed-mla-loaded-result-pair
        role=baseline
        route=decompressed-deepseek-v3
        slot=baseline
        git_commit_sha1=4400b8932df374945ebef2cc504782016297c0df
        git_tree_sha1=7c81bdf20225425e61efec24ce02835c9893fffa
        entry_count=2
        entry=100644\t1111111111111111111111111111111111111111\t0\taaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\tREADME.md
        entry=100755\t2222222222222222222222222222222222222222\t42\tbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\tSources/App/main.swift

        """.utf8
    )

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
        .appendingPathComponent("fast-mlx-source-manifest-\(UUID().uuidString)")
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

    func testCanonicalBuilderPinsBytesModesAndTypedManifestID() throws {
        let entries = [
            SourceManifestEntry(
                mode: .executable,
                gitBlobSHA1: Self.blobB,
                byteCount: 42,
                sha256: Self.shaB,
                path: "Sources/App/main.swift"
            ),
            SourceManifestEntry(
                mode: .regular,
                gitBlobSHA1: Self.blobA,
                byteCount: 0,
                sha256: Self.shaA,
                path: "README.md"
            ),
        ]

        XCTAssertEqual(
            String(describing: SourceManifestFileMode.regular.rawValue),
            "100644"
        )
        XCTAssertEqual(
            String(describing: SourceManifestFileMode.executable.rawValue),
            "100755"
        )
        XCTAssertEqual(
            try SourceManifestAdmission.manifestBytes(
                role: .baseline,
                gitCommitSHA1: Self.baselineCommit,
                gitTreeSHA1: Self.baselineTree,
                entries: entries
            ),
            Self.pinnedBaselineManifestBytes
        )

        let authorized = try authorizeManifest(
            bytes: Self.pinnedBaselineManifestBytes,
            name: "baseline-source.manifest"
        )
        let admitted = try SourceManifestAdmission.admit(
            authorizedFile: authorized,
            expectedRole: .baseline
        )

        XCTAssertEqual(
            admitted.manifestID.rawValue,
            expectedManifestID(admitted)
        )
        XCTAssertNotEqual(admitted.manifestID.rawValue, authorized.authorizationID.rawValue)
        XCTAssertNotEqual(admitted.manifestID.rawValue, authorized.file.sha256)
        XCTAssertEqual(admitted.authorizedFile, authorized)
        XCTAssertEqual(admitted.role, .baseline)
        XCTAssertEqual(admitted.route, .decompressedDeepSeekV3)
        XCTAssertEqual(admitted.slot, .baseline)
        XCTAssertEqual(admitted.gitCommitSHA1, Self.baselineCommit)
        XCTAssertEqual(admitted.gitTreeSHA1, Self.baselineTree)
        XCTAssertEqual(admitted.entries, entries.sorted { $0.path < $1.path })
    }

    func testAdmitRejectsUnexpectedPurposeAndExpectedRole() throws {
        let workerAuthorized = try authorizeManifest(
            bytes: Self.pinnedBaselineManifestBytes,
            name: "worker-authorized-source.manifest",
            purpose: .workerBytes
        )
        XCTAssertThrowsError(
            try SourceManifestAdmission.admit(
                authorizedFile: workerAuthorized,
                expectedRole: .baseline
            )
        ) { error in
            XCTAssertEqual(
                error as? SourceManifestAdmissionError,
                .unexpectedPurpose(
                    expected: .sourceManifest,
                    actual: .workerBytes
                )
            )
        }

        let candidateBytes = try SourceManifestAdmission.manifestBytes(
            role: .candidate,
            gitCommitSHA1: Self.candidateCommit,
            gitTreeSHA1: Self.candidateTree,
            entries: baselineEntries()
        )
        let candidateAuthorized = try authorizeManifest(
            bytes: candidateBytes,
            name: "candidate-source.manifest"
        )
        XCTAssertThrowsError(
            try SourceManifestAdmission.admit(
                authorizedFile: candidateAuthorized,
                expectedRole: .baseline
            )
        ) { error in
            XCTAssertEqual(
                error as? SourceManifestAdmissionError,
                .unexpectedRole(expected: .baseline, actual: .candidate)
            )
        }
    }

    func testRejectsNonCanonicalManifestStructureAndScalarEncoding()
        throws
    {
        var reorderedLines = String(
            decoding: Self.pinnedBaselineManifestBytes,
            as: UTF8.self
        )
        .split(separator: "\n", omittingEmptySubsequences: false)
        reorderedLines.swapAt(8, 9)

        let duplicatePath = Data(
            """
            fast-mlx-proof-control-source-manifest-v1
            subject=absorbed-mla-loaded-result-pair
            role=baseline
            route=decompressed-deepseek-v3
            slot=baseline
            git_commit_sha1=\(Self.baselineCommit)
            git_tree_sha1=\(Self.baselineTree)
            entry_count=2
            entry=100644\t\(Self.blobA)\t0\t\(Self.shaA)\tREADME.md
            entry=100755\t\(Self.blobB)\t42\t\(Self.shaB)\tREADME.md

            """.utf8
        )
        let caseFoldedDuplicatePath = Data(
            """
            fast-mlx-proof-control-source-manifest-v1
            subject=absorbed-mla-loaded-result-pair
            role=baseline
            route=decompressed-deepseek-v3
            slot=baseline
            git_commit_sha1=\(Self.baselineCommit)
            git_tree_sha1=\(Self.baselineTree)
            entry_count=2
            entry=100644\t\(Self.blobA)\t0\t\(Self.shaA)\tREADME.md
            entry=100755\t\(Self.blobB)\t42\t\(Self.shaB)\treadme.md

            """.utf8
        )
        let fileDirectoryPrefixConflict = Data(
            """
            fast-mlx-proof-control-source-manifest-v1
            subject=absorbed-mla-loaded-result-pair
            role=baseline
            route=decompressed-deepseek-v3
            slot=baseline
            git_commit_sha1=\(Self.baselineCommit)
            git_tree_sha1=\(Self.baselineTree)
            entry_count=3
            entry=100644\t\(Self.blobA)\t0\t\(Self.shaA)\tSources
            entry=100644\t\(Self.blobC)\t7\t\(Self.shaC)\tSources-archive
            entry=100755\t\(Self.blobB)\t42\t\(Self.shaB)\tSources/App/main.swift

            """.utf8
        )
        let caseFoldedFileDirectoryPrefixConflict = Data(
            """
            fast-mlx-proof-control-source-manifest-v1
            subject=absorbed-mla-loaded-result-pair
            role=baseline
            route=decompressed-deepseek-v3
            slot=baseline
            git_commit_sha1=\(Self.baselineCommit)
            git_tree_sha1=\(Self.baselineTree)
            entry_count=3
            entry=100644\t\(Self.blobA)\t0\t\(Self.shaA)\tSources
            entry=100644\t\(Self.blobC)\t7\t\(Self.shaC)\tsources-archive
            entry=100755\t\(Self.blobB)\t42\t\(Self.shaB)\tsources/App/main.swift

            """.utf8
        )

        let cases: [(String, Data)] = [
            (
                "domain",
                replacing(
                    "fast-mlx-proof-control-source-manifest-v1",
                    with: "other-domain-v1"
                )
            ),
            (
                "subject",
                replacing(
                    "subject=absorbed-mla-loaded-result-pair",
                    with: "subject=other"
                )
            ),
            (
                "utf8",
                Data([0xff])
            ),
            (
                "crlf",
                Data(
                    String(decoding: Self.pinnedBaselineManifestBytes, as: UTF8.self)
                        .replacingOccurrences(of: "\n", with: "\r\n")
                        .utf8
                )
            ),
            (
                "final-lf",
                Data(Self.pinnedBaselineManifestBytes.dropLast())
            ),
            (
                "extra-line",
                Self.pinnedBaselineManifestBytes + Data("extra=true\n".utf8)
            ),
            (
                "entry-count-zero",
                replacing("entry_count=2", with: "entry_count=0")
            ),
            (
                "entry-count-leading-zero",
                replacing("entry_count=2", with: "entry_count=02")
            ),
            (
                "entry-count-mismatch",
                replacing("entry_count=2", with: "entry_count=3")
            ),
            (
                "entry-count-too-large",
                replacing("entry_count=2", with: "entry_count=1000001")
            ),
            (
                "mode",
                replacing("entry=100644", with: "entry=100664")
            ),
            (
                "blob-hash-uppercase",
                replacing(
                    Self.blobA,
                    with: "A" + String(Self.blobA.dropFirst())
                )
            ),
            (
                "blob-hash-length",
                replacing(Self.blobA, with: String(Self.blobA.dropLast()))
            ),
            (
                "sha256-uppercase",
                replacing(Self.shaA, with: Self.shaA.uppercased())
            ),
            (
                "sha256-length",
                replacing(Self.shaA, with: String(Self.shaA.dropLast()))
            ),
            (
                "decimal-leading-zero",
                replacing("\t42\t", with: "\t042\t")
            ),
            (
                "decimal-overflow",
                replacing("\t42\t", with: "\t18446744073709551616\t")
            ),
            (
                "path-absolute",
                replacing("README.md", with: "/README.md")
            ),
            (
                "path-dot-component",
                replacing("README.md", with: "./README.md")
            ),
            (
                "path-dotdot-component",
                replacing("README.md", with: "Source/../README.md")
            ),
            (
                "path-git-component",
                replacing("README.md", with: ".git/config")
            ),
            (
                "path-case-folded-git-component",
                replacing("README.md", with: ".GIT/config")
            ),
            (
                "path-space",
                replacing("README.md", with: "READ ME.md")
            ),
            (
                "path-empty-component",
                replacing("README.md", with: "Docs//README.md")
            ),
            (
                "path-component-too-long",
                replacing(
                    "README.md",
                    with: "\(String(repeating: "a", count: 256)).md"
                )
            ),
            (
                "path-too-long",
                replacing("README.md", with: String(repeating: "a", count: 4097))
            ),
            (
                "sort-order",
                Data(reorderedLines.map(String.init).joined(separator: "\n").utf8)
            ),
            (
                "duplicate-path",
                duplicatePath
            ),
            (
                "case-folded-duplicate-path",
                caseFoldedDuplicatePath
            ),
            (
                "file-directory-prefix-conflict",
                fileDirectoryPrefixConflict
            ),
            (
                "case-folded-file-directory-prefix-conflict",
                caseFoldedFileDirectoryPrefixConflict
            ),
        ]

        for (label, manifest) in cases {
            let authorized = try authorizeManifest(
                bytes: manifest,
                name: "\(label).manifest"
            )
            XCTAssertThrowsError(
                try SourceManifestAdmission.admit(
                    authorizedFile: authorized,
                    expectedRole: .baseline
                ),
                "expected nonCanonicalManifest for \(label)"
            ) { error in
                XCTAssertEqual(
                    error as? SourceManifestAdmissionError,
                    .nonCanonicalManifest
                )
            }
        }
    }

    func testBuilderRejectsNonCanonicalEntryInputs() throws {
        let cases: [(String, [SourceManifestEntry])] = [
            ("empty", []),
            (
                "blob",
                [
                    SourceManifestEntry(
                        mode: .regular,
                        gitBlobSHA1:
                            "A" + String(Self.blobA.dropFirst()),
                        byteCount: 1,
                        sha256: Self.shaA,
                        path: "README.md"
                    ),
                ]
            ),
            (
                "sha256",
                [
                    SourceManifestEntry(
                        mode: .regular,
                        gitBlobSHA1: Self.blobA,
                        byteCount: 1,
                        sha256: Self.shaA.uppercased(),
                        path: "README.md"
                    ),
                ]
            ),
            (
                "path",
                [
                    SourceManifestEntry(
                        mode: .regular,
                        gitBlobSHA1: Self.blobA,
                        byteCount: 1,
                        sha256: Self.shaA,
                        path: "../README.md"
                    ),
                ]
            ),
            (
                "duplicate",
                [
                    SourceManifestEntry(
                        mode: .regular,
                        gitBlobSHA1: Self.blobA,
                        byteCount: 1,
                        sha256: Self.shaA,
                        path: "README.md"
                    ),
                    SourceManifestEntry(
                        mode: .executable,
                        gitBlobSHA1: Self.blobB,
                        byteCount: 2,
                        sha256: Self.shaB,
                        path: "README.md"
                    ),
                ]
            ),
            (
                "case-folded-duplicate",
                [
                    SourceManifestEntry(
                        mode: .regular,
                        gitBlobSHA1: Self.blobA,
                        byteCount: 1,
                        sha256: Self.shaA,
                        path: "README.md"
                    ),
                    SourceManifestEntry(
                        mode: .executable,
                        gitBlobSHA1: Self.blobB,
                        byteCount: 2,
                        sha256: Self.shaB,
                        path: "readme.md"
                    ),
                ]
            ),
            (
                "file-directory-prefix-conflict",
                [
                    SourceManifestEntry(
                        mode: .regular,
                        gitBlobSHA1: Self.blobA,
                        byteCount: 1,
                        sha256: Self.shaA,
                        path: "Sources"
                    ),
                    SourceManifestEntry(
                        mode: .regular,
                        gitBlobSHA1: Self.blobC,
                        byteCount: 7,
                        sha256: Self.shaC,
                        path: "Sources-archive"
                    ),
                    SourceManifestEntry(
                        mode: .executable,
                        gitBlobSHA1: Self.blobB,
                        byteCount: 2,
                        sha256: Self.shaB,
                        path: "Sources/App/main.swift"
                    ),
                ]
            ),
            (
                "case-folded-file-directory-prefix-conflict",
                [
                    SourceManifestEntry(
                        mode: .regular,
                        gitBlobSHA1: Self.blobA,
                        byteCount: 1,
                        sha256: Self.shaA,
                        path: "Sources"
                    ),
                    SourceManifestEntry(
                        mode: .regular,
                        gitBlobSHA1: Self.blobC,
                        byteCount: 7,
                        sha256: Self.shaC,
                        path: "sources-archive"
                    ),
                    SourceManifestEntry(
                        mode: .executable,
                        gitBlobSHA1: Self.blobB,
                        byteCount: 2,
                        sha256: Self.shaB,
                        path: "sources/App/main.swift"
                    ),
                ]
            ),
        ]

        for (label, entries) in cases {
            XCTAssertThrowsError(
                try SourceManifestAdmission.manifestBytes(
                    role: .baseline,
                    gitCommitSHA1: Self.baselineCommit,
                    gitTreeSHA1: Self.baselineTree,
                    entries: entries
                ),
                "expected nonCanonicalManifest for \(label)"
            ) { error in
                XCTAssertEqual(
                    error as? SourceManifestAdmissionError,
                    .nonCanonicalManifest
                )
            }
        }

        XCTAssertThrowsError(
            try SourceManifestAdmission.manifestBytes(
                role: .baseline,
                gitCommitSHA1: Self.baselineCommit.uppercased(),
                gitTreeSHA1: Self.baselineTree,
                entries: baselineEntries()
            )
        ) { error in
            XCTAssertEqual(
                error as? SourceManifestAdmissionError,
                .nonCanonicalManifest
            )
        }
    }

    func testCapturedManifestBytesSurviveBackingFileRewrite() throws {
        let url = caseRoot.appendingPathComponent("stable-source.manifest")
        try Self.pinnedBaselineManifestBytes.write(to: url)
        let captured = try AdmittedFile.capture(
            absolutePath: url.path,
            maximumBytes: 4_096
        )
        let authorized = try authorizeCapturedFile(captured)

        try Data("role=candidate\n".utf8).write(to: url)

        let admitted = try SourceManifestAdmission.admit(
            authorizedFile: authorized,
            expectedRole: .baseline
        )
        XCTAssertEqual(admitted.authorizedFile.file.bytes, Self.pinnedBaselineManifestBytes)
        XCTAssertNotEqual(admitted.authorizedFile.file.bytes, try Data(contentsOf: url))
        XCTAssertEqual(admitted.gitCommitSHA1, Self.baselineCommit)
        XCTAssertEqual(admitted.gitTreeSHA1, Self.baselineTree)
    }

    func testMatchesGenuinelyVerifiedRunClaimForBaselineAndCandidate()
        throws
    {
        let fixture = try makeVerifiedRunFixture()

        let baselineMatch = try SourceManifestAdmission.match(
            fixture.baselineManifest,
            to: fixture.signedClaim
        )
        XCTAssertEqual(
            baselineMatch.matchID.rawValue,
            expectedMatchID(
                signedClaim: fixture.signedClaim,
                manifest: fixture.baselineManifest
            )
        )
        XCTAssertEqual(baselineMatch.signedClaimID, fixture.signedClaim.claimID)
        XCTAssertEqual(baselineMatch.sourceManifest, fixture.baselineManifest)

        let candidateMatch = try SourceManifestAdmission.match(
            fixture.candidateManifest,
            to: fixture.signedClaim
        )
        XCTAssertEqual(
            candidateMatch.matchID.rawValue,
            expectedMatchID(
                signedClaim: fixture.signedClaim,
                manifest: fixture.candidateManifest
            )
        )
        XCTAssertEqual(candidateMatch.signedClaimID, fixture.signedClaim.claimID)
        XCTAssertEqual(candidateMatch.sourceManifest, fixture.candidateManifest)
        XCTAssertNotEqual(baselineMatch.matchID, candidateMatch.matchID)
    }

    func testMatchRejectsTypedClaimSourceMismatches() throws {
        let fixture = try makeVerifiedRunFixture()
        let alternateAuthorization = try makeVerifiedRunFixture(
            baselineSourceSigner: .alternateSource
        )
        let alternateCommit = try signedClaim(
            from: fixture,
            baselineCommit: Self.differentSHA1
        )
        let alternateTree = try signedClaim(
            from: fixture,
            baselineTree: Self.differentSHA1
        )

        let cases: [(String, OperatorSignedRunClaim, SourceManifestAdmissionError)] = [
            (
                "authorization",
                alternateAuthorization.signedClaim,
                .sourceAuthorizationIDMismatch(role: .baseline)
            ),
            (
                "commit",
                alternateCommit,
                .gitCommitMismatch(role: .baseline)
            ),
            (
                "tree",
                alternateTree,
                .gitTreeMismatch(role: .baseline)
            ),
        ]

        for (label, claim, expectedError) in cases {
            XCTAssertThrowsError(
                try SourceManifestAdmission.match(
                    fixture.baselineManifest,
                    to: claim
                ),
                "expected \(expectedError) for \(label)"
            ) { error in
                XCTAssertEqual(error as? SourceManifestAdmissionError, expectedError)
            }
        }
    }

    func testGitToolPolicyReferenceRetainsExactVerifiedClaimAndStaysInert()
        throws
    {
        let policyDocument = try makeAnchoredGitToolPolicy()
        let first = try makeVerifiedRunFixture(
            toolManifestSHA256: policyDocument.policySHA256,
            toolManifestBytes: policyDocument.policyBytes,
            runtimePolicySHA256: policyDocument.runtimePolicySHA256
        )
        let second = try makeVerifiedRunFixture(
            toolManifestSHA256: policyDocument.policySHA256,
            toolManifestBytes: policyDocument.policyBytes,
            runtimePolicySHA256: policyDocument.runtimePolicySHA256,
            resultPairID: hex(6)
        )

        let firstReference = try GitToolPolicyVerifier.reference(
            signedClaim: first.signedClaim,
            policyDocument: policyDocument
        )
        let secondReference = try GitToolPolicyVerifier.reference(
            signedClaim: second.signedClaim,
            policyDocument: policyDocument
        )

        XCTAssertNotEqual(
            first.signedClaim.claimID,
            second.signedClaim.claimID
        )
        XCTAssertEqual(firstReference.signedClaim, first.signedClaim)
        XCTAssertEqual(secondReference.signedClaim, second.signedClaim)
        XCTAssertEqual(firstReference.policyDocument, policyDocument)
        XCTAssertEqual(secondReference.policyDocument, policyDocument)
        XCTAssertNotEqual(firstReference, secondReference)
        XCTAssertFalse(firstReference.canImportGitObjects)
        XCTAssertFalse(firstReference.canMutateFileSystem)
        XCTAssertFalse(firstReference.canSpawn)
        XCTAssertFalse(firstReference.canBuild)
        XCTAssertFalse(firstReference.canLoadModel)
        XCTAssertFalse(firstReference.canReserveOutput)
        XCTAssertFalse(firstReference.canPublish)
    }

    func testGitToolPolicyReferenceRejectsEverySignedClaimMismatch() throws {
        let policyDocument = try makeAnchoredGitToolPolicy()
        let digestMismatch = try makeVerifiedRunFixture(
            toolManifestSHA256: hex(27),
            toolManifestBytes: policyDocument.policyBytes,
            runtimePolicySHA256: policyDocument.runtimePolicySHA256
        )
        let byteCountMismatch = try makeVerifiedRunFixture(
            toolManifestSHA256: policyDocument.policySHA256,
            toolManifestBytes: policyDocument.policyBytes + 1,
            runtimePolicySHA256: policyDocument.runtimePolicySHA256
        )
        let runtimeMismatch = try makeVerifiedRunFixture(
            toolManifestSHA256: policyDocument.policySHA256,
            toolManifestBytes: policyDocument.policyBytes,
            runtimePolicySHA256: hex(28)
        )
        let foreignPolicyDocument = try makeAnchoredGitToolPolicy(
            executableSHA256: hex(29)
        )
        let foreignClaim = try makeVerifiedRunFixture(
            toolManifestSHA256: foreignPolicyDocument.policySHA256,
            toolManifestBytes: foreignPolicyDocument.policyBytes,
            runtimePolicySHA256:
                foreignPolicyDocument.runtimePolicySHA256,
            resultPairID: hex(7)
        )

        let cases: [
            (
                String,
                AnchoredGitToolPolicyDocument,
                OperatorSignedRunClaim,
                GitToolPolicyClaimReferenceError
            )
        ] = [
            (
                "tool manifest digest",
                policyDocument,
                digestMismatch.signedClaim,
                .toolManifestDigestMismatch
            ),
            (
                "tool manifest byte count",
                policyDocument,
                byteCountMismatch.signedClaim,
                .toolManifestByteCountMismatch
            ),
            (
                "runtime policy digest",
                policyDocument,
                runtimeMismatch.signedClaim,
                .runtimePolicyDigestMismatch
            ),
            (
                "foreign claim policy",
                policyDocument,
                foreignClaim.signedClaim,
                .toolManifestDigestMismatch
            ),
        ]

        for (label, document, signedClaim, expectedError) in cases {
            XCTAssertThrowsError(
                try GitToolPolicyVerifier.reference(
                    signedClaim: signedClaim,
                    policyDocument: document
                ),
                "expected \(expectedError) for \(label)"
            ) { error in
                XCTAssertEqual(
                    error as? GitToolPolicyClaimReferenceError,
                    expectedError
                )
            }
        }

        let localClaim = try makeVerifiedRunFixture(
            toolManifestSHA256: policyDocument.policySHA256,
            toolManifestBytes: policyDocument.policyBytes,
            runtimePolicySHA256: policyDocument.runtimePolicySHA256
        )
        XCTAssertThrowsError(
            try GitToolPolicyVerifier.reference(
                signedClaim: localClaim.signedClaim,
                policyDocument: foreignPolicyDocument
            )
        ) { error in
            XCTAssertEqual(
                error as? GitToolPolicyClaimReferenceError,
                .toolManifestDigestMismatch
            )
        }
    }

    func testRuntimeDenialReferenceMatchReturnsOnlyTypedRefusalAndStaysInert()
        throws
    {
        let runtimeDocument = try makeAnchoredGitRuntimeDenialPolicy()
        let policyDocument = try makeAnchoredGitToolPolicy(
            runtimePolicySHA256: runtimeDocument.policySHA256
        )
        let first = try makeVerifiedRunFixture(
            toolManifestSHA256: policyDocument.policySHA256,
            toolManifestBytes: policyDocument.policyBytes,
            runtimePolicySHA256: runtimeDocument.policySHA256
        )
        let second = try makeVerifiedRunFixture(
            toolManifestSHA256: policyDocument.policySHA256,
            toolManifestBytes: policyDocument.policyBytes,
            runtimePolicySHA256: runtimeDocument.policySHA256,
            resultPairID: hex(8)
        )
        let firstReference = try GitToolPolicyVerifier.reference(
            signedClaim: first.signedClaim,
            policyDocument: policyDocument
        )
        let secondReference = try GitToolPolicyVerifier.reference(
            signedClaim: second.signedClaim,
            policyDocument: policyDocument
        )
        let effects = RuntimePolicyFakeEffects()

        XCTAssertNotEqual(
            firstReference.signedClaim.claimID,
            secondReference.signedClaim.claimID
        )
        for reference in [firstReference, secondReference] {
            XCTAssertThrowsError(
                try GitRuntimePolicyDenialVerifier.requireUnavailable(
                    policyDocument: runtimeDocument,
                    toolPolicyReference: reference
                )
            ) { error in
                XCTAssertEqual(
                    error as? GitRuntimePolicyDenialError,
                    .supportedSandboxMechanismUnavailable
                )
            }
        }

        let runtimeText = String(
            decoding: runtimeDocument.policyFile.bytes,
            as: UTF8.self
        )
        XCTAssertFalse(runtimeText.contains(runtimeDocument.policySHA256))
        XCTAssertFalse(runtimeText.contains(policyDocument.policySHA256))
        XCTAssertFalse(
            runtimeText.contains(firstReference.signedClaim.claimID.rawValue)
        )
        XCTAssertFalse(runtimeDocument.canImportGitObjects)
        XCTAssertFalse(runtimeDocument.canMutateFileSystem)
        XCTAssertFalse(runtimeDocument.canAccessNetwork)
        XCTAssertFalse(runtimeDocument.canConsumePack)
        XCTAssertFalse(runtimeDocument.canSpawn)
        XCTAssertFalse(runtimeDocument.canBuild)
        XCTAssertFalse(runtimeDocument.canLoadModel)
        XCTAssertFalse(runtimeDocument.canReserveOutput)
        XCTAssertFalse(runtimeDocument.canPublish)
        XCTAssertEqual(effects, .zero)
    }

    func testRuntimeDenialReferenceRejectsDigestSubstitution()
        throws
    {
        let runtimeDocument = try makeAnchoredGitRuntimeDenialPolicy()
        let foreignRuntimeDocument = try makeAnchoredGitRuntimeDenialPolicy(
            selfGuardSHA256: hex(31)
        )
        let foreignPolicyDocument = try makeAnchoredGitToolPolicy(
            runtimePolicySHA256: foreignRuntimeDocument.policySHA256
        )
        let foreignClaim = try makeVerifiedRunFixture(
            toolManifestSHA256: foreignPolicyDocument.policySHA256,
            toolManifestBytes: foreignPolicyDocument.policyBytes,
            runtimePolicySHA256: foreignRuntimeDocument.policySHA256
        )
        let foreignReference = try GitToolPolicyVerifier.reference(
            signedClaim: foreignClaim.signedClaim,
            policyDocument: foreignPolicyDocument
        )
        let effects = RuntimePolicyFakeEffects()

        XCTAssertThrowsError(
            try GitRuntimePolicyDenialVerifier.requireUnavailable(
                policyDocument: runtimeDocument,
                toolPolicyReference: foreignReference
            )
        ) { error in
            XCTAssertEqual(
                error as? GitRuntimePolicyDenialError,
                .runtimePolicyDigestMismatch
            )
        }
        XCTAssertEqual(effects, .zero)
    }

    func testRuntimeDenialReferenceRejectsEveryToolResourceMismatch()
        throws
    {
        let runtimeDocument = try makeAnchoredGitRuntimeDenialPolicy()
        let ceilings = GitToolPolicyVerifier.phase1ResourceCeilings
        let cases: [
            (
                GitRuntimePolicyDenialResourceField,
                GitToolPolicyResourceLimits
            )
        ] = [
            (
                .cpuTimeoutSeconds,
                toolLimits(cpuTimeoutSeconds: ceilings.cpuTimeoutSeconds - 1)
            ),
            (
                .maxAddressSpaceBytes,
                toolLimits(
                    maxAddressSpaceBytes:
                        ceilings.maxAddressSpaceBytes - 1
                )
            ),
            (
                .maxFileSizeBytes,
                toolLimits(
                    maxFileSizeBytes: ceilings.maxFileSizeBytes - 1
                )
            ),
            (
                .maxOpenFiles,
                toolLimits(maxOpenFiles: ceilings.maxOpenFiles - 1)
            ),
            (
                .maxObjectDatabaseBytes,
                toolLimits(
                    maxObjectDatabaseBytes:
                        ceilings.maxObjectDatabaseBytes - 1
                )
            ),
            (
                .maxStdoutBytes,
                toolLimits(maxStdoutBytes: ceilings.maxStdoutBytes - 1)
            ),
            (
                .maxStderrBytes,
                toolLimits(maxStderrBytes: ceilings.maxStderrBytes - 1)
            ),
            (
                .wallTimeoutMilliseconds,
                toolLimits(
                    wallTimeoutMilliseconds:
                        ceilings.wallTimeoutMilliseconds - 1
                )
            ),
        ]
        let effects = RuntimePolicyFakeEffects()

        for (field, limits) in cases {
            let policyDocument = try makeAnchoredGitToolPolicy(
                runtimePolicySHA256: runtimeDocument.policySHA256,
                limits: limits
            )
            let claim = try makeVerifiedRunFixture(
                toolManifestSHA256: policyDocument.policySHA256,
                toolManifestBytes: policyDocument.policyBytes,
                runtimePolicySHA256: runtimeDocument.policySHA256
            )
            let reference = try GitToolPolicyVerifier.reference(
                signedClaim: claim.signedClaim,
                policyDocument: policyDocument
            )

            XCTAssertThrowsError(
                try GitRuntimePolicyDenialVerifier.requireUnavailable(
                    policyDocument: runtimeDocument,
                    toolPolicyReference: reference
                )
            ) { error in
                XCTAssertEqual(
                    error as? GitRuntimePolicyDenialError,
                    .toolPolicyResourceMismatch(field)
                )
            }
            XCTAssertEqual(effects, .zero)
        }
    }

    private func makeVerifiedRunFixture(
        baselineRole: RunSourceRole = .baseline,
        baselineCommit: String = baselineCommit,
        baselineTree: String = baselineTree,
        baselineEntries: [SourceManifestEntry] = baselineEntries(),
        baselineSourceSigner: SourceSigner = .primarySource,
        toolManifestSHA256: String = String(repeating: "10", count: 32),
        toolManifestBytes: UInt64 = 19,
        runtimePolicySHA256: String = String(repeating: "0d", count: 32),
        resultPairID: String = String(repeating: "05", count: 32)
    ) throws -> VerifiedRunFixture {
        let worker = try authorizePayload(
            bytes: Data("worker\n".utf8),
            name: "worker-\(UUID().uuidString).swift",
            purpose: .workerBytes,
            signer: .worker
        )
        let baselineBytes = try SourceManifestAdmission.manifestBytes(
            role: baselineRole,
            gitCommitSHA1: baselineCommit,
            gitTreeSHA1: baselineTree,
            entries: baselineEntries
        )
        let baselineAuthorized = try authorizeManifest(
            bytes: baselineBytes,
            name: "baseline-\(UUID().uuidString).manifest",
            signer: baselineSourceSigner
        )
        let baselineManifest = try SourceManifestAdmission.admit(
            authorizedFile: baselineAuthorized,
            expectedRole: baselineRole
        )

        let candidateBytes = try SourceManifestAdmission.manifestBytes(
            role: .candidate,
            gitCommitSHA1: Self.candidateCommit,
            gitTreeSHA1: Self.candidateTree,
            entries: [
                SourceManifestEntry(
                    mode: .regular,
                    gitBlobSHA1: Self.blobA,
                    byteCount: 7,
                    sha256: Self.shaA,
                    path: "Candidate.swift"
                ),
            ]
        )
        let candidateAuthorized = try authorizeManifest(
            bytes: candidateBytes,
            name: "candidate-\(UUID().uuidString).manifest",
            signer: .candidateSource
        )
        let candidateManifest = try SourceManifestAdmission.admit(
            authorizedFile: candidateAuthorized,
            expectedRole: .candidate
        )

        let runner = try capturePayload(
            Data("runner\n".utf8),
            name: "runner-\(UUID().uuidString)"
        )
        let inputs = try RunAuthorizedInputs(
            worker: worker,
            baselineSourceManifest: baselineAuthorized,
            candidateSourceManifest: candidateAuthorized
        )
        let keyPolicy = try makeKeyPolicy()
        let expectations = OperatorRunClaimAdmissionExpectations(
            keyPolicy: keyPolicy,
            hostAdmissionID: hex(4),
            runner: runner,
            resultPairID: resultPairID,
            inputs: inputs
        )
        let fields = OperatorRunClaimFields(
            subject: .absorbedMLALoadedResultPair,
            operatorKeyID: keyPolicy.activeOperatorKeyID,
            operatorKeyPolicySHA256: keyPolicy.policySHA256,
            hostAdmissionID: expectations.hostAdmissionID,
            runner: OperatorRunClaimByteIdentity(
                sha256: runner.sha256,
                byteCount: UInt64(runner.bytes.count)
            ),
            worker: reference(worker),
            policies: OperatorRunClaimPolicyReferences(
                sourceSHA256: hex(10),
                dependencySHA256: hex(11),
                buildSHA256: hex(12),
                runtimeSHA256: runtimePolicySHA256,
                preflightSHA256: hex(14),
                publicationSHA256: hex(15)
            ),
            toolManifest: OperatorRunClaimByteIdentity(
                sha256: toolManifestSHA256,
                byteCount: toolManifestBytes
            ),
            baseline: sourceReference(
                role: baselineRole,
                manifest: baselineManifest,
                buildReceiptID: hex(17),
                binarySHA256: hex(18),
                binaryBytes: 1001
            ),
            candidate: sourceReference(
                role: .candidate,
                manifest: candidateManifest,
                buildReceiptID: hex(19),
                binarySHA256: hex(20),
                binaryBytes: 1002
            ),
            model: OperatorRunClaimAuthorizedPayloadReference(
                authorizationID: hex(21),
                payload: OperatorRunClaimByteIdentity(sha256: hex(22), byteCount: 55)
            ),
            tokenizer: OperatorRunClaimAuthorizedPayloadReference(
                authorizationID: hex(23),
                payload: OperatorRunClaimByteIdentity(sha256: hex(24), byteCount: 66)
            ),
            workload: OperatorRunClaimAuthorizedPayloadReference(
                authorizationID: hex(25),
                payload: OperatorRunClaimByteIdentity(sha256: hex(26), byteCount: 77)
            ),
            resultPairID: expectations.resultPairID
        )
        let claimBytes = try OperatorSignedRunClaimVerifier.claimBytes(fields: fields)
        let signature = try signingKey(.runClaim).signingMaterial.signature(for: claimBytes)
            .base64EncodedString()
        let signedClaim = try OperatorSignedRunClaimVerifier.verify(
            claimBytes: claimBytes,
            signatureBase64: signature,
            expectations: expectations
        )

        return VerifiedRunFixture(
            baselineManifest: baselineManifest,
            candidateManifest: candidateManifest,
            signedClaim: signedClaim,
            fields: fields,
            expectations: expectations
        )
    }

    private func makeAnchoredGitToolPolicy(
        executableSHA256: String = String(repeating: "ab", count: 32),
        runtimePolicySHA256: String = String(repeating: "0d", count: 32),
        limits: GitToolPolicyResourceLimits =
            GitToolPolicyVerifier.phase1ResourceCeilings
    ) throws -> AnchoredGitToolPolicyDocument {
        let fields = GitToolPolicyFields(
            policyGeneration: 7,
            validFromUnixSeconds: 1_900_000_000,
            validUntilUnixSeconds: 2_100_000_000,
            executableSHA256: executableSHA256,
            executableBytes: 1_048_576,
            executableMachOUUID: String(repeating: "cd", count: 16),
            executableCodeDirectorySHA256:
                String(repeating: "de", count: 32),
            runtimeClosureManifestSHA256:
                String(repeating: "bc", count: 32),
            runtimeClosureManifestBytes: 4_096,
            runtimePolicySHA256: runtimePolicySHA256,
            limits: limits
        )
        let bytes = try GitToolPolicyVerifier.policyBytes(fields: fields)
        let policyFile = try capturePayload(
            bytes,
            name: "git-tool-\(UUID().uuidString).policy"
        )
        return try GitToolPolicyVerifier.anchor(
            policyFile: policyFile,
            trustAnchor: GitToolPolicyTrustAnchor(
                expectedCurrentPolicySHA256: policyFile.sha256,
                expectedCurrentPolicyBytes:
                    UInt64(policyFile.bytes.count),
                minimumPolicyGeneration: 7,
                verificationUnixSeconds: 2_000_000_000
            )
        )
    }

    private func toolLimits(
        maxObjectDatabaseBytes: UInt64 =
            GitToolPolicyVerifier.phase1ResourceCeilings
                .maxObjectDatabaseBytes,
        maxStdoutBytes: UInt64 =
            GitToolPolicyVerifier.phase1ResourceCeilings.maxStdoutBytes,
        maxStderrBytes: UInt64 =
            GitToolPolicyVerifier.phase1ResourceCeilings.maxStderrBytes,
        wallTimeoutMilliseconds: UInt64 =
            GitToolPolicyVerifier.phase1ResourceCeilings
                .wallTimeoutMilliseconds,
        cpuTimeoutSeconds: UInt64 =
            GitToolPolicyVerifier.phase1ResourceCeilings
                .cpuTimeoutSeconds,
        maxAddressSpaceBytes: UInt64 =
            GitToolPolicyVerifier.phase1ResourceCeilings
                .maxAddressSpaceBytes,
        maxFileSizeBytes: UInt64 =
            GitToolPolicyVerifier.phase1ResourceCeilings
                .maxFileSizeBytes,
        maxOpenFiles: UInt64 =
            GitToolPolicyVerifier.phase1ResourceCeilings.maxOpenFiles
    ) -> GitToolPolicyResourceLimits {
        let base = GitToolPolicyVerifier.phase1ResourceCeilings
        return GitToolPolicyResourceLimits(
            maxPackBytes: base.maxPackBytes,
            maxPackObjects: base.maxPackObjects,
            maxCommitBytes: base.maxCommitBytes,
            maxSingleInflatedObjectBytes:
                base.maxSingleInflatedObjectBytes,
            maxTotalInflatedBytes: base.maxTotalInflatedBytes,
            maxCompressionRatio: base.maxCompressionRatio,
            maxTreeDepth: base.maxTreeDepth,
            maxTreeCount: base.maxTreeCount,
            maxObjectDatabaseBytes: maxObjectDatabaseBytes,
            maxStdoutBytes: maxStdoutBytes,
            maxStderrBytes: maxStderrBytes,
            wallTimeoutMilliseconds: wallTimeoutMilliseconds,
            cpuTimeoutSeconds: cpuTimeoutSeconds,
            maxAddressSpaceBytes: maxAddressSpaceBytes,
            maxFileSizeBytes: maxFileSizeBytes,
            maxOpenFiles: maxOpenFiles
        )
    }

    private func makeAnchoredGitRuntimeDenialPolicy(
        selfGuardSHA256: String = String(repeating: "a", count: 64),
        limits: GitToolPolicyResourceLimits =
            GitToolPolicyVerifier.phase1ResourceCeilings
    ) throws -> AnchoredGitRuntimePolicyDenialDocument {
        let fields = GitRuntimePolicyDenialFields(
            policyGeneration: 7,
            validFromUnixSeconds: 1_900_000_000,
            validUntilUnixSeconds: 2_100_000_000,
            selfGuardSHA256: selfGuardSHA256,
            selfGuardBytes: 1_048_576,
            selfGuardMachOUUID: String(repeating: "b", count: 32),
            selfGuardCodeDirectorySHA256:
                String(repeating: "c", count: 64),
            selfGuardRuntimeClosureID:
                String(repeating: "d", count: 64),
            gitRuntimeClosureID:
                String(repeating: "e", count: 64),
            limits: limits,
            terminationGraceMilliseconds: 2_000,
            reapTimeoutMilliseconds: 5_000
        )
        let bytes = try GitRuntimePolicyDenialVerifier.policyBytes(
            fields: fields
        )
        let policyFile = try capturePayload(
            bytes,
            name: "git-runtime-\(UUID().uuidString).policy"
        )
        return try GitRuntimePolicyDenialVerifier.anchor(
            policyFile: policyFile,
            trustAnchor: GitRuntimePolicyDenialTrustAnchor(
                expectedCurrentPolicySHA256: policyFile.sha256,
                expectedCurrentPolicyBytes:
                    UInt64(policyFile.bytes.count),
                minimumPolicyGeneration: 7,
                verificationUnixSeconds: 2_000_000_000
            )
        )
    }

    private func signedClaim(
        from fixture: VerifiedRunFixture,
        baselineCommit: String? = nil,
        baselineTree: String? = nil
    ) throws -> OperatorSignedRunClaim {
        let existing = fixture.fields
        let baseline = OperatorRunClaimSourceReference(
            role: existing.baseline.role,
            sourceManifest: existing.baseline.sourceManifest,
            gitCommitSHA1:
                baselineCommit ?? existing.baseline.gitCommitSHA1,
            gitTreeSHA1:
                baselineTree ?? existing.baseline.gitTreeSHA1,
            route: existing.baseline.route,
            slot: existing.baseline.slot,
            buildReceiptID: existing.baseline.buildReceiptID,
            binary: existing.baseline.binary
        )
        let fields = OperatorRunClaimFields(
            subject: existing.subject,
            operatorKeyID: existing.operatorKeyID,
            operatorKeyPolicySHA256:
                existing.operatorKeyPolicySHA256,
            hostAdmissionID: existing.hostAdmissionID,
            runner: existing.runner,
            worker: existing.worker,
            policies: existing.policies,
            toolManifest: existing.toolManifest,
            baseline: baseline,
            candidate: existing.candidate,
            model: existing.model,
            tokenizer: existing.tokenizer,
            workload: existing.workload,
            resultPairID: existing.resultPairID
        )
        let claimBytes = try OperatorSignedRunClaimVerifier.claimBytes(
            fields: fields
        )
        let signature = try signingKey(.runClaim).signingMaterial
            .signature(for: claimBytes)
            .base64EncodedString()
        return try OperatorSignedRunClaimVerifier.verify(
            claimBytes: claimBytes,
            signatureBase64: signature,
            expectations: fixture.expectations
        )
    }

    private func makeKeyPolicy() throws -> AdmittedOperatorKeyPolicy {
        let root = try signingKey(.rootPolicy)
        let active = try signingKey(.runClaim)
        let fields = OperatorKeyPolicyFields(
            rootKeyID: root.keyID,
            policyGeneration: 7,
            validFromUnixSeconds: 1_900_000_000,
            validUntilUnixSeconds: 2_100_000_000,
            activeOperatorKeyID: active.keyID,
            activeOperatorPublicKeyBase64: active.publicKeyBase64,
            activeOperatorScope: .runClaim,
            allowedClaimSubject: .absorbedMLALoadedResultPair,
            revokedOperatorKeyIDs: []
        )
        let policyBytes = try OperatorKeyPolicyVerifier.policyBytes(fields: fields)
        let policyFile = try capturePayload(
            policyBytes,
            name: "policy-\(UUID().uuidString).operator-key-policy"
        )
        let signature = try root.signingMaterial.signature(for: policyBytes)
            .base64EncodedString()
        return try OperatorKeyPolicyVerifier.admit(
            policyFile: policyFile,
            rootSignatureBase64: signature,
            trustAnchor: OperatorKeyPolicyTrustAnchor(
                rootPublicKeyBase64: root.publicKeyBase64,
                rootKeyID: root.keyID,
                expectedCurrentPolicySHA256: policyFile.sha256,
                minimumPolicyGeneration: 7,
                verificationUnixSeconds: 2_000_000_000
            )
        )
    }

    private func authorizeManifest(
        bytes: Data,
        name: String,
        purpose: OperatorAuthorizationPurpose = .sourceManifest,
        signer: SourceSigner = .primarySource
    ) throws -> OperatorAuthorizedFile {
        try authorizePayload(
            bytes: bytes,
            name: name,
            purpose: purpose,
            signer: signer
        )
    }

    private func authorizeCapturedFile(
        _ admittedFile: AdmittedFile,
        purpose: OperatorAuthorizationPurpose = .sourceManifest,
        signer: SourceSigner = .primarySource
    ) throws -> OperatorAuthorizedFile {
        let key = try signingKey(signer)
        let claimBytes = try OperatorAuthorization.claimBytes(
            purpose: purpose,
            payloadSHA256: admittedFile.sha256,
            payloadByteCount: UInt64(admittedFile.bytes.count)
        )
        let signature = try key.signingMaterial.signature(for: claimBytes)
            .base64EncodedString()
        return try OperatorAuthorization.verify(
            admittedFile: admittedFile,
            expectedPurpose: purpose,
            claimBytes: claimBytes,
            signatureBase64: signature,
            publicKeyBase64: key.publicKeyBase64,
            allowedKeyID: key.keyID
        )
    }

    private func authorizePayload(
        bytes: Data,
        name: String,
        purpose: OperatorAuthorizationPurpose,
        signer: SourceSigner
    ) throws -> OperatorAuthorizedFile {
        try authorizeCapturedFile(
            capturePayload(bytes, name: name),
            purpose: purpose,
            signer: signer
        )
    }

    private func capturePayload(_ bytes: Data, name: String) throws -> AdmittedFile {
        let url = caseRoot.appendingPathComponent(name)
        try bytes.write(to: url)
        return try AdmittedFile.capture(
            absolutePath: url.path,
            maximumBytes: 64 * 1024
        )
    }

    private static func baselineEntries() -> [SourceManifestEntry] {
        [
            SourceManifestEntry(
                mode: .regular,
                gitBlobSHA1: blobA,
                byteCount: 0,
                sha256: shaA,
                path: "README.md"
            ),
            SourceManifestEntry(
                mode: .executable,
                gitBlobSHA1: blobB,
                byteCount: 42,
                sha256: shaB,
                path: "Sources/App/main.swift"
            ),
        ]
    }

    private func baselineEntries() -> [SourceManifestEntry] {
        Self.baselineEntries()
    }

    private func reference(
        _ authorized: OperatorAuthorizedFile
    ) -> OperatorRunClaimAuthorizedPayloadReference {
        OperatorRunClaimAuthorizedPayloadReference(
            authorizationID: authorized.authorizationID.rawValue,
            payload: OperatorRunClaimByteIdentity(
                sha256: authorized.file.sha256,
                byteCount: UInt64(authorized.file.bytes.count)
            )
        )
    }

    private func sourceReference(
        role: RunSourceRole,
        manifest: AdmittedSourceManifest,
        buildReceiptID: String,
        binarySHA256: String,
        binaryBytes: UInt64
    ) -> OperatorRunClaimSourceReference {
        let claimRole: OperatorRunClaimSourceRole =
            role == .baseline ? .baseline : .candidate
        return OperatorRunClaimSourceReference(
            role: claimRole,
            sourceManifest: reference(manifest.authorizedFile),
            gitCommitSHA1: manifest.gitCommitSHA1,
            gitTreeSHA1: manifest.gitTreeSHA1,
            route: manifest.route,
            slot: manifest.slot,
            buildReceiptID: buildReceiptID,
            binary: OperatorRunClaimByteIdentity(
                sha256: binarySHA256,
                byteCount: binaryBytes
            )
        )
    }

    private func signingKey(_ signer: SourceSigner) throws -> DeterministicSigningKey {
        try DeterministicSigningKey(seed: signer.seed)
    }

    private func replacing(_ source: String, with replacement: String) -> Data {
        let text = String(decoding: Self.pinnedBaselineManifestBytes, as: UTF8.self)
        return Data(text.replacingOccurrences(of: source, with: replacement).utf8)
    }

    private func expectedManifestID(
        _ manifest: AdmittedSourceManifest
    ) -> String {
        typedID([
            "fast-mlx-proof-control-admitted-source-manifest-id-v1",
            "source_authorization_id=\(manifest.authorizedFile.authorizationID.rawValue)",
            "manifest_sha256=\(manifest.authorizedFile.file.sha256)",
            "manifest_bytes=\(UInt64(manifest.authorizedFile.file.bytes.count))",
            "role=\(manifest.role.rawValue)",
            "route=\(manifest.route.rawValue)",
            "slot=\(manifest.slot.rawValue)",
            "git_commit_sha1=\(manifest.gitCommitSHA1)",
            "git_tree_sha1=\(manifest.gitTreeSHA1)",
            "entry_count=\(UInt64(manifest.entries.count))",
        ])
    }

    private func expectedMatchID(
        signedClaim: OperatorSignedRunClaim,
        manifest: AdmittedSourceManifest
    ) -> String {
        typedID([
            "fast-mlx-proof-control-claim-matched-source-manifest-id-v1",
            "run_claim_id=\(signedClaim.claimID.rawValue)",
            "admitted_source_manifest_id=\(manifest.manifestID.rawValue)",
            "role=\(manifest.role.rawValue)",
            "route=\(manifest.route.rawValue)",
            "slot=\(manifest.slot.rawValue)",
            "git_commit_sha1=\(manifest.gitCommitSHA1)",
            "git_tree_sha1=\(manifest.gitTreeSHA1)",
        ])
    }

    private func typedID(_ lines: [String]) -> String {
        SHA256.hash(
            data: Data((lines.joined(separator: "\n") + "\n").utf8)
        )
        .map { String(format: "%02x", $0) }
        .joined()
    }

    private func hex(_ byte: UInt8) -> String {
        String(repeating: String(format: "%02x", byte), count: 32)
    }
}

private struct VerifiedRunFixture {
    let baselineManifest: AdmittedSourceManifest
    let candidateManifest: AdmittedSourceManifest
    let signedClaim: OperatorSignedRunClaim
    let fields: OperatorRunClaimFields
    let expectations: OperatorRunClaimAdmissionExpectations
}

private struct RuntimePolicyFakeEffects: Equatable {
    var spawnCount = 0
    var fileSystemMutationCount = 0
    var networkAccessCount = 0
    var packConsumptionCount = 0
    var sourceMutationCount = 0

    static let zero = Self()
}

private struct DeterministicSigningKey {
    // gitleaks:allow
    let signingMaterial: Curve25519.Signing.PrivateKey
    let publicKeyBase64: String
    let keyID: String

    init(seed: [UInt8]) throws {
        signingMaterial = try Curve25519.Signing.PrivateKey(
            rawRepresentation: Data(seed)
        )
        let publicKeyBytes = signingMaterial.publicKey.rawRepresentation
        publicKeyBase64 = publicKeyBytes.base64EncodedString()
        keyID = Self.sha256Hex(publicKeyBytes)
    }

    private static func sha256Hex(_ bytes: Data) -> String {
        SHA256.hash(data: bytes)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}

private enum SourceSigner {
    case rootPolicy
    case runClaim
    case worker
    case primarySource
    case alternateSource
    case candidateSource

    var seed: [UInt8] {
        let start: UInt8
        switch self {
        case .rootPolicy:
            start = 0x10
        case .runClaim:
            start = 0x30
        case .worker:
            start = 0x50
        case .primarySource:
            start = 0x70
        case .alternateSource:
            start = 0x90
        case .candidateSource:
            start = 0xb0
        }
        return (0..<32).map { start &+ UInt8($0) }
    }
}
