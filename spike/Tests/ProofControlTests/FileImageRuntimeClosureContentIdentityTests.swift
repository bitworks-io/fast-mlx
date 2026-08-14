import CryptoKit
import Foundation
import XCTest

@testable import ProofControl

final class FileImageRuntimeClosureContentIdentityTests: XCTestCase {
    func testReviewedD2C2TypesExist() {
        _ = FileImageRuntimeClosureContentEvidenceID.self
        _ = FileImageRuntimeClosureContentIdentityFailure.self
        _ = FileImageRuntimeClosureContentExpectationComparison.self
        _ = FileImageRuntimeClosureContentIdentityVerifier.self
    }

    func testFrozenCompleteGitAndSelfGuardPredecessorFixtures() throws {
        let git = try Self.fixture(role: .git)
        let selfGuard = try Self.fixture(role: .selfGuard)

        let gitPreimage = Self.expectedPreimage(git)
        let selfGuardPreimage = Self.expectedPreimage(selfGuard)
        XCTAssertEqual(gitPreimage.count, 504)
        XCTAssertEqual(
            Self.sha256Hex(gitPreimage),
            "0f08e67415675fb5ce9738107e1886379" +
                "a4f6b48047b5f49a433e81b62a85e4e"
        )
        XCTAssertEqual(selfGuardPreimage.count, 511)
        XCTAssertEqual(
            Self.sha256Hex(selfGuardPreimage),
            "f0717cc182bf4728f91e6f8fe1b4c04c" +
                "8dddac31f9668d4829ac427a224cd20a"
        )

        XCTAssertEqual(git.reference.declaredFileImageMemberCount, 1)
        XCTAssertEqual(selfGuard.reference.declaredFileImageMemberCount, 1)
        XCTAssertEqual(git.graph.fileImageMemberCount, 1)
        XCTAssertEqual(selfGuard.graph.fileImageMemberCount, 1)
    }

    func testCanonicalRolesProduceExactCompactInertResults() throws {
        let git = try Self.fixture(role: .git)
        let selfGuard = try Self.fixture(role: .selfGuard)

        let gitResult = try Self.compare(git)
        let repeatedGitResult = try Self.compare(git)
        let selfGuardResult = try Self.compare(selfGuard)

        XCTAssertEqual(gitResult, repeatedGitResult)
        XCTAssertEqual(
            gitResult.contentEvidenceID.sha256,
            "0f08e67415675fb5ce9738107e1886379" +
                "a4f6b48047b5f49a433e81b62a85e4e"
        )
        XCTAssertEqual(
            selfGuardResult.contentEvidenceID.sha256,
            "f0717cc182bf4728f91e6f8fe1b4c04c" +
                "8dddac31f9668d4829ac427a224cd20a"
        )
        XCTAssertEqual(
            gitResult.contentEvidenceID.sha256,
            Self.sha256Hex(Self.expectedPreimage(git))
        )
        XCTAssertEqual(
            selfGuardResult.contentEvidenceID.sha256,
            Self.sha256Hex(Self.expectedPreimage(selfGuard))
        )
        try Self.assertResult(gitResult, fixture: git)
        try Self.assertResult(selfGuardResult, fixture: selfGuard)
        XCTAssertEqual(Self.effects, .zero)
    }

    func testIndependentRoleVectorsFreezeExactIdentityBytes() {
        let git = Self.independentPreimage(role: .git)
        let selfGuard = Self.independentPreimage(role: .selfGuard)

        XCTAssertEqual(git.count, 503)
        XCTAssertEqual(
            Self.sha256Hex(git),
            "534995290724819a5f09eca4c81f6bef" +
                "c3b90679805b6bc10f20dae7d3fc8ea3"
        )
        XCTAssertEqual(selfGuard.count, 510)
        XCTAssertEqual(
            Self.sha256Hex(selfGuard),
            "ab6f1b97450fd79ee978a885af195311" +
                "af00dc9f6fe679935e2857c5f4b3da22"
        )
        XCTAssertEqual(Self.effects, .zero)
    }

    func testEveryReviewedIdentityFieldDriftChangesTheDigest() {
        let baseline = Self.independentPreimage(role: .git)
        let baselineID = Self.sha256Hex(baseline)
        let baselineLines = String(decoding: baseline, as: UTF8.self)
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        let replacements = [
            "artifact_role=self-guard",
            "manifest_sha256=" + String(repeating: "f", count: 64),
            "manifest_bytes=124",
            "root_executable_content_evidence_id=" +
                String(repeating: "e", count: 64),
            "dynamic_loader_content_evidence_id=" +
                String(repeating: "d", count: 64),
            "shared_cache_set_id=" + String(repeating: "c", count: 64),
            "file_image_member_count=2",
        ]

        for (index, replacement) in replacements.enumerated() {
            var lines = baselineLines
            lines[index + 1] = replacement
            let changed = Data(lines.joined(separator: "\n").utf8)
            XCTAssertNotEqual(Self.sha256Hex(changed), baselineID)
        }
        XCTAssertEqual(Self.effects, .zero)
    }

    func testMixedAndAllFileImageGraphsJoinPositiveCounts() throws {
        let mixed = try Self.fixture(
            role: .git,
            rootTargets: ["/usr/lib/libD2MixedFile.dylib"],
            specs: [
                .fileImage(
                    "/usr/lib/libD2MixedFile.dylib",
                    targets: ["/usr/lib/libD2MixedCache.dylib"]
                ),
                .sharedCache("/usr/lib/libD2MixedCache.dylib"),
            ]
        )
        let representative = try Self.allFileImageChain(count: 4)
        let upperBound = try Self.allFileImageChain(count: 256)

        let mixedResult = try Self.compare(mixed)
        let representativeResult = try Self.compare(representative)
        let upperBoundResult = try Self.compare(upperBound)

        XCTAssertEqual(mixedResult.memberCount, 2)
        XCTAssertEqual(mixedResult.fileImageMemberCount, 1)
        XCTAssertEqual(representativeResult.memberCount, 4)
        XCTAssertEqual(representativeResult.fileImageMemberCount, 4)
        XCTAssertEqual(upperBoundResult.memberCount, 256)
        XCTAssertEqual(upperBoundResult.fileImageMemberCount, 256)
        XCTAssertEqual(Self.effects, .zero)
    }

    func testReferenceRootLoaderAndGraphFailuresKeepReviewedOrder()
        throws
    {
        let git = try Self.fixture(role: .git)
        let selfGuard = try Self.fixture(role: .selfGuard)
        let badDigestAnchor = Self.anchor(
            for: git.expectationFile,
            role: .git,
            sha256: String(repeating: "f", count: 64)
        )
        Self.assertFailure(
            fixture: git,
            currentAnchor: badDigestAnchor,
            expected: .expectationReference(
                .expectationReanchor(.documentDigestMismatch)
            )
        )
        let changedEvaluationAnchor = Self.anchor(
            for: git.expectationFile,
            role: .git,
            verificationUnixSeconds: 2_000_000_001
        )
        Self.assertFailure(
            fixture: git,
            currentAnchor: changedEvaluationAnchor,
            expected: .expectationReference(
                .expectationEvidenceMismatch
            )
        )
        Self.assertFailure(
            fixture: git,
            root: selfGuard.root,
            expected: .rootRole(expected: .git, actual: .selfGuard)
        )

        let foreignRootReference = try Self.reference(
            for: git,
            rootID: String(repeating: "e", count: 64)
        )
        Self.assertFailure(
            fixture: git,
            expectationReference: foreignRootReference,
            currentAnchor:
                foreignRootReference.anchoredExpectation.trustAnchor,
            expected: .rootManifestIDMismatch
        )
        let foreignLoaderReference = try Self.reference(
            for: git,
            loaderID: String(repeating: "d", count: 64)
        )
        Self.assertFailure(
            fixture: git,
            expectationReference: foreignLoaderReference,
            currentAnchor:
                foreignLoaderReference.anchoredExpectation.trustAnchor,
            expected: .dynamicLoaderManifestIDMismatch
        )
        Self.assertFailure(
            fixture: git,
            memberInventories: [],
            expected: .graph(.memberInventoryCountMismatch)
        )
        let foreignGraph = try Self.fixture(
            role: .git,
            rootTargets: ["/usr/lib/libD2C2Foreign.dylib"],
            specs: [.fileImage("/usr/lib/libD2C2Foreign.dylib")]
        )
        Self.assertFailure(
            fixture: git,
            graphComparison: foreignGraph.graph,
            expected: .graphComparisonMismatch
        )
        XCTAssertEqual(Self.effects, .zero)
    }

    func testCountAndRowFailuresPrecedeSectionAndIdentity()
        throws
    {
        let twoFiles = try Self.fixture(
            role: .git,
            rootTargets: ["/usr/lib/libD2CountA.dylib"],
            specs: [
                .fileImage(
                    "/usr/lib/libD2CountA.dylib",
                    targets: ["/usr/lib/libD2CountB.dylib"]
                ),
                .fileImage("/usr/lib/libD2CountB.dylib"),
            ]
        )
        let rootEdge = try XCTUnwrap(
            twoFiles.edges.first {
                $0.parentContentEvidenceID ==
                    twoFiles.root.contentEvidenceID.sha256
            }
        )
        let rootMember = try XCTUnwrap(
            twoFiles.members.first {
                $0.contentEvidenceID ==
                    rootEdge.resolvedContentEvidenceID
            }
        )
        let oneMemberReference = try Self.reference(
            for: twoFiles,
            memberFields: Self.memberFields([rootMember]),
            edgeFields: Self.edgeFields([rootEdge])
        )
        Self.assertFailure(
            fixture: twoFiles,
            expectationReference: oneMemberReference,
            currentAnchor:
                oneMemberReference.anchoredExpectation.trustAnchor,
            expected: .graphExpectation(.memberCount)
        )
        let oneEdgeReference = try Self.reference(
            for: twoFiles,
            memberFields: Self.memberFields(twoFiles.members),
            edgeFields: Self.edgeFields(Array(twoFiles.edges.dropLast()))
        )
        Self.assertFailure(
            fixture: twoFiles,
            expectationReference: oneEdgeReference,
            currentAnchor:
                oneEdgeReference.anchoredExpectation.trustAnchor,
            expected: .graphExpectation(.edgeCount)
        )

        let mixed = try Self.fixture(
            role: .git,
            rootTargets: ["/usr/lib/libD2CountFile.dylib"],
            specs: [
                .fileImage(
                    "/usr/lib/libD2CountFile.dylib",
                    targets: ["/usr/lib/libD2CountCache.dylib"]
                ),
                .sharedCache("/usr/lib/libD2CountCache.dylib"),
            ]
        )
        var bothFileFields = Self.memberFields(mixed.members)
        let sharedIndex = try XCTUnwrap(
            mixed.members.firstIndex { $0.storage == .sharedCache }
        )
        bothFileFields[sharedIndex] = Self.memberField(
            bothFileFields[sharedIndex],
            storage: .file,
            primaryCodeDirectoryBlobSHA256:
                String(repeating: "f", count: 64)
        )
        let declaredCountReference = try Self.reference(
            for: mixed,
            memberFields: bothFileFields
        )
        Self.assertFailure(
            fixture: mixed,
            expectationReference: declaredCountReference,
            currentAnchor:
                declaredCountReference.anchoredExpectation.trustAnchor,
            expected: .graphExpectation(.declaredFileImageMemberCount)
        )

        let canonical = try Self.fixture(role: .git)
        var changedUUID = Self.memberFields(canonical.members)
        changedUUID[0] = Self.memberField(
            changedUUID[0],
            machOUUID: String(repeating: "f", count: 32)
        )
        let rowReference = try Self.reference(
            for: canonical,
            memberFields: changedUUID
        )
        Self.assertFailure(
            fixture: canonical,
            expectationReference: rowReference,
            currentAnchor: rowReference.anchoredExpectation.trustAnchor,
            expected: .memberExpectation(
                index: 0,
                field: .machOUUID
            )
        )

        let canonicalEdge = try XCTUnwrap(canonical.edges.first)
        let longerOrdinal = RuntimeClosureExpectationEdgeFields(
            parentContentEvidenceID:
                canonicalEdge.parentContentEvidenceID,
            loadCommandOrdinal: 10,
            kind: canonicalEdge.kind,
            installName: canonicalEdge.installName,
            decodedInstallName: canonicalEdge.decodedInstallName,
            resolvedContentEvidenceID:
                canonicalEdge.resolvedContentEvidenceID
        )
        let longerSectionReference = try Self.reference(
            for: canonical,
            edgeFields: [longerOrdinal]
        )
        Self.assertFailure(
            fixture: canonical,
            expectationReference: longerSectionReference,
            currentAnchor:
                longerSectionReference.anchoredExpectation.trustAnchor,
            expected: .graphExpectation(.canonicalSectionRange)
        )

        let changedSingleDigitOrdinal: UInt64 =
            canonicalEdge.loadCommandOrdinal == 1 ? 2 : 1
        let sameLengthOrdinal = RuntimeClosureExpectationEdgeFields(
            parentContentEvidenceID:
                canonicalEdge.parentContentEvidenceID,
            loadCommandOrdinal: changedSingleDigitOrdinal,
            kind: canonicalEdge.kind,
            installName: canonicalEdge.installName,
            decodedInstallName: canonicalEdge.decodedInstallName,
            resolvedContentEvidenceID:
                canonicalEdge.resolvedContentEvidenceID
        )
        let changedSectionReference = try Self.reference(
            for: canonical,
            edgeFields: [sameLengthOrdinal]
        )
        Self.assertFailure(
            fixture: canonical,
            expectationReference: changedSectionReference,
            currentAnchor:
                changedSectionReference.anchoredExpectation.trustAnchor,
            expected: .graphExpectation(.canonicalSectionBytes)
        )
        XCTAssertEqual(Self.effects, .zero)
    }

    func testEveryConstructibleExpectationMemberFieldDriftsIndependently()
        throws
    {
        let fixture = try Self.fixture(role: .git)
        let member = try XCTUnwrap(
            Self.memberFields(fixture.members).first
        )
        let edge = try XCTUnwrap(
            Self.edgeFields(fixture.edges).first
        )

        let foreignContentEvidenceID = String(repeating: "e", count: 64)
        try Self.assertMemberExpectationFailure(
            fixture: fixture,
            member: Self.memberField(
                member,
                contentEvidenceID: foreignContentEvidenceID
            ),
            edge: Self.edgeField(
                edge,
                resolvedContentEvidenceID: foreignContentEvidenceID
            ),
            field: .contentEvidenceID
        )

        let longerName = FMAFileImageFixture.installName(
            "/usr/lib/libD2C2FixtureLong.dylib"
        )
        try Self.assertMemberExpectationFailure(
            fixture: fixture,
            member: Self.memberField(
                member,
                installName: longerName,
                decodedInstallName:
                    Data("/usr/lib/libD2C2FixtureLong.dylib".utf8)
            ),
            edge: Self.edgeField(
                edge,
                installName: longerName,
                decodedInstallName:
                    Data("/usr/lib/libD2C2FixtureLong.dylib".utf8)
            ),
            field: .installNameBytes
        )

        let sameLengthNameString =
            "/usr/lib/libD2C2Fixture.dylia"
        XCTAssertEqual(
            sameLengthNameString.utf8.count,
            Int(member.installName.bytes)
        )
        let sameLengthName = FMAFileImageFixture.installName(
            sameLengthNameString
        )
        try Self.assertMemberExpectationFailure(
            fixture: fixture,
            member: Self.memberField(
                member,
                installName: sameLengthName,
                decodedInstallName: Data(sameLengthNameString.utf8)
            ),
            edge: Self.edgeField(
                edge,
                installName: sameLengthName,
                decodedInstallName: Data(sameLengthNameString.utf8)
            ),
            field: .installNameBase64URL
        )

        try Self.assertMemberExpectationFailure(
            fixture: fixture,
            member: Self.memberField(
                member,
                machOUUID: String(repeating: "f", count: 32)
            ),
            edge: edge,
            field: .machOUUID
        )
        try Self.assertMemberExpectationFailure(
            fixture: fixture,
            member: Self.memberField(
                member,
                primaryCodeDirectoryBlobSHA256:
                    String(repeating: "f", count: 64)
            ),
            edge: edge,
            field: .primaryCodeDirectoryBlobSHA256
        )
        try Self.assertMemberExpectationFailure(
            fixture: fixture,
            member: Self.memberField(
                member,
                loadCommandsSHA256: String(repeating: "e", count: 64)
            ),
            edge: edge,
            field: .loadCommandsSHA256
        )
        XCTAssertEqual(Self.effects, .zero)
    }

    func testGenericFileRefusesThroughFMBeforeLocalCountWork() throws {
        let fixture = try Self.fixture(role: .git)
        let fileEvidence = try ExecutableContentIdentityVerifier.derive(
            artifactRole: .fileImage,
            comparison: fixture.root.comparison
        )
        let member = try SyntheticRuntimeClosureRecordSchemaVerifier.member(
            index: 0,
            source: .file(fileEvidence),
            installName: FMAFileImageFixture.installName(Self.memberName)
        )
        let edge = try SyntheticRuntimeClosureRecordSchemaVerifier.edge(
            index: 0,
            parent: .root(fixture.root),
            loadCommandOrdinal:
                try XCTUnwrap(fixture.rootInventory.entries.first)
                .loadCommandOrdinal,
            kind: .load,
            installName: member.installName,
            resolved: member
        )
        let collection = try
            SyntheticRuntimeClosureRecordCollectionVerifier.derive(
                members: [member],
                edges: [edge]
            )

        Self.assertFailure(
            fixture: fixture,
            members: [member],
            edges: [edge],
            collection: collection,
            memberInventories: [],
            expected: .graph(
                .unsupportedFileMemberIdentity(memberIndex: 0)
            )
        )
        XCTAssertEqual(Self.effects, .zero)
    }

    func testAllSharedCacheGraphRefusesAtD2C1() {
        XCTAssertThrowsError(
            try Self.fixture(
                role: .git,
                rootTargets: [Self.memberName],
                specs: [.sharedCache(Self.memberName)]
            )
        ) {
            XCTAssertEqual(
                $0 as? FileImageRuntimeClosureExpectationFailure,
                .declaredFileImageMemberCount
            )
        }
        XCTAssertEqual(Self.effects, .zero)
    }

    func testSharedCacheEvidenceMustMatchTheAnchoredCacheSet() throws {
        let specs: [MemberSpec] = [
            .fileImage(
                "/usr/lib/libD2CacheFile.dylib",
                targets: ["/usr/lib/libD2CacheMember.dylib"]
            ),
            .sharedCache("/usr/lib/libD2CacheMember.dylib"),
        ]
        let actual = try Self.fixture(
            role: .git,
            rootTargets: ["/usr/lib/libD2CacheFile.dylib"],
            specs: specs,
            cacheSeed: "1"
        )
        let foreign = try Self.fixture(
            role: .git,
            rootTargets: ["/usr/lib/libD2CacheFile.dylib"],
            specs: specs,
            cacheSeed: "2"
        )
        XCTAssertEqual(
            actual.members.map(\.storage),
            foreign.members.map(\.storage)
        )
        let foreignReference = try Self.reference(
            for: actual,
            cacheSet: foreign.cacheSet,
            memberFields: Self.memberFields(foreign.members),
            edgeFields: Self.edgeFields(foreign.edges)
        )
        let sharedIndex = try XCTUnwrap(
            actual.members.firstIndex { $0.storage == .sharedCache }
        )

        Self.assertFailure(
            fixture: actual,
            expectationReference: foreignReference,
            currentAnchor:
                foreignReference.anchoredExpectation.trustAnchor,
            expected: .memberCacheSetMismatch(index: sharedIndex)
        )
        XCTAssertEqual(Self.effects, .zero)
    }

    func testNonzeroDataIndexAndCallerMutationPreserveCompactResult()
        throws
    {
        var fixture = try Self.fixture(role: .git)
        var backing = Data([0])
        backing.append(fixture.expectationFile.bytes)
        backing.append(0)
        let slice = backing[1..<(1 + fixture.expectationFile.bytes.count)]
        XCTAssertEqual(slice.startIndex, 1)
        let slicedFile = Self.admittedFile(slice)
        let slicedAnchor = Self.anchor(
            for: slicedFile,
            role: fixture.role
        )
        let slicedExpectation = try RuntimeClosureExpectationVerifier
            .anchor(
                expectationFile: slicedFile,
                trustAnchor: slicedAnchor
            )
        let slicedReference = try
            FileImageRuntimeClosureExpectationVerifier.reference(
                anchoredExpectation: slicedExpectation,
                currentExpectationAnchor: slicedAnchor
            )
        let result = try Self.compare(
            fixture,
            expectationReference: slicedReference,
            currentAnchor: slicedAnchor
        )
        let retained = result

        backing[slice.startIndex] = 0xff
        fixture.members.removeAll()
        fixture.edges.removeAll()
        fixture.memberInventories.removeAll()

        XCTAssertEqual(result, retained)
        XCTAssertEqual(result.memberCount, 1)
        XCTAssertEqual(result.fileImageMemberCount, 1)
        XCTAssertEqual(Self.effects, .zero)
    }
}

private extension FileImageRuntimeClosureContentIdentityTests {
    struct Effects: Equatable {
        var processSpawns = 0
        var fileSystemMutations = 0
        var networkOperations = 0
        var packConsumptions = 0
        var objectDatabaseImports = 0
        var buildOperations = 0
        var modelLoads = 0
        var outputReservations = 0
        var publications = 0

        static let zero = Self()
    }

    struct MemberSpec {
        enum Kind {
            case sharedCache
            case fileImage
        }

        let name: String
        let kind: Kind
        let targets: [String]

        static func sharedCache(
            _ name: String,
            targets: [String] = []
        ) -> Self {
            Self(name: name, kind: .sharedCache, targets: targets)
        }

        static func fileImage(
            _ name: String,
            targets: [String] = []
        ) -> Self {
            Self(name: name, kind: .fileImage, targets: targets)
        }
    }

    struct MemberInput {
        let name: String
        let source: SyntheticRuntimeClosureMemberSource
        let inventory:
            SyntheticAcceptedDependencyCommandInventoryComparison
        let contentEvidenceID: String
    }

    struct EdgeInput {
        let parent: SyntheticRuntimeClosureEdgeParent
        let parentID: String
        let ordinal: UInt64
        let kind: SyntheticRuntimeClosureEdgeKind
        let resolved: SyntheticRuntimeClosureMemberRecordComparison
    }

    struct Fixture {
        let role: RuntimeClosureExpectationArtifactRole
        let root: ExecutableContentIdentityEvidence
        let loader: DynamicLoaderContentIdentityEvidence
        let cacheSet: SyntheticSharedCacheSetIdentityEvidence
        var members: [SyntheticRuntimeClosureMemberRecordComparison]
        var edges: [SyntheticRuntimeClosureEdgeRecordComparison]
        let collection: SyntheticRuntimeClosureRecordCollectionComparison
        let rootInventory:
            SyntheticAcceptedDependencyCommandInventoryComparison
        var memberInventories:
            [SyntheticAcceptedDependencyCommandInventoryComparison]
        let graph: SyntheticFileImageRuntimeClosureGraphComparison
        let expectationFile: AdmittedFile
        let anchor: RuntimeClosureExpectationTrustAnchor
        let expectation: AnchoredRuntimeClosureExpectationDocument
        let reference: FileImageRuntimeClosureExpectationReference
    }

    static let effects = Effects()
    static let memberName = "/usr/lib/libD2C2Fixture.dylib"

    static func fixture(
        role: RuntimeClosureExpectationArtifactRole,
        rootTargets: [String]? = nil,
        specs: [MemberSpec]? = nil,
        cacheSeed: Character = "1"
    ) throws -> Fixture {
        let resolvedSpecs = specs ?? [.fileImage(memberName)]
        let resolvedRootTargets = rootTargets ?? [resolvedSpecs[0].name]
        let cacheSet = try SyntheticSharedCacheSetIdentityVerifier.derive(
            records: [
                SyntheticSharedCacheFileRecord(
                    suffixBytes: 0,
                    suffixBase64URL: "",
                    fileSHA256: String(
                        repeating: cacheSeed,
                        count: 64
                    ),
                    fileBytes: 4_096,
                    headerUUID: String(
                        repeating: cacheSeed,
                        count: 32
                    )
                )
            ]
        )
        var inputs: [MemberInput] = []
        for (index, spec) in resolvedSpecs.enumerated() {
            let commands = spec.targets.map {
                FMAFileImageFixture.dylibCommand(
                    command: FMAFileImageFixture.lcLoadDylib,
                    name: $0
                )
            }
            switch spec.kind {
            case .fileImage:
                let snapshot = try FMAFileImageFixture.snapshot(
                    identityName: Data(spec.name.utf8),
                    commands: commands
                )
                let inventory = try
                    SyntheticAcceptedDependencyCommandInventoryVerifier
                    .fileImageMember(snapshot)
                inputs.append(
                    MemberInput(
                        name: spec.name,
                        source: .fileImage(snapshot),
                        inventory: inventory,
                        contentEvidenceID:
                            snapshot.fileImageEvidence.contentEvidenceID
                            .sha256
                    )
                )
            case .sharedCache:
                let snapshot = try sharedCacheSnapshot(
                    name: spec.name,
                    commands: commands,
                    cacheSet: cacheSet,
                    uuidSeed: UInt8(truncatingIfNeeded: 0x70 + index)
                )
                let inventory = try
                    SyntheticAcceptedDependencyCommandInventoryVerifier
                    .sharedCacheMember(snapshot)
                inputs.append(
                    MemberInput(
                        name: spec.name,
                        source: .sharedCache(snapshot.imageEvidence),
                        inventory: inventory,
                        contentEvidenceID:
                            snapshot.imageEvidence.contentEvidenceID.sha256
                    )
                )
            }
        }
        inputs.sort {
            $0.contentEvidenceID.utf8.lexicographicallyPrecedes(
                $1.contentEvidenceID.utf8
            )
        }
        let members = try inputs.enumerated().map { index, input in
            try SyntheticRuntimeClosureRecordSchemaVerifier.member(
                index: index,
                source: input.source,
                installName: FMAFileImageFixture.installName(input.name)
            )
        }
        let inventoryByID = Dictionary(
            uniqueKeysWithValues: inputs.map {
                ($0.contentEvidenceID, $0.inventory)
            }
        )
        let memberInventories = members.map {
            inventoryByID[$0.contentEvidenceID]!
        }
        let memberByName = Dictionary(
            uniqueKeysWithValues: members.map {
                (
                    String(
                        decoding: $0.decodedInstallName,
                        as: UTF8.self
                    ),
                    $0
                )
            }
        )
        let root = try SyntheticRuntimeClosureGraphTests.fmbRootEvidence(
            role: executableRole(role),
            loadCommands: resolvedRootTargets.map {
                FMAFileImageFixture.dylibCommand(
                    command: FMAFileImageFixture.lcLoadDylib,
                    name: $0
                )
            }
        )
        let rootInventory = try
            SyntheticAcceptedDependencyCommandInventoryVerifier.root(root)
        var edgeInputs: [EdgeInput] = []
        for entry in rootInventory.entries {
            let resolved = try XCTUnwrap(
                memberByName[
                    String(
                        decoding: entry.decodedInstallName,
                        as: UTF8.self
                    )
                ]
            )
            edgeInputs.append(
                EdgeInput(
                    parent: .root(root),
                    parentID: root.contentEvidenceID.sha256,
                    ordinal: entry.loadCommandOrdinal,
                    kind: entry.kind,
                    resolved: resolved
                )
            )
        }
        for (member, inventory) in zip(members, memberInventories) {
            for entry in inventory.entries {
                let resolved = try XCTUnwrap(
                    memberByName[
                        String(
                            decoding: entry.decodedInstallName,
                            as: UTF8.self
                        )
                    ]
                )
                edgeInputs.append(
                    EdgeInput(
                        parent: .member(member),
                        parentID: member.contentEvidenceID,
                        ordinal: entry.loadCommandOrdinal,
                        kind: entry.kind,
                        resolved: resolved
                    )
                )
            }
        }
        edgeInputs.sort {
            if $0.parentID != $1.parentID {
                return $0.parentID.utf8.lexicographicallyPrecedes(
                    $1.parentID.utf8
                )
            }
            return $0.ordinal < $1.ordinal
        }
        let edges = try edgeInputs.enumerated().map { index, input in
            try SyntheticRuntimeClosureRecordSchemaVerifier.edge(
                index: index,
                parent: input.parent,
                loadCommandOrdinal: input.ordinal,
                kind: input.kind,
                installName: input.resolved.installName,
                resolved: input.resolved
            )
        }
        let collection = try
            SyntheticRuntimeClosureRecordCollectionVerifier.derive(
                members: members,
                edges: edges
            )
        let graph = try
            SyntheticFileImageRuntimeClosureGraphVerifier.compare(
                root: root,
                members: members,
                edges: edges,
                collection: collection,
                rootInventory: rootInventory,
                memberInventories: memberInventories
            )
        let loader = try dynamicLoaderEvidence()
        let documentBytes = renderExpectation(
            role: role,
            rootID: root.contentEvidenceID.sha256,
            loaderID: loader.contentEvidenceID.sha256,
            cacheSet: cacheSet,
            members: members,
            edges: edges
        )
        let expectationFile = admittedFile(documentBytes)
        let anchor = RuntimeClosureExpectationTrustAnchor(
            expectedCurrentDocumentSHA256: expectationFile.sha256,
            expectedCurrentDocumentBytes: UInt64(documentBytes.count),
            minimumEvidenceGeneration: 9,
            verificationUnixSeconds: 2_000_000_000,
            expectedArtifactRole: role
        )
        let expectation = try RuntimeClosureExpectationVerifier.anchor(
            expectationFile: expectationFile,
            trustAnchor: anchor
        )
        let reference = try
            FileImageRuntimeClosureExpectationVerifier.reference(
                anchoredExpectation: expectation,
                currentExpectationAnchor: anchor
            )
        return Fixture(
            role: role,
            root: root,
            loader: loader,
            cacheSet: cacheSet,
            members: members,
            edges: edges,
            collection: collection,
            rootInventory: rootInventory,
            memberInventories: memberInventories,
            graph: graph,
            expectationFile: expectationFile,
            anchor: anchor,
            expectation: expectation,
            reference: reference
        )
    }

    static func executableRole(
        _ role: RuntimeClosureExpectationArtifactRole
    ) -> ExecutableContentArtifactRole {
        switch role {
        case .git:
            .git
        case .selfGuard:
            .selfGuard
        }
    }

    static func allFileImageChain(count: Int) throws -> Fixture {
        let names = (0..<count).map {
            "/usr/lib/libD2C2Chain\(fourDigit($0)).dylib"
        }
        let specs = names.enumerated().map { index, name in
            MemberSpec.fileImage(
                name,
                targets: index + 1 < names.count
                    ? [names[index + 1]]
                    : []
            )
        }
        return try fixture(
            role: .git,
            rootTargets: [names[0]],
            specs: specs
        )
    }

    static func compare(
        _ fixture: Fixture,
        expectationReference:
            FileImageRuntimeClosureExpectationReference? = nil,
        currentAnchor: RuntimeClosureExpectationTrustAnchor? = nil
    ) throws -> FileImageRuntimeClosureContentExpectationComparison {
        try FileImageRuntimeClosureContentIdentityVerifier.compare(
            expectationReference:
                expectationReference ?? fixture.reference,
            currentExpectationAnchor:
                currentAnchor ?? fixture.anchor,
            rootExecutableContentEvidence: fixture.root,
            dynamicLoaderContentEvidence: fixture.loader,
            members: fixture.members,
            edges: fixture.edges,
            collection: fixture.collection,
            rootInventory: fixture.rootInventory,
            memberInventories: fixture.memberInventories,
            graphComparison: fixture.graph
        )
    }

    static func assertFailure(
        fixture: Fixture,
        expectationReference:
            FileImageRuntimeClosureExpectationReference? = nil,
        currentAnchor: RuntimeClosureExpectationTrustAnchor? = nil,
        root: ExecutableContentIdentityEvidence? = nil,
        members: [SyntheticRuntimeClosureMemberRecordComparison]? = nil,
        edges: [SyntheticRuntimeClosureEdgeRecordComparison]? = nil,
        collection:
            SyntheticRuntimeClosureRecordCollectionComparison? = nil,
        memberInventories:
            [SyntheticAcceptedDependencyCommandInventoryComparison]? = nil,
        graphComparison:
            SyntheticFileImageRuntimeClosureGraphComparison? = nil,
        expected: FileImageRuntimeClosureContentIdentityFailure,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(
            try FileImageRuntimeClosureContentIdentityVerifier.compare(
                expectationReference:
                    expectationReference ?? fixture.reference,
                currentExpectationAnchor:
                    currentAnchor ?? fixture.anchor,
                rootExecutableContentEvidence: root ?? fixture.root,
                dynamicLoaderContentEvidence: fixture.loader,
                members: members ?? fixture.members,
                edges: edges ?? fixture.edges,
                collection: collection ?? fixture.collection,
                rootInventory: fixture.rootInventory,
                memberInventories:
                    memberInventories ?? fixture.memberInventories,
                graphComparison: graphComparison ?? fixture.graph
            ),
            file: file,
            line: line
        ) {
            XCTAssertEqual(
                $0 as? FileImageRuntimeClosureContentIdentityFailure,
                expected,
                file: file,
                line: line
            )
        }
    }

    static func assertResult(
        _ result: FileImageRuntimeClosureContentExpectationComparison,
        fixture: Fixture,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        XCTAssertEqual(result.artifactRole, fixture.role, file: file, line: line)
        XCTAssertEqual(
            result.manifestSHA256,
            fixture.expectation.documentSHA256,
            file: file,
            line: line
        )
        XCTAssertEqual(
            result.manifestBytes,
            fixture.expectation.documentBytes,
            file: file,
            line: line
        )
        XCTAssertEqual(
            result.rootExecutableContentEvidenceID,
            fixture.root.contentEvidenceID.sha256,
            file: file,
            line: line
        )
        XCTAssertEqual(
            result.dynamicLoaderContentEvidenceID,
            fixture.loader.contentEvidenceID.sha256,
            file: file,
            line: line
        )
        XCTAssertEqual(
            result.sharedCacheSetID,
            fixture.cacheSet.sharedCacheSetID.sha256,
            file: file,
            line: line
        )
        XCTAssertEqual(result.memberCount, fixture.members.count, file: file, line: line)
        XCTAssertEqual(result.edgeCount, fixture.edges.count, file: file, line: line)
        XCTAssertEqual(
            result.fileImageMemberCount,
            fixture.reference.declaredFileImageMemberCount,
            file: file,
            line: line
        )
        XCTAssertTrue(result.provesExpectationAnchorMatch, file: file, line: line)
        XCTAssertTrue(result.provesManifestContentMatch, file: file, line: line)
        XCTAssertTrue(result.provesDeclaredStaticGraphMatch, file: file, line: line)
        XCTAssertTrue(result.provesSealedFileImageContinuity, file: file, line: line)
        XCTAssertFalse(result.isCompleteRuntimeClosure, file: file, line: line)
        XCTAssertFalse(result.provesRuntimeLaunchability, file: file, line: line)
        XCTAssertEqual(
            result.runtimeResolutionOutcome,
            "unproved-static-comparison-only",
            file: file,
            line: line
        )
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

        let mirror = Mirror(reflecting: result)
        XCTAssertEqual(
            mirror.children.compactMap(\.label),
            [
                "artifactRole",
                "manifestSHA256",
                "manifestBytes",
                "rootExecutableContentEvidenceID",
                "dynamicLoaderContentEvidenceID",
                "sharedCacheSetID",
                "memberCount",
                "edgeCount",
                "fileImageMemberCount",
                "contentEvidenceID",
                "provesExpectationAnchorMatch",
                "provesManifestContentMatch",
                "provesDeclaredStaticGraphMatch",
                "provesSealedFileImageContinuity",
                "isCompleteRuntimeClosure",
                "provesRuntimeLaunchability",
                "runtimeResolutionOutcome",
                "runtimeDecision",
                "canExecute",
                "canSpawn",
                "canAccessNetwork",
                "canConsumePack",
                "canMutateFileSystem",
                "canImportGitObjects",
                "canBuild",
                "canLoadModel",
                "canReserveOutput",
                "canPublish",
                "constructionSeal",
            ],
            file: file,
            line: line
        )
        let forbiddenTypeFragments = [
            "FileImageRuntimeClosureExpectationReference",
            "AnchoredRuntimeClosureExpectationDocument",
            "RuntimeClosureExpectationTrustAnchor",
            "SyntheticFileImageRuntimeClosureGraphComparison",
            "SyntheticRuntimeClosureRecordCollectionComparison",
            "SyntheticRuntimeClosureMemberRecordComparison",
            "SyntheticRuntimeClosureEdgeRecordComparison",
            "SyntheticAcceptedDependencyCommandInventoryComparison",
            "Foundation.Data",
            "Range<",
        ]
        for child in mirror.children {
            let typeName = String(reflecting: type(of: child.value))
            for forbidden in forbiddenTypeFragments {
                XCTAssertFalse(
                    typeName.contains(forbidden),
                    "retained forbidden type \(typeName)",
                    file: file,
                    line: line
                )
            }
        }
    }

    static func reference(
        for fixture: Fixture,
        rootID: String? = nil,
        loaderID: String? = nil,
        cacheSet: SyntheticSharedCacheSetIdentityEvidence? = nil,
        memberFields: [RuntimeClosureExpectationMemberFields]? = nil,
        edgeFields: [RuntimeClosureExpectationEdgeFields]? = nil
    ) throws -> FileImageRuntimeClosureExpectationReference {
        let resolvedCacheSet = cacheSet ?? fixture.cacheSet
        let resolvedRootID =
            rootID ?? fixture.root.contentEvidenceID.sha256
        var resolvedEdgeFields =
            edgeFields ?? Self.edgeFields(fixture.edges)
        if resolvedRootID != fixture.root.contentEvidenceID.sha256 {
            resolvedEdgeFields = resolvedEdgeFields.map { edge in
                RuntimeClosureExpectationEdgeFields(
                    parentContentEvidenceID:
                        edge.parentContentEvidenceID ==
                            fixture.root.contentEvidenceID.sha256
                        ? resolvedRootID
                        : edge.parentContentEvidenceID,
                    loadCommandOrdinal: edge.loadCommandOrdinal,
                    kind: edge.kind,
                    installName: edge.installName,
                    decodedInstallName: edge.decodedInstallName,
                    resolvedContentEvidenceID:
                        edge.resolvedContentEvidenceID
                )
            }
        }
        let bytes = renderExpectation(
            role: fixture.role,
            rootID: resolvedRootID,
            loaderID:
                loaderID ?? fixture.loader.contentEvidenceID.sha256,
            cacheSet: resolvedCacheSet,
            memberFields:
                memberFields ?? Self.memberFields(fixture.members),
            edgeFields: resolvedEdgeFields
        )
        let file = admittedFile(bytes)
        let anchor = Self.anchor(for: file, role: fixture.role)
        let expectation = try RuntimeClosureExpectationVerifier.anchor(
            expectationFile: file,
            trustAnchor: anchor
        )
        return try FileImageRuntimeClosureExpectationVerifier.reference(
            anchoredExpectation: expectation,
            currentExpectationAnchor: anchor
        )
    }

    static func memberFields(
        _ members: [SyntheticRuntimeClosureMemberRecordComparison]
    ) -> [RuntimeClosureExpectationMemberFields] {
        members.map {
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
    }

    static func edgeFields(
        _ edges: [SyntheticRuntimeClosureEdgeRecordComparison]
    ) -> [RuntimeClosureExpectationEdgeFields] {
        edges.map {
            RuntimeClosureExpectationEdgeFields(
                parentContentEvidenceID: $0.parentContentEvidenceID,
                loadCommandOrdinal: $0.loadCommandOrdinal,
                kind: $0.kind,
                installName: $0.installName,
                decodedInstallName: $0.decodedInstallName,
                resolvedContentEvidenceID:
                    $0.resolvedContentEvidenceID
            )
        }
    }

    static func memberField(
        _ value: RuntimeClosureExpectationMemberFields,
        contentEvidenceID: String? = nil,
        storage: SyntheticRuntimeClosureMemberStorage? = nil,
        installName: SyntheticRuntimeClosureInstallName? = nil,
        decodedInstallName: Data? = nil,
        machOUUID: String? = nil,
        primaryCodeDirectoryBlobSHA256: String? = nil,
        loadCommandsSHA256: String? = nil
    ) -> RuntimeClosureExpectationMemberFields {
        RuntimeClosureExpectationMemberFields(
            contentEvidenceID:
                contentEvidenceID ?? value.contentEvidenceID,
            storage: storage ?? value.storage,
            installName: installName ?? value.installName,
            decodedInstallName:
                decodedInstallName ?? value.decodedInstallName,
            machOUUID: machOUUID ?? value.machOUUID,
            primaryCodeDirectoryBlobSHA256:
                primaryCodeDirectoryBlobSHA256 ??
                value.primaryCodeDirectoryBlobSHA256,
            loadCommandsSHA256:
                loadCommandsSHA256 ?? value.loadCommandsSHA256
        )
    }

    static func edgeField(
        _ value: RuntimeClosureExpectationEdgeFields,
        installName: SyntheticRuntimeClosureInstallName? = nil,
        decodedInstallName: Data? = nil,
        resolvedContentEvidenceID: String? = nil
    ) -> RuntimeClosureExpectationEdgeFields {
        RuntimeClosureExpectationEdgeFields(
            parentContentEvidenceID: value.parentContentEvidenceID,
            loadCommandOrdinal: value.loadCommandOrdinal,
            kind: value.kind,
            installName: installName ?? value.installName,
            decodedInstallName:
                decodedInstallName ?? value.decodedInstallName,
            resolvedContentEvidenceID:
                resolvedContentEvidenceID ??
                value.resolvedContentEvidenceID
        )
    }

    static func assertMemberExpectationFailure(
        fixture: Fixture,
        member: RuntimeClosureExpectationMemberFields,
        edge: RuntimeClosureExpectationEdgeFields,
        field: RuntimeClosureExpectationMemberRecordField,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let reference = try Self.reference(
            for: fixture,
            memberFields: [member],
            edgeFields: [edge]
        )
        Self.assertFailure(
            fixture: fixture,
            expectationReference: reference,
            currentAnchor: reference.anchoredExpectation.trustAnchor,
            expected: .memberExpectation(index: 0, field: field),
            file: file,
            line: line
        )
    }

    static func independentPreimage(
        role: RuntimeClosureExpectationArtifactRole
    ) -> Data {
        Data(
            ([
                "fast-mlx-proof-control-file-image-runtime-closure-content-evidence-id-v1",
                "artifact_role=\(role.rawValue)",
                "manifest_sha256=" + String(repeating: "1", count: 64),
                "manifest_bytes=123",
                "root_executable_content_evidence_id=" +
                    String(repeating: "2", count: 64),
                "dynamic_loader_content_evidence_id=" +
                    String(repeating: "3", count: 64),
                "shared_cache_set_id=" +
                    String(repeating: "4", count: 64),
                "file_image_member_count=1",
            ].joined(separator: "\n") + "\n").utf8
        )
    }

    static func expectedPreimage(_ fixture: Fixture) -> Data {
        Data(
            ([
                "fast-mlx-proof-control-file-image-runtime-closure-content-evidence-id-v1",
                "artifact_role=\(fixture.role.rawValue)",
                "manifest_sha256=\(fixture.expectation.documentSHA256)",
                "manifest_bytes=\(fixture.expectation.documentBytes)",
                "root_executable_content_evidence_id=" +
                    fixture.root.contentEvidenceID.sha256,
                "dynamic_loader_content_evidence_id=" +
                    fixture.loader.contentEvidenceID.sha256,
                "shared_cache_set_id=" +
                    fixture.cacheSet.sharedCacheSetID.sha256,
                "file_image_member_count=" +
                    String(
                        fixture.reference.declaredFileImageMemberCount
                    ),
            ].joined(separator: "\n") + "\n").utf8
        )
    }

    static func renderExpectation(
        role: RuntimeClosureExpectationArtifactRole,
        rootID: String,
        loaderID: String,
        cacheSet: SyntheticSharedCacheSetIdentityEvidence,
        members: [SyntheticRuntimeClosureMemberRecordComparison],
        edges: [SyntheticRuntimeClosureEdgeRecordComparison]
    ) -> Data {
        renderExpectation(
            role: role,
            rootID: rootID,
            loaderID: loaderID,
            cacheSet: cacheSet,
            memberFields: memberFields(members),
            edgeFields: edgeFields(edges)
        )
    }

    static func renderExpectation(
        role: RuntimeClosureExpectationArtifactRole,
        rootID: String,
        loaderID: String,
        cacheSet: SyntheticSharedCacheSetIdentityEvidence,
        memberFields: [RuntimeClosureExpectationMemberFields],
        edgeFields: [RuntimeClosureExpectationEdgeFields]
    ) -> Data {
        var lines = [
            RuntimeClosureExpectationVerifier.documentDomain,
            "subject=absorbed-mla-source-import-runtime-closure-identity",
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
                "\(prefix)suffix_base64url=\(record.suffixBase64URL)"
            )
            lines.append("\(prefix)sha256=\(record.fileSHA256)")
            lines.append("\(prefix)bytes=\(record.fileBytes)")
            lines.append("\(prefix)header_uuid=\(record.headerUUID)")
        }
        lines.append(
            "shared_cache_set_id=\(cacheSet.sharedCacheSetID.sha256)"
        )
        lines.append("member_count=\(memberFields.count)")
        for (index, member) in memberFields.enumerated() {
            let prefix = "member_\(fourDigit(index))_"
            lines.append(
                "\(prefix)content_evidence_id=\(member.contentEvidenceID)"
            )
            lines.append("\(prefix)storage=\(member.storage.rawValue)")
            lines.append(
                "\(prefix)install_name_bytes=\(member.installName.bytes)"
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
        lines.append("edge_count=\(edgeFields.count)")
        for (index, edge) in edgeFields.enumerated() {
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
                "\(prefix)install_name_bytes=\(edge.installName.bytes)"
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
            "runtime_resolution_outcome=unproved-static-comparison-only"
        )
        lines.append("runtime_authority=none")
        return Data((lines.joined(separator: "\n") + "\n").utf8)
    }

    static func admittedFile(_ bytes: Data) -> AdmittedFile {
        AdmittedFile(
            bytes: bytes,
            sha256: sha256Hex(bytes),
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

    static func anchor(
        for file: AdmittedFile,
        role: RuntimeClosureExpectationArtifactRole,
        sha256: String? = nil,
        bytes: UInt64? = nil,
        minimumGeneration: UInt64 = 9,
        verificationUnixSeconds: UInt64 = 2_000_000_000
    ) -> RuntimeClosureExpectationTrustAnchor {
        RuntimeClosureExpectationTrustAnchor(
            expectedCurrentDocumentSHA256: sha256 ?? file.sha256,
            expectedCurrentDocumentBytes:
                bytes ?? UInt64(file.bytes.count),
            minimumEvidenceGeneration: minimumGeneration,
            verificationUnixSeconds: verificationUnixSeconds,
            expectedArtifactRole: role
        )
    }

    static func sharedCacheSnapshot(
        name: String,
        commands: [Data],
        cacheSet: SyntheticSharedCacheSetIdentityEvidence,
        uuidSeed: UInt8
    ) throws -> SyntheticSharedCacheImageLoadCommandSnapshot {
        let installName = FMAFileImageFixture.installName(name)
        let commandBytes = commands.reduce(into: Data()) {
            $0.append($1)
        }
        let evidence = try
            SyntheticSharedCacheImageContentIdentityVerifier.derive(
                cacheSetEvidence: cacheSet,
                facts: SyntheticSharedCacheImageContentFacts(
                    installNameBytes: installName.bytes,
                    installNameBase64URL: installName.base64URL,
                    machOUUID: fixedHex(
                        seed: uuidSeed,
                        count: 16
                    ),
                    primaryCodeDirectory: .absent,
                    loadCommandsSHA256: sha256Hex(commandBytes)
                )
            )
        return try SyntheticSharedCacheImageLoadCommandSnapshotVerifier
            .derive(
                imageEvidence: evidence,
                loadCommandBytes: commandBytes
            )
    }

    static func dynamicLoaderEvidence()
        throws -> DynamicLoaderContentIdentityEvidence
    {
        let signature = superBlob(
            entries: [
                (
                    0,
                    codeDirectory(
                        signingIdentifier:
                            Data("com.example.d2c2.dyld".utf8)
                    )
                )
            ]
        )
        let comparison = try
            SyntheticDynamicLoaderMachOIdentityParser.parse(
                dynamicLoaderMachO(signatureRegion: signature)
            )
        return try DynamicLoaderContentIdentityVerifier.derive(
            comparison: comparison
        )
    }

    static func dynamicLoaderMachO(signatureRegion: Data) -> Data {
        var commands = uuidCommand(seed: 0x20)
        commands.append(dynamicLoaderIdentityCommand())
        let signatureOffset = commands.count
        appendUInt32LE(&commands, 0x1d)
        appendUInt32LE(&commands, 16)
        appendUInt32LE(&commands, 0)
        appendUInt32LE(&commands, UInt32(signatureRegion.count))
        writeUInt32LE(
            &commands,
            at: signatureOffset + 8,
            value: UInt32(32 + commands.count)
        )
        return machOHeader(
            fileType: 7,
            commandCount: 3,
            commands: commands,
            signatureRegion: signatureRegion
        )
    }

    static func machOHeader(
        fileType: UInt32,
        commandCount: Int,
        commands: Data,
        signatureRegion: Data
    ) -> Data {
        var result = Data()
        appendUInt32LE(&result, 0xfeed_facf)
        appendUInt32LE(&result, 0x0100_000c)
        appendUInt32LE(&result, 2)
        appendUInt32LE(&result, fileType)
        appendUInt32LE(&result, UInt32(commandCount))
        appendUInt32LE(&result, UInt32(commands.count))
        appendUInt32LE(&result, 0x0020_0085)
        appendUInt32LE(&result, 0)
        result.append(commands)
        result.append(signatureRegion)
        return result
    }

    static func uuidCommand(seed: UInt8) -> Data {
        var result = Data()
        appendUInt32LE(&result, 0x1b)
        appendUInt32LE(&result, 24)
        result.append(contentsOf: (0..<16).map { seed &+ UInt8($0) })
        return result
    }

    static func dynamicLoaderIdentityCommand() -> Data {
        var result = Data()
        appendUInt32LE(&result, 0x0f)
        appendUInt32LE(&result, 0)
        appendUInt32LE(&result, 12)
        result.append(Data("/usr/lib/dyld".utf8))
        result.append(0)
        while !result.count.isMultiple(of: 8) {
            result.append(0)
        }
        writeUInt32LE(&result, at: 4, value: UInt32(result.count))
        return result
    }

    static func superBlob(entries: [(UInt32, Data)]) -> Data {
        var nextOffset = 12 + entries.count * 8
        var result = Data()
        appendUInt32BE(&result, 0xfade_0cc0)
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
        writeUInt32BE(&result, at: 0, value: 0xfade_0c02)
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

    static func fourDigit(_ value: Int) -> String {
        String(format: "%04d", value)
    }

    static func fixedHex(seed: UInt8, count: Int) -> String {
        (0..<count).map {
            String(format: "%02x", seed &+ UInt8($0))
        }.joined()
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
