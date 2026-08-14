import XCTest
@testable import ProofControl

final class SyntheticFileImageRuntimeClosureGraphTests: XCTestCase {
    func testVersionedFileImageGraphsPreserveAllBoundedTopologies()
        throws
    {
        let effects = Effects.zero
        let cases: [(
            rootTargets: [String],
            specs: [MemberSpec],
            expectedFileImages: Int
        )] = [
            (
                ["/usr/lib/libD2A.dylib"],
                [.fileImage("/usr/lib/libD2A.dylib")],
                1
            ),
            (
                ["/usr/lib/libD2A.dylib"],
                [
                    .fileImage(
                        "/usr/lib/libD2A.dylib",
                        targets: ["/usr/lib/libCache.dylib"]
                    ),
                    .sharedCache("/usr/lib/libCache.dylib"),
                ],
                1
            ),
            (
                ["/usr/lib/libCache.dylib"],
                [
                    .sharedCache(
                        "/usr/lib/libCache.dylib",
                        targets: ["/usr/lib/libD2A.dylib"]
                    ),
                    .fileImage("/usr/lib/libD2A.dylib"),
                ],
                1
            ),
            (
                ["/usr/lib/libD2A.dylib"],
                [
                    .fileImage(
                        "/usr/lib/libD2A.dylib",
                        targets: ["/usr/lib/libD2B.dylib"]
                    ),
                    .fileImage("/usr/lib/libD2B.dylib"),
                ],
                2
            ),
            (
                ["/usr/lib/libD2A.dylib"],
                [
                    .fileImage(
                        "/usr/lib/libD2A.dylib",
                        targets: ["/usr/lib/libD2A.dylib"]
                    ),
                ],
                1
            ),
            (
                ["/usr/lib/libD2A.dylib"],
                [
                    .fileImage(
                        "/usr/lib/libD2A.dylib",
                        targets: ["/usr/lib/libD2B.dylib"]
                    ),
                    .fileImage(
                        "/usr/lib/libD2B.dylib",
                        targets: ["/usr/lib/libD2A.dylib"]
                    ),
                ],
                2
            ),
            (
                [
                    "/usr/lib/libD2A.dylib",
                    "/usr/lib/libD2B.dylib",
                ],
                [
                    .fileImage(
                        "/usr/lib/libD2A.dylib",
                        targets: ["/usr/lib/libD2C.dylib"]
                    ),
                    .fileImage(
                        "/usr/lib/libD2B.dylib",
                        targets: ["/usr/lib/libD2C.dylib"]
                    ),
                    .fileImage("/usr/lib/libD2C.dylib"),
                ],
                3
            ),
        ]

        for item in cases {
            let fixture = try Self.fixture(
                rootTargets: item.rootTargets,
                specs: item.specs
            )
            let first = try Self.compare(fixture)
            let second = try Self.compare(fixture)
            XCTAssertEqual(first, second)
            XCTAssertEqual(
                first.fileImageMemberCount,
                item.expectedFileImages
            )
            XCTAssertEqual(
                first.reachableMemberContentEvidenceIDs,
                fixture.members.map(\.contentEvidenceID)
            )
            XCTAssertEqual(first.validatedEdges.count, fixture.edges.count)
            Self.assertClaims(first)
        }
        XCTAssertEqual(effects, .zero)
    }

    func testVersionedZeroFileImageGraphMatchesV1Projection() throws {
        let effects = Effects.zero
        let fixture = try SyntheticRuntimeClosureGraphTests
            .fmbCanonicalSharedCacheFixture()
        let v1 = try SyntheticRuntimeClosureGraphVerifier.compare(
            root: fixture.root,
            members: fixture.members,
            edges: fixture.edges,
            collection: fixture.collection,
            rootInventory: fixture.rootInventory,
            memberInventories: fixture.memberInventories
        )
        let versioned = try SyntheticFileImageRuntimeClosureGraphVerifier
            .compare(
                root: fixture.root,
                members: fixture.members,
                edges: fixture.edges,
                collection: fixture.collection,
                rootInventory: fixture.rootInventory,
                memberInventories: fixture.memberInventories
            )

        XCTAssertEqual(versioned.fileImageMemberCount, 0)
        XCTAssertEqual(
            versioned.rootContentEvidenceID,
            v1.rootContentEvidenceID
        )
        XCTAssertEqual(versioned.rootRole, v1.rootRole)
        XCTAssertEqual(versioned.collection, v1.collection)
        XCTAssertEqual(
            versioned.inventorySummaries,
            v1.inventorySummaries
        )
        XCTAssertEqual(versioned.validatedEdges, v1.validatedEdges)
        XCTAssertEqual(
            versioned.reachableMemberContentEvidenceIDs,
            v1.reachableMemberContentEvidenceIDs
        )
        Self.assertClaims(versioned)
        XCTAssertEqual(effects, .zero)
    }

    func testV1StillRejectsSealedFileImageBeforeInventoryMutation()
        throws
    {
        let effects = Effects.zero
        let fixture = try Self.fixture(
            rootTargets: ["/usr/lib/libD2A.dylib"],
            specs: [.fileImage("/usr/lib/libD2A.dylib")]
        )
        XCTAssertThrowsError(
            try SyntheticRuntimeClosureGraphVerifier.compare(
                root: fixture.root,
                members: fixture.members,
                edges: fixture.edges,
                collection: fixture.collection,
                rootInventory: fixture.rootInventory,
                memberInventories: fixture.memberInventories
            )
        ) {
            XCTAssertEqual(
                $0 as? SyntheticRuntimeClosureGraphFailure,
                .unsupportedFileMemberIdentity(memberIndex: 0)
            )
        }
        XCTAssertEqual(effects, .zero)
    }

    func testVersionedGraphRejectsGenericFileBeforeInventoryMatching()
        throws
    {
        let effects = Effects.zero
        let memberName = "/usr/lib/libGeneric.dylib"
        let root = try SyntheticRuntimeClosureGraphTests.fmbRootEvidence(
            role: .git,
            loadCommands: [
                FMAFileImageFixture.dylibCommand(
                    command: FMAFileImageFixture.lcLoadDylib,
                    name: memberName
                ),
            ]
        )
        let generic = try SyntheticRuntimeClosureGraphTests
            .fmbExecutableEvidence(role: .fileImage, seed: 601)
        let member = try SyntheticRuntimeClosureRecordSchemaVerifier
            .member(
                index: 0,
                source: .file(generic),
                installName: Self.installName(memberName)
            )
        let rootInventory = try
            SyntheticAcceptedDependencyCommandInventoryVerifier.root(root)
        let entry = try XCTUnwrap(rootInventory.entries.first)
        let edge = try SyntheticRuntimeClosureRecordSchemaVerifier.edge(
            index: 0,
            parent: .root(root),
            loadCommandOrdinal: entry.loadCommandOrdinal,
            kind: entry.kind,
            installName: member.installName,
            resolved: member
        )
        let collection = try
            SyntheticRuntimeClosureRecordCollectionVerifier.derive(
                members: [member],
                edges: [edge]
            )

        XCTAssertThrowsError(
            try SyntheticFileImageRuntimeClosureGraphVerifier.compare(
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

    func testCrossImageInventorySubstitutionUsesMissingForeignPrecedence()
        throws
    {
        let effects = Effects.zero
        let first = try Self.fixture(
            rootTargets: ["/usr/lib/libD2A.dylib"],
            specs: [.fileImage("/usr/lib/libD2A.dylib")]
        )
        let second = try Self.fixture(
            rootTargets: ["/usr/lib/libD2B.dylib"],
            specs: [.fileImage("/usr/lib/libD2B.dylib")]
        )

        XCTAssertThrowsError(
            try SyntheticFileImageRuntimeClosureGraphVerifier.compare(
                root: first.root,
                members: first.members,
                edges: first.edges,
                collection: first.collection,
                rootInventory: first.rootInventory,
                memberInventories: second.memberInventories
            )
        ) {
            XCTAssertEqual(
                $0 as? SyntheticRuntimeClosureGraphFailure,
                .missingMemberInventory(memberIndex: 0)
            )
        }
        XCTAssertEqual(effects, .zero)
    }
}

private extension SyntheticFileImageRuntimeClosureGraphTests {
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

    enum MemberKind {
        case sharedCache
        case fileImage
    }

    struct MemberSpec {
        let name: String
        let kind: MemberKind
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

    struct Fixture {
        let root: ExecutableContentIdentityEvidence
        let members: [SyntheticRuntimeClosureMemberRecordComparison]
        let edges: [SyntheticRuntimeClosureEdgeRecordComparison]
        let collection:
            SyntheticRuntimeClosureRecordCollectionComparison
        let rootInventory:
            SyntheticAcceptedDependencyCommandInventoryComparison
        let memberInventories:
            [SyntheticAcceptedDependencyCommandInventoryComparison]
    }

    struct EdgeInput {
        let parent: SyntheticRuntimeClosureEdgeParent
        let parentID: String
        let ordinal: UInt64
        let kind: SyntheticRuntimeClosureEdgeKind
        let resolved: SyntheticRuntimeClosureMemberRecordComparison
    }

    static func fixture(
        rootTargets: [String],
        specs: [MemberSpec]
    ) throws -> Fixture {
        var inputs: [MemberInput] = []
        for spec in specs {
            let commands = spec.targets.map {
                FMAFileImageFixture.dylibCommand(
                    command: FMAFileImageFixture.lcLoadDylib,
                    name: $0
                )
            }
            switch spec.kind {
            case .sharedCache:
                let snapshot = try SyntheticRuntimeClosureGraphTests
                    .fmbSharedCacheSnapshot(
                        loadCommands: commands,
                        installName: spec.name
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
                installName: installName(input.name)
            )
        }
        let inventoryByID = Dictionary(
            uniqueKeysWithValues: inputs.map {
                ($0.contentEvidenceID, $0.inventory)
            }
        )
        let inventories = members.map {
            inventoryByID[$0.contentEvidenceID]!
        }
        let memberByName = Dictionary(
            uniqueKeysWithValues: members.map {
                (String(decoding: $0.decodedInstallName, as: UTF8.self), $0)
            }
        )

        let rootCommands = rootTargets.map {
            FMAFileImageFixture.dylibCommand(
                command: FMAFileImageFixture.lcLoadDylib,
                name: $0
            )
        }
        let root = try SyntheticRuntimeClosureGraphTests.fmbRootEvidence(
            role: .git,
            loadCommands: rootCommands
        )
        let rootInventory = try
            SyntheticAcceptedDependencyCommandInventoryVerifier.root(root)

        var edgeInputs: [EdgeInput] = []
        for entry in rootInventory.entries {
            let resolved = memberByName[
                String(decoding: entry.decodedInstallName, as: UTF8.self)
            ]!
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
        for (member, inventory) in zip(members, inventories) {
            for entry in inventory.entries {
                let resolved = memberByName[
                    String(
                        decoding: entry.decodedInstallName,
                        as: UTF8.self
                    )
                ]!
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
        return Fixture(
            root: root,
            members: members,
            edges: edges,
            collection: collection,
            rootInventory: rootInventory,
            memberInventories: inventories
        )
    }

    static func compare(
        _ fixture: Fixture
    ) throws -> SyntheticFileImageRuntimeClosureGraphComparison {
        try SyntheticFileImageRuntimeClosureGraphVerifier.compare(
            root: fixture.root,
            members: fixture.members,
            edges: fixture.edges,
            collection: fixture.collection,
            rootInventory: fixture.rootInventory,
            memberInventories: fixture.memberInventories
        )
    }

    static func installName(
        _ value: String
    ) -> SyntheticRuntimeClosureInstallName {
        FMAFileImageFixture.installName(value)
    }

    static func assertClaims(
        _ value: SyntheticFileImageRuntimeClosureGraphComparison,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertTrue(value.provesGraphMembership, file: file, line: line)
        XCTAssertTrue(
            value.provesAcceptedCommandCompleteness,
            file: file,
            line: line
        )
        XCTAssertTrue(value.provesBoundedTraversal, file: file, line: line)
        XCTAssertTrue(value.provesRootReachability, file: file, line: line)
        XCTAssertTrue(
            value.provesSealedFileImageContinuity,
            file: file,
            line: line
        )
        XCTAssertFalse(value.isCompleteRuntimeClosure, file: file, line: line)
        XCTAssertFalse(
            value.provesRuntimeLaunchability,
            file: file,
            line: line
        )
        XCTAssertEqual(value.runtimeDecision, .noGo, file: file, line: line)
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
