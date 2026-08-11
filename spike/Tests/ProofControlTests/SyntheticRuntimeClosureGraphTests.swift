import CryptoKit
import Foundation
import XCTest
@testable import ProofControl

final class SyntheticRuntimeClosureGraphTests: XCTestCase {
    func testThreeMemberGraphRetainsOnlyCompactStaticClaims()
        throws
    {
        let effects = Effects()
        var fixture = try Self.canonicalFixture()

        let result = try Self.compare(fixture)

        XCTAssertEqual(
            result.rootContentEvidenceID,
            fixture.root.contentEvidenceID.sha256
        )
        XCTAssertEqual(result.rootRole, .git)
        XCTAssertEqual(result.collection, fixture.collection)
        let orderedInventories = fixture.orderedInventories
        XCTAssertEqual(
            result.inventorySummaries.map(\.parentContentEvidenceID),
            orderedInventories.map(\.parentContentEvidenceID)
        )
        XCTAssertEqual(
            result.inventorySummaries.map(\.sourceLoadCommandsSHA256),
            orderedInventories.map(\.sourceLoadCommandsSHA256)
        )
        XCTAssertEqual(
            result.inventorySummaries.map(\.acceptedCommandCount),
            orderedInventories.map { $0.entries.count }
        )
        XCTAssertEqual(
            result.validatedEdges.map(\.parentContentEvidenceID),
            fixture.edges.map(\.parentContentEvidenceID)
        )
        XCTAssertEqual(
            result.validatedEdges.map(\.loadCommandOrdinal),
            fixture.edges.map(\.loadCommandOrdinal)
        )
        XCTAssertEqual(
            result.validatedEdges.map(\.kind),
            fixture.edges.map(\.kind)
        )
        XCTAssertEqual(
            result.validatedEdges.map(\.decodedInstallName),
            fixture.edges.map(\.decodedInstallName)
        )
        XCTAssertEqual(
            result.validatedEdges.map(\.resolvedContentEvidenceID),
            fixture.edges.map(\.resolvedContentEvidenceID)
        )
        XCTAssertEqual(
            result.reachableMemberContentEvidenceIDs,
            fixture.members.map(\.contentEvidenceID)
        )
        Self.assertGraphClaims(result)

        fixture.members.removeAll()
        fixture.edges.removeAll()
        fixture.memberInventories.removeAll()
        XCTAssertEqual(result.collection, fixture.originalCollection)
        XCTAssertEqual(
            result.reachableMemberContentEvidenceIDs,
            fixture.originalMembers.map(\.contentEvidenceID)
        )
        XCTAssertEqual(effects, .zero)
    }

    func testRootAndPredecessorContinuityFailuresWinBeforeGraphWork()
        throws
    {
        let effects = Effects()
        let fixture = try Self.canonicalFixture()
        let otherRoot = try Self.rootEvidence(
            role: .selfGuard,
            loadCommands: [
                Self.dylibCommand(
                    cmd: Self.lcLoadDylib,
                    name: fixture.name(of: fixture.members[0])
                ),
            ]
        )
        let otherRootInventory =
            try SyntheticAcceptedDependencyCommandInventoryVerifier
            .root(otherRoot)

        try Self.assertGraphFailure(
            root: try Self.rootEvidence(
                role: .dynamicLoader,
                loadCommands: []
            ),
            fixture: fixture,
            expected: .rootRole(.dynamicLoader)
        )
        try Self.assertGraphFailure(
            rootInventory: otherRootInventory,
            fixture: fixture,
            expected: .rootInventoryMismatch
        )

        let mismatchedCollection =
            try SyntheticRuntimeClosureRecordCollectionVerifier.derive(
                members: Array(fixture.members.prefix(2)),
                edges: Array(fixture.edges.prefix(1))
            )
        try Self.assertGraphFailure(
            collection: mismatchedCollection,
            fixture: fixture,
            expected: .collectionMismatch
        )

        let duplicateName = try Self.fixture(
            rootTargets: ["/usr/lib/libDuplicateName.dylib"],
            memberSpecs: [
                ImageSpec(
                    name: "/usr/lib/libDuplicateName.dylib",
                    commands: [
                        Self.dylibCommand(
                            cmd: Self.lcLoadDylib,
                            name: "/usr/lib/libDuplicateName.dylib"
                        ),
                    ]
                ),
                ImageSpec(
                    name: "/usr/lib/libDuplicateName.dylib",
                    commands: []
                ),
            ],
            allowCollectionFailure: true
        )
        XCTAssertThrowsError(
            try Self.compare(duplicateName)
        ) {
            XCTAssertEqual(
                $0 as? SyntheticRuntimeClosureRecordCollectionFailure,
                .duplicateMemberInstallName(index: 1)
            )
        }
        XCTAssertEqual(effects, .zero)
    }

    func testUnsupportedFileAndDynamicLoaderInputsStayClosed()
        throws
    {
        let effects = Effects()
        let fixture = try Self.canonicalFixture()
        let fileMember = try SyntheticRuntimeClosureRecordSchemaVerifier
            .member(
                index: 0,
                source: .file(
                    try Self.executableEvidence(
                        role: .fileImage,
                        seed: 91
                    )
                ),
                installName: Self.installName(
                    "/usr/lib/libFileMember.dylib"
                )
            )
        try Self.assertGraphFailure(
            members: [fileMember] + Array(fixture.members.dropFirst()),
            fixture: fixture,
            expected: .unsupportedFileMemberIdentity(memberIndex: 0)
        )

        XCTAssertThrowsError(
            try SyntheticRuntimeClosureRecordSchemaVerifier.member(
                index: 0,
                source: .file(
                    try Self.rootEvidence(
                        role: .dynamicLoader,
                        loadCommands: []
                    )
                ),
                installName: Self.installName(
                    "/usr/lib/libDyldAsMember.dylib"
                )
            )
        ) {
            XCTAssertEqual(
                $0 as? SyntheticRuntimeClosureRecordSchemaFailure,
                .unsupportedFileMemberRole(.dynamicLoader)
            )
        }

        let sameBytesRoot = try Self.rootEvidence(
            role: .git,
            loadCommands: []
        )
        let sameBytesFile = try Self.executableEvidence(
            role: .fileImage,
            seed: 0
        )
        XCTAssertNotEqual(
            sameBytesRoot.contentEvidenceID.sha256,
            sameBytesFile.contentEvidenceID.sha256
        )
        XCTAssertEqual(effects, .zero)
    }

    func testMemberInventoryContinuityRejectsCountDuplicateMissingAndForeign()
        throws
    {
        let effects = Effects()
        let fixture = try Self.canonicalFixture()
        let foreign = try Self.fixture(
            rootTargets: ["/usr/lib/libForeign.dylib"],
            memberSpecs: [
                ImageSpec(name: "/usr/lib/libForeign.dylib")
            ]
        )

        try Self.assertGraphFailure(
            memberInventories:
                Array(fixture.memberInventories.dropLast()),
            fixture: fixture,
            expected: .memberInventoryCountMismatch
        )
        try Self.assertGraphFailure(
            memberInventories: [
                fixture.memberInventories[0],
                fixture.memberInventories[0],
                fixture.memberInventories[2],
            ],
            fixture: fixture,
            expected: .duplicateMemberInventory(memberIndex: 0)
        )
        try Self.assertGraphFailure(
            memberInventories: [
                fixture.memberInventories[0],
                fixture.memberInventories[2],
                foreign.memberInventories[0],
            ],
            fixture: fixture,
            expected: .missingMemberInventory(memberIndex: 1)
        )
        try Self.assertGraphFailure(
            memberInventories: [
                fixture.memberInventories[0],
                foreign.memberInventories[0],
                fixture.memberInventories[1],
            ],
            fixture: fixture,
            expected: .foreignMemberInventory(inventoryIndex: 1)
        )
        XCTAssertEqual(effects, .zero)
    }

    func testEndpointMembershipRejectsMissingForeignAndMismatchedEdges()
        throws
    {
        let effects = Effects()
        let fixture = try Self.canonicalFixture()
        let outside = try Self.fixture(
            rootTargets: ["/usr/lib/libOutside.dylib"],
            memberSpecs: [
                ImageSpec(name: "/usr/lib/libOutside.dylib")
            ]
        )
        let outsideMember = outside.members[0]
        let rootEdge = fixture.edgeFromRoot(
            to: fixture.record(named: "/usr/lib/libA.dylib")
        )

        try Self.assertGraphFailure(
            edges: try Self.orderedEdges([
                EdgeInput(
                    parent: .member(outsideMember),
                    parentContentEvidenceID:
                        outsideMember.contentEvidenceID,
                    loadCommandOrdinal: 0,
                    kind: .load,
                    resolved: fixture.members[0]
                ),
            ]),
            fixture: fixture,
            expected: .edgeParentMissing(edgeIndex: 0)
        )

        let otherRoot = try Self.rootEvidence(
            role: .selfGuard,
            loadCommands: [
                Self.dylibCommand(
                    cmd: Self.lcLoadDylib,
                    name: fixture.name(of: fixture.members[0])
                ),
            ]
        )
        try Self.assertGraphFailure(
            edges: try Self.orderedEdges([
                EdgeInput(
                    parent: .root(otherRoot),
                    parentContentEvidenceID:
                        otherRoot.contentEvidenceID.sha256,
                    loadCommandOrdinal: rootEdge.loadCommandOrdinal,
                    kind: rootEdge.kind,
                    resolved: fixture.members[0]
                ),
            ]),
            fixture: fixture,
            expected: .edgeParentEvidenceMismatch(edgeIndex: 0)
        )

        try Self.assertGraphFailure(
            edges: try Self.orderedEdges([
                EdgeInput(
                    parent: .root(fixture.root),
                    parentContentEvidenceID:
                        fixture.root.contentEvidenceID.sha256,
                    loadCommandOrdinal: rootEdge.loadCommandOrdinal,
                    kind: rootEdge.kind,
                    resolved: outsideMember
                ),
            ]),
            fixture: fixture,
            expected: .edgeResolvedMissing(edgeIndex: 0)
        )

        let memberA = fixture.record(named: "/usr/lib/libA.dylib")
        let driftedMemberA = try
            SyntheticRuntimeClosureRecordSchemaVerifier.member(
                index: memberA.index == 0 ? 1 : 0,
                source: memberA.source,
                installName: memberA.installName
            )
        try Self.assertGraphFailure(
            edges: try Self.orderedEdges([
                EdgeInput(
                    parent: .root(fixture.root),
                    parentContentEvidenceID:
                        fixture.root.contentEvidenceID.sha256,
                    loadCommandOrdinal: rootEdge.loadCommandOrdinal,
                    kind: rootEdge.kind,
                    resolved: driftedMemberA
                ),
            ]),
            fixture: fixture,
            expected: .edgeResolvedEvidenceMismatch(edgeIndex: 0)
        )
        XCTAssertEqual(effects, .zero)
    }

    func testAcceptedCommandBijectionRejectsOrdinalKindNameAndUnconsumedDrift()
        throws
    {
        let effects = Effects()
        let fixture = try Self.canonicalFixture()
        let memberA = fixture.record(named: "/usr/lib/libA.dylib")
        let memberB = fixture.record(named: "/usr/lib/libB.dylib")
        let memberC = fixture.record(named: "/usr/lib/libC.dylib")
        let memberACommand = fixture.inventory(for: memberA).entries[0]

        let missingInventoryEntryEdges = try Self.replacingEdge(
            parent: .member(memberA),
            parentContentEvidenceID: memberA.contentEvidenceID,
            ordinal: 99,
            kind: .load,
            resolved: memberC,
            in: fixture
        )
        try Self.assertGraphFailure(
            edges: missingInventoryEntryEdges,
            fixture: fixture,
            expected: .edgeInventoryEntryMissing(
                edgeIndex: missingInventoryEntryEdges.index(
                    parentContentEvidenceID: memberA.contentEvidenceID,
                    ordinal: 99
                )
            )
        )
        let kindMismatchEdges = try Self.replacingEdge(
            parent: .member(memberA),
            parentContentEvidenceID: memberA.contentEvidenceID,
            ordinal: memberACommand.loadCommandOrdinal,
            kind: .reexport,
            resolved: memberC,
            in: fixture
        )
        try Self.assertGraphFailure(
            edges: kindMismatchEdges,
            fixture: fixture,
            expected: .edgeInventoryKindMismatch(
                edgeIndex: kindMismatchEdges.index(
                    parentContentEvidenceID: memberA.contentEvidenceID,
                    ordinal: memberACommand.loadCommandOrdinal
                )
            )
        )
        let nameMismatchEdges = try Self.replacingEdge(
            parent: .member(memberA),
            parentContentEvidenceID: memberA.contentEvidenceID,
            ordinal: memberACommand.loadCommandOrdinal,
            kind: .load,
            resolved: memberB,
            in: fixture
        )
        try Self.assertGraphFailure(
            edges: nameMismatchEdges,
            fixture: fixture,
            expected: .edgeInventoryInstallNameMismatch(
                edgeIndex: nameMismatchEdges.index(
                    parentContentEvidenceID: memberA.contentEvidenceID,
                    ordinal: memberACommand.loadCommandOrdinal
                )
            )
        )

        let missingEdgeFixture = try Self.fixture(
            rootTargets: ["/usr/lib/libA.dylib"],
            memberSpecs: [
                ImageSpec(
                    name: "/usr/lib/libA.dylib",
                    commands: [
                        Self.dylibCommand(
                            cmd: Self.lcLoadDylib,
                            name: "/usr/lib/libB.dylib"
                        ),
                    ]
                ),
                ImageSpec(name: "/usr/lib/libB.dylib"),
            ]
        )
        let onlyRootEdges = missingEdgeFixture.edges.filter {
            $0.parentContentEvidenceID ==
                missingEdgeFixture.root.contentEvidenceID.sha256
        }
        try Self.assertGraphFailure(
            edges: onlyRootEdges,
            fixture: missingEdgeFixture,
            expected: .unconsumedInventoryEntry(
                parentIndex: missingEdgeFixture.inventoryParentIndex(
                    for: missingEdgeFixture.record(
                        named: "/usr/lib/libA.dylib"
                    ).contentEvidenceID
                ),
                ordinal: 0
            )
        )

        let unresolved = try Self.fixture(
            rootTargets: ["/usr/lib/libA.dylib"],
            memberSpecs: [
                ImageSpec(
                    name: "/usr/lib/libA.dylib",
                    commands: [
                        Self.dylibCommand(
                            cmd: Self.lcLoadDylib,
                            name: "/usr/lib/libMissing.dylib"
                        ),
                    ]
                ),
            ],
            allowUnresolvedCommands: true
        )
        try Self.assertGraphFailure(
            fixture: unresolved,
            expected: .unresolvedInventoryEntry(
                parentIndex: unresolved.inventoryParentIndex(
                    for: unresolved.record(
                        named: "/usr/lib/libA.dylib"
                    ).contentEvidenceID
                ),
                ordinal: 0
            )
        )
        XCTAssertEqual(effects, .zero)
    }

    func testCyclesSelfLoopDiamondAndLeafStayReachableWithoutRuntimeClaims()
        throws
    {
        let effects = Effects()
        let cycle = try Self.fixture(
            rootTargets: ["/usr/lib/libA.dylib"],
            memberSpecs: [
                ImageSpec(
                    name: "/usr/lib/libA.dylib",
                    commands: [
                        Self.dylibCommand(
                            cmd: Self.lcLoadDylib,
                            name: "/usr/lib/libB.dylib"
                        ),
                    ]
                ),
                ImageSpec(
                    name: "/usr/lib/libB.dylib",
                    commands: [
                        Self.dylibCommand(
                            cmd: Self.lcReexportDylib,
                            name: "/usr/lib/libA.dylib"
                        ),
                    ]
                ),
            ]
        )
        Self.assertGraphClaims(try Self.compare(cycle))

        let selfLoop = try Self.fixture(
            rootTargets: ["/usr/lib/libSelf.dylib"],
            memberSpecs: [
                ImageSpec(
                    name: "/usr/lib/libSelf.dylib",
                    commands: [
                        Self.dylibCommand(
                            cmd: Self.lcLoadDylib,
                            name: "/usr/lib/libSelf.dylib"
                        ),
                    ]
                ),
            ]
        )
        Self.assertGraphClaims(try Self.compare(selfLoop))

        let diamond = try Self.fixture(
            rootTargets: [
                "/usr/lib/libA.dylib",
                "/usr/lib/libB.dylib",
            ],
            memberSpecs: [
                ImageSpec(
                    name: "/usr/lib/libA.dylib",
                    commands: [
                        Self.dylibCommand(
                            cmd: Self.lcLoadDylib,
                            name: "/usr/lib/libLeaf.dylib"
                        ),
                    ]
                ),
                ImageSpec(
                    name: "/usr/lib/libB.dylib",
                    commands: [
                        Self.dylibCommand(
                            cmd: Self.lcLoadDylib,
                            name: "/usr/lib/libLeaf.dylib"
                        ),
                    ]
                ),
                ImageSpec(name: "/usr/lib/libLeaf.dylib"),
            ]
        )
        Self.assertGraphClaims(try Self.compare(diamond))
        XCTAssertEqual(effects, .zero)
    }

    func testUnreachableIslandRejectsAfterValidatedCycleWork()
        throws
    {
        let effects = Effects()
        let fixture = try Self.fixture(
            rootTargets: ["/usr/lib/libA.dylib"],
            memberSpecs: [
                ImageSpec(
                    name: "/usr/lib/libA.dylib",
                    commands: [
                        Self.dylibCommand(
                            cmd: Self.lcLoadDylib,
                            name: "/usr/lib/libB.dylib"
                        ),
                    ]
                ),
                ImageSpec(
                    name: "/usr/lib/libB.dylib",
                    commands: [
                        Self.dylibCommand(
                            cmd: Self.lcLoadDylib,
                            name: "/usr/lib/libA.dylib"
                        ),
                    ]
                ),
                ImageSpec(
                    name: "/usr/lib/libIsland.dylib",
                    commands: [
                        Self.dylibCommand(
                            cmd: Self.lcLoadDylib,
                            name: "/usr/lib/libIsland.dylib"
                        ),
                    ]
                ),
            ]
        )

        try Self.assertGraphFailure(
            fixture: fixture,
            expected: .unreachableMember(
                memberIndex: fixture.index(
                    of: "/usr/lib/libIsland.dylib"
                )
            )
        )
        XCTAssertEqual(effects, .zero)
    }

    func testMaximumMemberAndEdgeGraphExercisesCheckedLinearWork()
        throws
    {
        let effects = Effects()
        let names = (0..<256).map {
            "/usr/lib/libChain\($0).dylib"
        }
        let specs = names.enumerated().map { index, name in
            var targets: [String] = []
            if index + 1 < names.count {
                targets.append(names[index + 1])
            }
            targets.append(
                contentsOf: repeatElement(names[0], count: 15)
            )
            return ImageSpec(
                name: name,
                commands: targets.map {
                    Self.dylibCommand(
                        cmd: Self.lcLoadDylib,
                        name: $0
                    )
                }
            )
        }
        let fixture = try Self.fixture(
            rootTargets: [names[0]],
            memberSpecs: specs
        )

        let result = try Self.compare(fixture)

        XCTAssertEqual(result.collection.memberCount, 256)
        XCTAssertEqual(result.collection.edgeCount, 4_096)
        XCTAssertEqual(
            result.reachableMemberContentEvidenceIDs.count,
            256
        )
        Self.assertGraphClaims(result)

        var overflowSpecs = specs
        overflowSpecs[overflowSpecs.index(before: overflowSpecs.endIndex)] =
            ImageSpec(
                name: names[names.index(before: names.endIndex)],
                commands: Array(repeating: Self.dylibCommand(
                    cmd: Self.lcLoadDylib,
                    name: names[0]
                ), count: 16)
            )
        let overflow = try Self.fixture(
            rootTargets: [names[0]],
            memberSpecs: overflowSpecs,
            maximumConstructedEdges: 4_096
        )
        XCTAssertEqual(overflow.edges.count, 4_096)
        let capWitness = overflow.aggregateInventoryEntry(at: 4_096)
        try Self.assertGraphFailure(
            fixture: overflow,
            expected: .unconsumedInventoryEntry(
                parentIndex: capWitness.parentIndex,
                ordinal: capWitness.entry.loadCommandOrdinal
            )
        )
        XCTAssertEqual(effects, .zero)
    }

    func testFirstFailurePrecedenceIsDeterministic()
        throws
    {
        let effects = Effects()
        let fixture = try Self.canonicalFixture()
        let mismatchedCollection =
            try SyntheticRuntimeClosureRecordCollectionVerifier.derive(
                members: Array(fixture.members.prefix(2)),
                edges: Array(fixture.edges.prefix(1))
            )
        let otherRootInventory =
            try SyntheticAcceptedDependencyCommandInventoryVerifier
            .root(
                Self.rootEvidence(
                    role: .selfGuard,
                    loadCommands: []
                )
            )

        try Self.assertGraphFailure(
            root: try Self.rootEvidence(
                role: .dynamicLoader,
                loadCommands: []
            ),
            collection: mismatchedCollection,
            rootInventory: otherRootInventory,
            fixture: fixture,
            expected: .rootRole(.dynamicLoader)
        )
        try Self.assertGraphFailure(
            collection: mismatchedCollection,
            rootInventory: otherRootInventory,
            fixture: fixture,
            expected: .rootInventoryMismatch
        )
        XCTAssertEqual(effects, .zero)
    }
}

private extension SyntheticRuntimeClosureGraphTests {
    struct Effects: Equatable {
        var fileSystemMutations = 0
        var networkOperations = 0
        var processSpawns = 0
        var packConsumes = 0
        var objectDatabaseImports = 0
        var buildOperations = 0
        var modelLoads = 0
        var outputReservations = 0
        var publications = 0

        static let zero = Self()
    }

    struct ImageSpec {
        let name: String
        let commands: [Data]

        init(name: String, commands: [Data] = []) {
            self.name = name
            self.commands = commands
        }
    }

    struct ImageInput {
        let name: String
        let snapshot: SyntheticSharedCacheImageLoadCommandSnapshot

        var contentEvidenceID: String {
            snapshot.imageEvidence.contentEvidenceID.sha256
        }
    }

    struct EdgeInput {
        let parent: SyntheticRuntimeClosureEdgeParent
        let parentContentEvidenceID: String
        let loadCommandOrdinal: UInt64
        let kind: SyntheticRuntimeClosureEdgeKind
        let resolved: SyntheticRuntimeClosureMemberRecordComparison
    }

    struct Fixture {
        var root: ExecutableContentIdentityEvidence
        var members: [SyntheticRuntimeClosureMemberRecordComparison]
        var edges: [SyntheticRuntimeClosureEdgeRecordComparison]
        var collection:
            SyntheticRuntimeClosureRecordCollectionComparison
        var rootInventory:
            SyntheticAcceptedDependencyCommandInventoryComparison
        var memberInventories:
            [SyntheticAcceptedDependencyCommandInventoryComparison]
        let originalMembers:
            [SyntheticRuntimeClosureMemberRecordComparison]
        let originalCollection:
            SyntheticRuntimeClosureRecordCollectionComparison

        var orderedInventories: [
            SyntheticAcceptedDependencyCommandInventoryComparison
        ] {
            ([rootInventory] + memberInventories).sorted {
                $0.parentContentEvidenceID.utf8
                    .lexicographicallyPrecedes(
                        $1.parentContentEvidenceID.utf8
                    )
            }
        }

        func name(
            of member: SyntheticRuntimeClosureMemberRecordComparison
        ) -> String {
            String(decoding: member.decodedInstallName, as: UTF8.self)
        }

        func record(
            named name: String
        ) -> SyntheticRuntimeClosureMemberRecordComparison {
            members.first {
                String(decoding: $0.decodedInstallName, as: UTF8.self) ==
                    name
            }!
        }

        func inventory(
            for member: SyntheticRuntimeClosureMemberRecordComparison
        ) -> SyntheticAcceptedDependencyCommandInventoryComparison {
            memberInventories.first {
                $0.parentContentEvidenceID == member.contentEvidenceID
            }!
        }

        func index(of name: String) -> Int {
            record(named: name).index
        }

        func inventoryParentIndex(for contentEvidenceID: String) -> Int {
            orderedInventories.firstIndex {
                $0.parentContentEvidenceID == contentEvidenceID
            }!
        }

        func aggregateInventoryEntry(
            at aggregateIndex: Int
        ) -> (
            parentIndex: Int,
            entry: SyntheticAcceptedDependencyCommand
        ) {
            var remaining = aggregateIndex
            for (parentIndex, inventory) in
                orderedInventories.enumerated()
            {
                if remaining < inventory.entries.count {
                    return (parentIndex, inventory.entries[remaining])
                }
                remaining -= inventory.entries.count
            }
            fatalError("aggregate inventory index out of bounds")
        }

        func edgeFromRoot(
            to member: SyntheticRuntimeClosureMemberRecordComparison
        ) -> SyntheticRuntimeClosureEdgeRecordComparison {
            edges.first {
                $0.parentContentEvidenceID ==
                    root.contentEvidenceID.sha256 &&
                    $0.resolvedContentEvidenceID ==
                    member.contentEvidenceID
            }!
        }
    }

    static let lcLoadDylib: UInt32 = 0x0c
    static let lcReexportDylib: UInt32 = 0x8000001f

    static func canonicalFixture() throws -> Fixture {
        try fixture(
            rootTargets: [
                "/usr/lib/libA.dylib",
                "/usr/lib/libB.dylib",
            ],
            rootKinds: [.load, .reexport],
            memberSpecs: [
                ImageSpec(
                    name: "/usr/lib/libA.dylib",
                    commands: [
                        dylibCommand(
                            cmd: lcLoadDylib,
                            name: "/usr/lib/libC.dylib"
                        ),
                    ]
                ),
                ImageSpec(name: "/usr/lib/libB.dylib"),
                ImageSpec(name: "/usr/lib/libC.dylib"),
            ]
        )
    }

    static func fixture(
        rootTargets: [String],
        rootKinds: [SyntheticRuntimeClosureEdgeKind]? = nil,
        memberSpecs: [ImageSpec],
        rootRole: ExecutableContentArtifactRole = .git,
        allowUnresolvedCommands: Bool = false,
        allowCollectionFailure: Bool = false,
        maximumConstructedEdges: Int? = nil
    ) throws -> Fixture {
        let input = try memberSpecs.map {
            try ImageInput(
                name: $0.name,
                snapshot: sharedCacheSnapshot(
                    loadCommands: $0.commands,
                    installName: $0.name
                )
            )
        }.sorted {
            $0.contentEvidenceID.utf8.lexicographicallyPrecedes(
                $1.contentEvidenceID.utf8
            )
        }
        let members = try input.enumerated().map { index, value in
            try SyntheticRuntimeClosureRecordSchemaVerifier.member(
                index: index,
                source: .sharedCache(value.snapshot.imageEvidence),
                installName: installName(value.name)
            )
        }
        let snapshotsByID = Dictionary(
            uniqueKeysWithValues: input.map {
                ($0.contentEvidenceID, $0.snapshot)
            }
        )
        let inventories = try members.map {
            try SyntheticAcceptedDependencyCommandInventoryVerifier
                .sharedCacheMember(
                    snapshotsByID[$0.contentEvidenceID]!
                )
        }

        let kinds = rootKinds ??
            Array(repeating: .load, count: rootTargets.count)
        XCTAssertEqual(kinds.count, rootTargets.count)
        let rootCommands = zip(rootTargets, kinds).map { name, kind in
            dylibCommand(
                cmd: kind == .load ? lcLoadDylib : lcReexportDylib,
                name: name
            )
        }
        let root = try rootEvidence(
            role: rootRole,
            loadCommands: rootCommands
        )
        let rootInventory =
            try SyntheticAcceptedDependencyCommandInventoryVerifier
            .root(root)

        var membersByName: [
            String: SyntheticRuntimeClosureMemberRecordComparison
        ] = [:]
        for member in members {
            let name = String(
                decoding: member.decodedInstallName,
                as: UTF8.self
            )
            if membersByName[name] == nil {
                membersByName[name] = member
            }
        }
        var edgeInputs: [EdgeInput] = []
        for entry in rootInventory.entries {
            let name = String(
                decoding: entry.decodedInstallName,
                as: UTF8.self
            )
            guard let resolved = membersByName[name] else {
                if allowUnresolvedCommands { continue }
                throw TestFixtureFailure.unresolvedInstallName(name)
            }
            edgeInputs.append(
                EdgeInput(
                    parent: .root(root),
                    parentContentEvidenceID:
                        root.contentEvidenceID.sha256,
                    loadCommandOrdinal: entry.loadCommandOrdinal,
                    kind: entry.kind,
                    resolved: resolved
                )
            )
        }
        for (member, inventory) in zip(members, inventories) {
            for entry in inventory.entries {
                let name = String(
                    decoding: entry.decodedInstallName,
                    as: UTF8.self
                )
                guard let resolved = membersByName[name] else {
                    if allowUnresolvedCommands { continue }
                    throw TestFixtureFailure.unresolvedInstallName(name)
                }
                edgeInputs.append(
                    EdgeInput(
                        parent: .member(member),
                        parentContentEvidenceID:
                            member.contentEvidenceID,
                        loadCommandOrdinal: entry.loadCommandOrdinal,
                        kind: entry.kind,
                        resolved: resolved
                    )
                )
            }
        }

        let boundedEdgeInputs: [EdgeInput]
        if let maximumConstructedEdges {
            boundedEdgeInputs = Array(
                edgeInputs.sorted(by: edgeInputLess)
                    .prefix(maximumConstructedEdges)
            )
        } else {
            boundedEdgeInputs = edgeInputs
        }
        let edges = try orderedEdges(boundedEdgeInputs)
        let collection: SyntheticRuntimeClosureRecordCollectionComparison
        if allowCollectionFailure {
            collection = try SyntheticRuntimeClosureRecordCollectionVerifier
                .derive(
                    members: Array(members.prefix(1)),
                    edges: Array(edges.prefix(1))
                )
        } else {
            collection = try SyntheticRuntimeClosureRecordCollectionVerifier
                .derive(members: members, edges: edges)
        }
        return Fixture(
            root: root,
            members: members,
            edges: edges,
            collection: collection,
            rootInventory: rootInventory,
            memberInventories: inventories,
            originalMembers: members,
            originalCollection: collection
        )
    }

    static func compare(
        _ fixture: Fixture
    ) throws -> SyntheticRuntimeClosureGraphComparison {
        try SyntheticRuntimeClosureGraphVerifier.compare(
            root: fixture.root,
            members: fixture.members,
            edges: fixture.edges,
            collection: fixture.collection,
            rootInventory: fixture.rootInventory,
            memberInventories: fixture.memberInventories
        )
    }

    static func assertGraphFailure(
        root: ExecutableContentIdentityEvidence? = nil,
        members: [SyntheticRuntimeClosureMemberRecordComparison]? = nil,
        edges: [SyntheticRuntimeClosureEdgeRecordComparison]? = nil,
        collection: SyntheticRuntimeClosureRecordCollectionComparison? =
            nil,
        rootInventory:
            SyntheticAcceptedDependencyCommandInventoryComparison? =
            nil,
        memberInventories:
            [SyntheticAcceptedDependencyCommandInventoryComparison]? =
            nil,
        fixture: Fixture,
        expected: SyntheticRuntimeClosureGraphFailure,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let actualMembers = members ?? fixture.members
        let actualEdges = edges ?? fixture.edges
        let actualCollection:
            SyntheticRuntimeClosureRecordCollectionComparison
        if let collection {
            actualCollection = collection
        } else {
            actualCollection =
                try SyntheticRuntimeClosureRecordCollectionVerifier
                .derive(
                members: actualMembers,
                edges: actualEdges
                )
        }
        XCTAssertThrowsError(
            try SyntheticRuntimeClosureGraphVerifier.compare(
                root: root ?? fixture.root,
                members: actualMembers,
                edges: actualEdges,
                collection: actualCollection,
                rootInventory: rootInventory ?? fixture.rootInventory,
                memberInventories:
                    memberInventories ?? fixture.memberInventories
            ),
            file: file,
            line: line
        ) {
            XCTAssertEqual(
                $0 as? SyntheticRuntimeClosureGraphFailure,
                expected,
                file: file,
                line: line
            )
        }
    }

    static func replacingEdge(
        parent: SyntheticRuntimeClosureEdgeParent,
        parentContentEvidenceID: String,
        ordinal: UInt64,
        kind: SyntheticRuntimeClosureEdgeKind,
        resolved: SyntheticRuntimeClosureMemberRecordComparison,
        in fixture: Fixture
    ) throws -> [SyntheticRuntimeClosureEdgeRecordComparison] {
        let replacement = EdgeInput(
            parent: parent,
            parentContentEvidenceID: parentContentEvidenceID,
            loadCommandOrdinal: ordinal,
            kind: kind,
            resolved: resolved
        )
        let original = fixture.edges.map {
            EdgeInput(
                parent: $0.parent,
                parentContentEvidenceID: $0.parentContentEvidenceID,
                loadCommandOrdinal: $0.loadCommandOrdinal,
                kind: $0.kind,
                resolved: $0.resolved
            )
        }.filter {
            !(
                $0.parentContentEvidenceID ==
                    parentContentEvidenceID &&
                    $0.loadCommandOrdinal ==
                    fixture.inventory(
                        for: fixture.record(
                            named: "/usr/lib/libA.dylib"
                        )
                    ).entries[0].loadCommandOrdinal
            )
        }
        return try orderedEdges(original + [replacement])
    }

    static func orderedEdges(
        _ inputs: [EdgeInput]
    ) throws -> [SyntheticRuntimeClosureEdgeRecordComparison] {
        try inputs.sorted(by: edgeInputLess).enumerated().map {
            index,
            input in
            try SyntheticRuntimeClosureRecordSchemaVerifier.edge(
                index: index,
                parent: input.parent,
                loadCommandOrdinal: input.loadCommandOrdinal,
                kind: input.kind,
                installName: input.resolved.installName,
                resolved: input.resolved
            )
        }
    }

    static func edgeInputLess(
        _ lhs: EdgeInput,
        _ rhs: EdgeInput
    ) -> Bool {
        if lhs.parentContentEvidenceID != rhs.parentContentEvidenceID {
            return lhs.parentContentEvidenceID.utf8
                .lexicographicallyPrecedes(
                    rhs.parentContentEvidenceID.utf8
                )
        }
        return lhs.loadCommandOrdinal < rhs.loadCommandOrdinal
    }

    static func assertGraphClaims(
        _ value: SyntheticRuntimeClosureGraphComparison,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertTrue(
            value.provesGraphMembership,
            file: file,
            line: line
        )
        XCTAssertTrue(
            value.provesAcceptedCommandCompleteness,
            file: file,
            line: line
        )
        XCTAssertTrue(
            value.provesBoundedTraversal,
            file: file,
            line: line
        )
        XCTAssertTrue(
            value.provesRootReachability,
            file: file,
            line: line
        )
        XCTAssertFalse(
            value.isCompleteRuntimeClosure,
            file: file,
            line: line
        )
        XCTAssertFalse(
            value.provesRuntimeLaunchability,
            file: file,
            line: line
        )
        XCTAssertEqual(
            value.runtimeDecision,
            .noGo,
            file: file,
            line: line
        )
        XCTAssertFalse(value.canExecute, file: file, line: line)
        XCTAssertFalse(value.canSpawn, file: file, line: line)
        XCTAssertFalse(
            value.canAccessNetwork,
            file: file,
            line: line
        )
        XCTAssertFalse(value.canConsumePack, file: file, line: line)
        XCTAssertFalse(
            value.canMutateFileSystem,
            file: file,
            line: line
        )
        XCTAssertFalse(
            value.canImportGitObjects,
            file: file,
            line: line
        )
        XCTAssertFalse(value.canBuild, file: file, line: line)
        XCTAssertFalse(value.canLoadModel, file: file, line: line)
        XCTAssertFalse(
            value.canReserveOutput,
            file: file,
            line: line
        )
        XCTAssertFalse(value.canPublish, file: file, line: line)
    }

    static func sharedCacheSnapshot(
        loadCommands: [Data],
        installName: String
    ) throws -> SyntheticSharedCacheImageLoadCommandSnapshot {
        let bytes = commandRegion(loadCommands)
        return try SyntheticSharedCacheImageLoadCommandSnapshotVerifier
            .derive(
                imageEvidence: sharedCacheImageEvidence(
                    installName: installName,
                    loadCommandsSHA256: sha256Hex(bytes)
                ),
                loadCommandBytes: bytes
            )
    }

    static func sharedCacheImageEvidence(
        installName: String,
        loadCommandsSHA256: String
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
        return try SyntheticSharedCacheImageContentIdentityVerifier
            .derive(
                cacheSetEvidence: set,
                facts: SyntheticSharedCacheImageContentFacts(
                    installNameBytes: UInt64(name.count),
                    installNameBase64URL: base64URL(name),
                    machOUUID: String(repeating: "3", count: 32),
                    primaryCodeDirectory: .absent,
                    loadCommandsSHA256: loadCommandsSHA256
                )
            )
    }

    static func rootEvidence(
        role: ExecutableContentArtifactRole,
        loadCommands: [Data]
    ) throws -> ExecutableContentIdentityEvidence {
        let primary = codeDirectory(
            signingIdentifier:
                Data("com.example.graph.\(role.rawValue)".utf8)
        )
        let comparison = try SyntheticMachOIdentityParser.parse(
            machO(
                loadCommands: loadCommands,
                signatureRegion: superBlob(entries: [(0, primary)])
            )
        )
        return try ExecutableContentIdentityVerifier.derive(
            artifactRole: role,
            comparison: comparison
        )
    }

    static func executableEvidence(
        role: ExecutableContentArtifactRole,
        seed: UInt16
    ) throws -> ExecutableContentIdentityEvidence {
        let primary = codeDirectory(
            signingIdentifier: Data("com.example.graph.\(seed)".utf8)
        )
        let comparison = try SyntheticMachOIdentityParser.parse(
            machO(
                loadCommands: [],
                signatureRegion: superBlob(entries: [(0, primary)])
            )
        )
        return try ExecutableContentIdentityVerifier.derive(
            artifactRole: role,
            comparison: comparison
        )
    }

    static func machO(
        loadCommands extraCommands: [Data],
        signatureRegion: Data
    ) -> Data {
        var loadCommands = Data()
        appendUInt32LE(&loadCommands, 0x1b)
        appendUInt32LE(&loadCommands, 24)
        loadCommands.append(contentsOf: [
            0x00, 0x11, 0x22, 0x33,
            0x44, 0x55, 0x66, 0x77,
            0x88, 0x99, 0xaa, 0xbb,
            0xcc, 0xdd, 0xee, 0xff,
        ])
        for command in extraCommands {
            loadCommands.append(command)
        }

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
        appendUInt32LE(&result, UInt32(extraCommands.count + 2))
        appendUInt32LE(&result, UInt32(loadCommands.count))
        appendUInt32LE(&result, 0x00200085)
        appendUInt32LE(&result, 0)
        result.append(loadCommands)
        result.append(signatureRegion)
        return result
    }

    static func commandRegion(_ commands: [Data]) -> Data {
        var result = Data()
        for command in commands {
            result.append(command)
        }
        return result
    }

    static func dylibCommand(
        cmd: UInt32,
        name: String
    ) -> Data {
        var result = Data()
        appendUInt32LE(&result, cmd)
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
        writeUInt32LE(&result, at: 4, value: UInt32(result.count))
        return result
    }

    static func installName(
        _ value: String
    ) -> SyntheticRuntimeClosureInstallName {
        let data = Data(value.utf8)
        return SyntheticRuntimeClosureInstallName(
            bytes: UInt64(data.count),
            base64URL: base64URL(data)
        )
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
        writeUInt32BE(&result, at: 12, value: 0x2)
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
        result[36] = 32
        result[37] = 2
        writeUInt32BE(&result, at: 48, value: 0)
        return result
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

private enum TestFixtureFailure: Error {
    case unresolvedInstallName(String)
}

private extension Array
where Element == SyntheticRuntimeClosureEdgeRecordComparison {
    func index(
        parentContentEvidenceID: String,
        ordinal: UInt64
    ) -> Int {
        firstIndex {
            $0.parentContentEvidenceID == parentContentEvidenceID &&
                $0.loadCommandOrdinal == ordinal
        }!
    }
}

extension SyntheticRuntimeClosureGraphTests {
    struct FMBSharedCacheFixture {
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

    static func fmbCanonicalSharedCacheFixture()
        throws -> FMBSharedCacheFixture
    {
        let value = try canonicalFixture()
        return FMBSharedCacheFixture(
            root: value.root,
            members: value.members,
            edges: value.edges,
            collection: value.collection,
            rootInventory: value.rootInventory,
            memberInventories: value.memberInventories
        )
    }

    static func fmbRootEvidence(
        role: ExecutableContentArtifactRole,
        loadCommands: [Data]
    ) throws -> ExecutableContentIdentityEvidence {
        try rootEvidence(role: role, loadCommands: loadCommands)
    }

    static func fmbExecutableEvidence(
        role: ExecutableContentArtifactRole,
        seed: UInt16
    ) throws -> ExecutableContentIdentityEvidence {
        try executableEvidence(role: role, seed: seed)
    }

    static func fmbSharedCacheSnapshot(
        loadCommands: [Data],
        installName: String
    ) throws -> SyntheticSharedCacheImageLoadCommandSnapshot {
        try sharedCacheSnapshot(
            loadCommands: loadCommands,
            installName: installName
        )
    }
}
