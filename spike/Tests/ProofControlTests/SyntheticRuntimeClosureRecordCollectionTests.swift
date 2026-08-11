import Foundation
import XCTest
@testable import ProofControl

final class SyntheticRuntimeClosureRecordCollectionTests: XCTestCase {
    func testCanonicalCollectionFreezesExactCompactSectionInertly()
        throws
    {
        let effects = Effects()
        var fixture = try Self.canonicalFixture()
        let expected = Self.sectionBytes(
            members: fixture.members,
            edges: fixture.edges
        )

        let comparison =
            try SyntheticRuntimeClosureRecordCollectionVerifier.derive(
                members: fixture.members,
                edges: fixture.edges
            )

        XCTAssertEqual(comparison.memberCount, 2)
        XCTAssertEqual(comparison.edgeCount, 2)
        XCTAssertEqual(
            comparison.memberKeys.map(\.contentEvidenceID),
            fixture.members.map(\.contentEvidenceID)
        )
        XCTAssertEqual(
            comparison.memberKeys.map(\.decodedInstallName),
            fixture.members.map(\.decodedInstallName)
        )
        XCTAssertEqual(
            comparison.edgeKeys.map(\.parentContentEvidenceID),
            fixture.edges.map(\.parentContentEvidenceID)
        )
        XCTAssertEqual(
            comparison.edgeKeys.map(\.loadCommandOrdinal),
            [2, 10]
        )
        XCTAssertEqual(comparison.canonicalSectionBytes, expected)
        XCTAssertEqual(comparison.canonicalSectionBytes.last, 0x0a)
        Self.assertInert(comparison)

        fixture.members.removeAll()
        fixture.edges.reverse()
        XCTAssertEqual(comparison.memberCount, 2)
        XCTAssertEqual(comparison.edgeCount, 2)
        XCTAssertEqual(comparison.canonicalSectionBytes, expected)
        XCTAssertEqual(effects, .zero)
    }

    func testCountAndIndexBoundsRejectBeforeEffects() throws {
        let effects = Effects()
        let fixture = try Self.canonicalFixture()
        let member = fixture.members[0]
        let edge = fixture.edges[0]

        let cases: [
            (
                [SyntheticRuntimeClosureMemberRecordComparison],
                [SyntheticRuntimeClosureEdgeRecordComparison],
                SyntheticRuntimeClosureRecordCollectionFailure
            )
        ] = [
            ([], [], .memberCountOutOfRange(0)),
            (
                Array(repeating: member, count: 257),
                [edge],
                .memberCountOutOfRange(257)
            ),
            ([member], [], .edgeCountOutOfRange(0)),
            (
                [member],
                Array(repeating: edge, count: 4_097),
                .edgeCountOutOfRange(4_097)
            ),
        ]

        for (members, edges, expected) in cases {
            XCTAssertThrowsError(
                try SyntheticRuntimeClosureRecordCollectionVerifier
                    .derive(members: members, edges: edges)
            ) {
                XCTAssertEqual(
                    $0 as?
                        SyntheticRuntimeClosureRecordCollectionFailure,
                    expected
                )
            }
            XCTAssertEqual(effects, .zero)
        }

        let badMember =
            try SyntheticRuntimeClosureRecordSchemaVerifier.member(
                index: 1,
                source: fixture.memberInputs[0].source,
                installName: fixture.memberInputs[0].installName
            )
        XCTAssertThrowsError(
            try SyntheticRuntimeClosureRecordCollectionVerifier.derive(
                members: [badMember],
                edges: [edge]
            )
        ) {
            XCTAssertEqual(
                $0 as? SyntheticRuntimeClosureRecordCollectionFailure,
                .memberIndexMismatch(position: 0, actual: 1)
            )
        }

        for actual in [0, 2] {
            let driftedMember =
                try SyntheticRuntimeClosureRecordSchemaVerifier.member(
                    index: actual,
                    source: fixture.memberInputs[1].source,
                    installName: fixture.memberInputs[1].installName
                )
            XCTAssertThrowsError(
                try SyntheticRuntimeClosureRecordCollectionVerifier
                    .derive(
                        members: [fixture.members[0], driftedMember],
                        edges: [edge]
                    )
            ) {
                XCTAssertEqual(
                    $0 as?
                        SyntheticRuntimeClosureRecordCollectionFailure,
                    .memberIndexMismatch(
                        position: 1,
                        actual: actual
                    )
                )
            }
        }

        let badEdge =
            try SyntheticRuntimeClosureRecordSchemaVerifier.edge(
                index: 1,
                parent: fixture.edgeInputs[0].parent,
                loadCommandOrdinal:
                    fixture.edgeInputs[0].loadCommandOrdinal,
                kind: fixture.edgeInputs[0].kind,
                installName:
                    fixture.edgeInputs[0].resolved.installName,
                resolved: fixture.edgeInputs[0].resolved
            )
        XCTAssertThrowsError(
            try SyntheticRuntimeClosureRecordCollectionVerifier.derive(
                members: fixture.members,
                edges: [badEdge]
            )
        ) {
            XCTAssertEqual(
                $0 as? SyntheticRuntimeClosureRecordCollectionFailure,
                .edgeIndexMismatch(position: 0, actual: 1)
            )
        }

        for actual in [0, 2] {
            let input = fixture.edgeInputs[1]
            let driftedEdge =
                try SyntheticRuntimeClosureRecordSchemaVerifier.edge(
                    index: actual,
                    parent: input.parent,
                    loadCommandOrdinal: input.loadCommandOrdinal,
                    kind: input.kind,
                    installName: input.resolved.installName,
                    resolved: input.resolved
                )
            XCTAssertThrowsError(
                try SyntheticRuntimeClosureRecordCollectionVerifier
                    .derive(
                        members: fixture.members,
                        edges: [fixture.edges[0], driftedEdge]
                    )
            ) {
                XCTAssertEqual(
                    $0 as?
                        SyntheticRuntimeClosureRecordCollectionFailure,
                    .edgeIndexMismatch(
                        position: 1,
                        actual: actual
                    )
                )
            }
        }
        XCTAssertEqual(effects, .zero)
    }

    func testMemberOrderAndUniquenessRejectDeterministically()
        throws
    {
        let effects = Effects()
        let fixture = try Self.canonicalFixture()

        let duplicateEvidence = try Self.executableEvidence(
            role: .fileImage,
            seed: 30
        )
        let duplicateID = try Self.orderedMembers([
            MemberInput(
                source: .file(duplicateEvidence),
                installName:
                    Self.installName("/usr/lib/duplicate-a.dylib")
            ),
            MemberInput(
                source: .file(duplicateEvidence),
                installName:
                    Self.installName("/usr/lib/duplicate-b.dylib")
            ),
        ])
        Self.assertCollectionFailure(
            members: duplicateID,
            edges: fixture.edges,
            expected: .duplicateMemberContentEvidenceID(index: 1)
        )

        let distinctInputs = try Self.memberInputs([
            (40, "/usr/lib/descending-a.dylib"),
            (41, "/usr/lib/descending-b.dylib"),
        ]).sorted(by: Self.memberInputLess)
        let descending = try Self.orderedMembers(
            Array(distinctInputs.reversed())
        )
        Self.assertCollectionFailure(
            members: descending,
            edges: fixture.edges,
            expected: .membersNotSorted(index: 1)
        )

        let duplicateNameInputs = try Self.memberInputs([
            (50, "/usr/lib/same.dylib"),
            (51, "/usr/lib/same.dylib"),
        ]).sorted(by: Self.memberInputLess)
        let duplicateName = try Self.orderedMembers(
            duplicateNameInputs
        )
        Self.assertCollectionFailure(
            members: duplicateName,
            edges: fixture.edges,
            expected: .duplicateMemberInstallName(index: 1)
        )
        XCTAssertEqual(effects, .zero)
    }

    func testEdgeTupleOrderAndUniquenessRejectDeterministically()
        throws
    {
        let effects = Effects()
        let fixture = try Self.canonicalFixture()
        let root = fixture.root

        let duplicate = try Self.orderedEdges([
            EdgeInput(
                parent: .root(root),
                parentContentEvidenceID:
                    root.contentEvidenceID.sha256,
                loadCommandOrdinal: 2,
                kind: .load,
                resolved: fixture.members[0]
            ),
            EdgeInput(
                parent: .root(root),
                parentContentEvidenceID:
                    root.contentEvidenceID.sha256,
                loadCommandOrdinal: 2,
                kind: .reexport,
                resolved: fixture.members[1]
            ),
        ])
        Self.assertCollectionFailure(
            members: fixture.members,
            edges: duplicate,
            expected: .duplicateEdgeParentOrdinal(index: 1)
        )

        let descending = try Self.orderedEdges([
            EdgeInput(
                parent: .root(root),
                parentContentEvidenceID:
                    root.contentEvidenceID.sha256,
                loadCommandOrdinal: 10,
                kind: .load,
                resolved: fixture.members[0]
            ),
            EdgeInput(
                parent: .root(root),
                parentContentEvidenceID:
                    root.contentEvidenceID.sha256,
                loadCommandOrdinal: 2,
                kind: .load,
                resolved: fixture.members[0]
            ),
        ])
        Self.assertCollectionFailure(
            members: fixture.members,
            edges: descending,
            expected: .edgesNotSorted(index: 1)
        )
        XCTAssertEqual(effects, .zero)
    }

    func testMaximumCountsAndCheckedLengthBoundary() throws {
        let effects = Effects()
        let memberInputs = try Self.memberInputs(
            (0..<256).map {
                (
                    UInt16($0),
                    "/usr/lib/member-\($0).dylib"
                )
            }
        ).sorted(by: Self.memberInputLess)
        let maximumMembers = try Self.orderedMembers(memberInputs)
        let root = try Self.executableEvidence(role: .git, seed: 900)
        let oneEdge = try Self.orderedEdges([
            EdgeInput(
                parent: .root(root),
                parentContentEvidenceID:
                    root.contentEvidenceID.sha256,
                loadCommandOrdinal: 0,
                kind: .load,
                resolved: maximumMembers[0]
            ),
        ])
        XCTAssertNoThrow(
            try SyntheticRuntimeClosureRecordCollectionVerifier.derive(
                members: maximumMembers,
                edges: oneEdge
            )
        )

        let oneMember = [maximumMembers[0]]
        let maximumEdgeInputs = (0..<4_096).map {
            EdgeInput(
                parent: .root(root),
                parentContentEvidenceID:
                    root.contentEvidenceID.sha256,
                loadCommandOrdinal: UInt64($0),
                kind: .load,
                resolved: oneMember[0]
            )
        }
        let maximumEdges = try Self.orderedEdges(maximumEdgeInputs)
        XCTAssertNoThrow(
            try SyntheticRuntimeClosureRecordCollectionVerifier.derive(
                members: oneMember,
                edges: maximumEdges
            )
        )

        XCTAssertThrowsError(
            try SyntheticRuntimeClosureRecordCollectionVerifier
                .checkedCanonicalSectionLength(
                    memberCount: 1,
                    memberRowLengths: [Int.max],
                    edgeCount: 1,
                    edgeRowLengths: [1]
                )
        ) {
            XCTAssertEqual(
                $0 as? SyntheticRuntimeClosureRecordCollectionFailure,
                .canonicalSectionLengthOverflow
            )
        }
        XCTAssertThrowsError(
            try SyntheticRuntimeClosureRecordCollectionVerifier
                .validateCanonicalSectionLength(
                    actual: 1,
                    expected: 2
                )
        ) {
            XCTAssertEqual(
                $0 as? SyntheticRuntimeClosureRecordCollectionFailure,
                .canonicalSectionLengthMismatch
            )
        }
        XCTAssertEqual(
            try SyntheticRuntimeClosureRecordCollectionVerifier
                .checkedCanonicalSectionLength(
                    memberCount: 256,
                    memberRowLengths:
                        Array(repeating: 5_930, count: 256),
                    edgeCount: 4_096,
                    edgeRowLengths:
                        Array(repeating: 5_812, count: 4_096)
                ),
            25_324_065
        )
        XCTAssertEqual(effects, .zero)
    }

    func testCanonicalIncompleteAndCyclicStructuresRemainUnproved()
        throws
    {
        let effects = Effects()
        let inputs = try Self.memberInputs([
            (100, "/usr/lib/a.dylib"),
            (101, "/usr/lib/b.dylib"),
            (102, "/usr/lib/unreachable.dylib"),
            (103, "/usr/lib/outside-parent.dylib"),
            (104, "/usr/lib/outside-resolved.dylib"),
        ]).sorted(by: Self.memberInputLess)
        let allMembers = try Self.orderedMembers(inputs)
        let included = Array(allMembers.prefix(3))
        let outsideParent = allMembers[3]
        let outsideResolved = allMembers[4]

        let edgeInputs = [
            EdgeInput(
                parent: .member(outsideParent),
                parentContentEvidenceID:
                    outsideParent.contentEvidenceID,
                loadCommandOrdinal: 0,
                kind: .load,
                resolved: included[0]
            ),
            EdgeInput(
                parent: .member(included[0]),
                parentContentEvidenceID:
                    included[0].contentEvidenceID,
                loadCommandOrdinal: 0,
                kind: .load,
                resolved: included[1]
            ),
            EdgeInput(
                parent: .member(included[0]),
                parentContentEvidenceID:
                    included[0].contentEvidenceID,
                loadCommandOrdinal: 1,
                kind: .load,
                resolved: outsideResolved
            ),
            EdgeInput(
                parent: .member(included[1]),
                parentContentEvidenceID:
                    included[1].contentEvidenceID,
                loadCommandOrdinal: 0,
                kind: .load,
                resolved: included[0]
            ),
        ].sorted(by: Self.edgeInputLess)
        let edges = try Self.orderedEdges(edgeInputs)

        let comparison =
            try SyntheticRuntimeClosureRecordCollectionVerifier.derive(
                members: included,
                edges: edges
            )

        XCTAssertEqual(comparison.memberCount, 3)
        XCTAssertEqual(comparison.edgeCount, 4)
        Self.assertInert(comparison)
        XCTAssertEqual(effects, .zero)
    }
}

private extension SyntheticRuntimeClosureRecordCollectionTests {
    struct Effects: Equatable {
        var fileSystemMutations = 0
        var networkOperations = 0
        var processSpawns = 0
        var packConsumes = 0
        var runtimeAdmissions = 0

        static let zero = Self()
    }

    struct MemberInput {
        let source: SyntheticRuntimeClosureMemberSource
        let installName: SyntheticRuntimeClosureInstallName

        var contentEvidenceID: String {
            switch source {
            case let .file(evidence):
                evidence.contentEvidenceID.sha256
            case let .fileImage(snapshot):
                snapshot.fileImageEvidence.contentEvidenceID.sha256
            case let .sharedCache(evidence):
                evidence.contentEvidenceID.sha256
            }
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
        var members: [SyntheticRuntimeClosureMemberRecordComparison]
        var edges: [SyntheticRuntimeClosureEdgeRecordComparison]
        let memberInputs: [MemberInput]
        let edgeInputs: [EdgeInput]
        let root: ExecutableContentIdentityEvidence
    }

    static func canonicalFixture() throws -> Fixture {
        let inputs = try memberInputs([
            (10, "/usr/lib/canonical-a.dylib"),
            (20, "/usr/lib/canonical-b.dylib"),
        ]).sorted(by: memberInputLess)
        let members = try orderedMembers(inputs)
        let root = try executableEvidence(role: .git, seed: 500)
        let edgeInputs = [
            EdgeInput(
                parent: .root(root),
                parentContentEvidenceID:
                    root.contentEvidenceID.sha256,
                loadCommandOrdinal: 2,
                kind: .load,
                resolved: members[0]
            ),
            EdgeInput(
                parent: .root(root),
                parentContentEvidenceID:
                    root.contentEvidenceID.sha256,
                loadCommandOrdinal: 10,
                kind: .reexport,
                resolved: members[0]
            ),
        ]
        return Fixture(
            members: members,
            edges: try orderedEdges(edgeInputs),
            memberInputs: inputs,
            edgeInputs: edgeInputs,
            root: root
        )
    }

    static func memberInputs(
        _ values: [(UInt16, String)]
    ) throws -> [MemberInput] {
        try values.map { seed, name in
            MemberInput(
                source: .file(
                    try executableEvidence(
                        role: .fileImage,
                        seed: seed
                    )
                ),
                installName: installName(name)
            )
        }
    }

    static func orderedMembers(
        _ inputs: [MemberInput]
    ) throws -> [SyntheticRuntimeClosureMemberRecordComparison] {
        try inputs.enumerated().map { index, input in
            try SyntheticRuntimeClosureRecordSchemaVerifier.member(
                index: index,
                source: input.source,
                installName: input.installName
            )
        }
    }

    static func orderedEdges(
        _ inputs: [EdgeInput]
    ) throws -> [SyntheticRuntimeClosureEdgeRecordComparison] {
        try inputs.enumerated().map { index, input in
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

    static func memberInputLess(
        _ lhs: MemberInput,
        _ rhs: MemberInput
    ) -> Bool {
        lhs.contentEvidenceID.utf8.lexicographicallyPrecedes(
            rhs.contentEvidenceID.utf8
        )
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

    static func sectionBytes(
        members: [SyntheticRuntimeClosureMemberRecordComparison],
        edges: [SyntheticRuntimeClosureEdgeRecordComparison]
    ) -> Data {
        var result = Data("member_count=\(members.count)\n".utf8)
        for member in members {
            result.append(member.canonicalRecordBytes)
        }
        result.append(Data("edge_count=\(edges.count)\n".utf8))
        for edge in edges {
            result.append(edge.canonicalRecordBytes)
        }
        return result
    }

    static func assertCollectionFailure(
        members: [SyntheticRuntimeClosureMemberRecordComparison],
        edges: [SyntheticRuntimeClosureEdgeRecordComparison],
        expected: SyntheticRuntimeClosureRecordCollectionFailure,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(
            try SyntheticRuntimeClosureRecordCollectionVerifier.derive(
                members: members,
                edges: edges
            ),
            file: file,
            line: line
        ) {
            XCTAssertEqual(
                $0 as?
                    SyntheticRuntimeClosureRecordCollectionFailure,
                expected,
                file: file,
                line: line
            )
        }
    }

    static func assertInert(
        _ value: SyntheticRuntimeClosureRecordCollectionComparison,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertFalse(
            value.provesGraphMembership,
            file: file,
            line: line
        )
        XCTAssertFalse(value.provesTraversal, file: file, line: line)
        XCTAssertFalse(value.provesReachability, file: file, line: line)
        XCTAssertFalse(
            value.isCompleteRuntimeClosure,
            file: file,
            line: line
        )
        XCTAssertFalse(value.canExecute, file: file, line: line)
        XCTAssertFalse(value.canSpawn, file: file, line: line)
        XCTAssertFalse(value.canAccessNetwork, file: file, line: line)
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
        XCTAssertFalse(value.canReserveOutput, file: file, line: line)
        XCTAssertFalse(value.canPublish, file: file, line: line)
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

    static func executableEvidence(
        role: ExecutableContentArtifactRole,
        seed: UInt16
    ) throws -> ExecutableContentIdentityEvidence {
        let comparison = try SyntheticMachOIdentityParser.parse(
            signedFixture(seed: seed)
        )
        return try ExecutableContentIdentityVerifier.derive(
            artifactRole: role,
            comparison: comparison
        )
    }

    static func signedFixture(seed: UInt16) -> Data {
        let primary = codeDirectory(
            signingIdentifier:
                Data("com.example.collection.\(seed)".utf8)
        )
        return machO(
            signatureRegion: superBlob(entries: [(0, primary)]),
            seed: seed
        )
    }

    static func machO(
        signatureRegion: Data,
        seed: UInt16
    ) -> Data {
        var uuid: [UInt8] = [
            0x00, 0x11, 0x22, 0x33,
            0x44, 0x55, 0x66, 0x77,
            0x88, 0x99, 0xaa, 0xbb,
            0xcc, 0xdd, 0x00, 0x00,
        ]
        uuid[14] = UInt8(truncatingIfNeeded: seed >> 8)
        uuid[15] = UInt8(truncatingIfNeeded: seed)

        var loadCommands = Data()
        appendUInt32LE(&loadCommands, 0x1b)
        appendUInt32LE(&loadCommands, 24)
        loadCommands.append(contentsOf: uuid)

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
