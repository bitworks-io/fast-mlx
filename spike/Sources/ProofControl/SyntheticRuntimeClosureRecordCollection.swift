import Foundation

enum SyntheticRuntimeClosureRecordCollectionFailure:
    Error,
    Equatable
{
    case memberCountOutOfRange(Int)
    case edgeCountOutOfRange(Int)
    case memberIndexMismatch(position: Int, actual: Int)
    case duplicateMemberContentEvidenceID(index: Int)
    case membersNotSorted(index: Int)
    case duplicateMemberInstallName(index: Int)
    case edgeIndexMismatch(position: Int, actual: Int)
    case duplicateEdgeParentOrdinal(index: Int)
    case edgesNotSorted(index: Int)
    case canonicalSectionLengthOverflow
    case canonicalSectionLengthMismatch
}

struct SyntheticRuntimeClosureMemberCollectionKey:
    Equatable,
    Sendable
{
    let contentEvidenceID: String
    let decodedInstallName: Data
}

struct SyntheticRuntimeClosureEdgeCollectionKey:
    Equatable,
    Sendable
{
    let parentContentEvidenceID: String
    let loadCommandOrdinal: UInt64
}

fileprivate enum SyntheticRuntimeClosureRecordCollectionConstructionSeal:
    Equatable
{
    case verified
}

struct SyntheticRuntimeClosureRecordCollectionComparison: Equatable {
    let memberCount: Int
    let edgeCount: Int
    let memberKeys: [SyntheticRuntimeClosureMemberCollectionKey]
    let edgeKeys: [SyntheticRuntimeClosureEdgeCollectionKey]
    let canonicalSectionBytes: Data
    fileprivate let constructionSeal:
        SyntheticRuntimeClosureRecordCollectionConstructionSeal

    let provesGraphMembership = false
    let provesTraversal = false
    let provesReachability = false
    let isCompleteRuntimeClosure = false

    let canExecute = false
    let canSpawn = false
    let canAccessNetwork = false
    let canConsumePack = false
    let canMutateFileSystem = false
    let canImportGitObjects = false
    let canBuild = false
    let canLoadModel = false
    let canReserveOutput = false
    let canPublish = false

    fileprivate init(
        memberCount: Int,
        edgeCount: Int,
        memberKeys: [SyntheticRuntimeClosureMemberCollectionKey],
        edgeKeys: [SyntheticRuntimeClosureEdgeCollectionKey],
        canonicalSectionBytes: Data
    ) {
        self.memberCount = memberCount
        self.edgeCount = edgeCount
        self.memberKeys = memberKeys
        self.edgeKeys = edgeKeys
        self.canonicalSectionBytes = canonicalSectionBytes
        self.constructionSeal = .verified
    }
}

enum SyntheticRuntimeClosureRecordCollectionVerifier {
    static let maximumMemberCount = 256
    static let maximumEdgeCount = 4_096
    static let maximumCanonicalSectionBytes = 25_324_065

    static func derive(
        members: [SyntheticRuntimeClosureMemberRecordComparison],
        edges: [SyntheticRuntimeClosureEdgeRecordComparison]
    ) throws -> SyntheticRuntimeClosureRecordCollectionComparison {
        guard (1...maximumMemberCount).contains(members.count) else {
            throw SyntheticRuntimeClosureRecordCollectionFailure
                .memberCountOutOfRange(members.count)
        }
        guard (1...maximumEdgeCount).contains(edges.count) else {
            throw SyntheticRuntimeClosureRecordCollectionFailure
                .edgeCountOutOfRange(edges.count)
        }

        var memberKeys: [
            SyntheticRuntimeClosureMemberCollectionKey
        ] = []
        memberKeys.reserveCapacity(members.count)
        var memberIDs = Set<String>()
        memberIDs.reserveCapacity(members.count)
        var memberInstallNames = Set<Data>()
        memberInstallNames.reserveCapacity(members.count)
        var previousMemberID: String?

        for (position, record) in members.enumerated() {
            guard record.index == position else {
                throw SyntheticRuntimeClosureRecordCollectionFailure
                    .memberIndexMismatch(
                        position: position,
                        actual: record.index
                    )
            }
            guard memberIDs.insert(record.contentEvidenceID).inserted
            else {
                throw SyntheticRuntimeClosureRecordCollectionFailure
                    .duplicateMemberContentEvidenceID(
                        index: position
                    )
            }
            if let previousMemberID,
               !asciiLess(previousMemberID, record.contentEvidenceID)
            {
                throw SyntheticRuntimeClosureRecordCollectionFailure
                    .membersNotSorted(index: position)
            }
            guard
                memberInstallNames
                    .insert(record.decodedInstallName)
                    .inserted
            else {
                throw SyntheticRuntimeClosureRecordCollectionFailure
                    .duplicateMemberInstallName(index: position)
            }
            memberKeys.append(
                SyntheticRuntimeClosureMemberCollectionKey(
                    contentEvidenceID: record.contentEvidenceID,
                    decodedInstallName: record.decodedInstallName
                )
            )
            previousMemberID = record.contentEvidenceID
        }

        var edgeKeys: [
            SyntheticRuntimeClosureEdgeCollectionKey
        ] = []
        edgeKeys.reserveCapacity(edges.count)
        var edgePairs = Set<EdgePair>()
        edgePairs.reserveCapacity(edges.count)
        var previousEdgePair: EdgePair?

        for (position, record) in edges.enumerated() {
            guard record.index == position else {
                throw SyntheticRuntimeClosureRecordCollectionFailure
                    .edgeIndexMismatch(
                        position: position,
                        actual: record.index
                    )
            }
            let pair = EdgePair(
                parentContentEvidenceID:
                    record.parentContentEvidenceID,
                loadCommandOrdinal: record.loadCommandOrdinal
            )
            guard edgePairs.insert(pair).inserted else {
                throw SyntheticRuntimeClosureRecordCollectionFailure
                    .duplicateEdgeParentOrdinal(index: position)
            }
            if let previousEdgePair,
               !edgePairLess(previousEdgePair, pair)
            {
                throw SyntheticRuntimeClosureRecordCollectionFailure
                    .edgesNotSorted(index: position)
            }
            edgeKeys.append(
                SyntheticRuntimeClosureEdgeCollectionKey(
                    parentContentEvidenceID:
                        record.parentContentEvidenceID,
                    loadCommandOrdinal:
                        record.loadCommandOrdinal
                )
            )
            previousEdgePair = pair
        }

        let memberHeader = Data(
            "member_count=\(members.count)\n".utf8
        )
        let edgeHeader = Data("edge_count=\(edges.count)\n".utf8)
        let expectedLength = try checkedCanonicalSectionLength(
            memberCount: members.count,
            memberRowLengths:
                members.map(\.canonicalRecordBytes.count),
            edgeCount: edges.count,
            edgeRowLengths:
                edges.map(\.canonicalRecordBytes.count)
        )

        var section = Data()
        section.reserveCapacity(expectedLength)
        section.append(memberHeader)
        for member in members {
            section.append(member.canonicalRecordBytes)
        }
        section.append(edgeHeader)
        for edge in edges {
            section.append(edge.canonicalRecordBytes)
        }
        try validateCanonicalSectionLength(
            actual: section.count,
            expected: expectedLength
        )

        return SyntheticRuntimeClosureRecordCollectionComparison(
            memberCount: members.count,
            edgeCount: edges.count,
            memberKeys: memberKeys,
            edgeKeys: edgeKeys,
            canonicalSectionBytes: section
        )
    }

    static func checkedCanonicalSectionLength(
        memberCount: Int,
        memberRowLengths: [Int],
        edgeCount: Int,
        edgeRowLengths: [Int]
    ) throws -> Int {
        try checkedCanonicalSectionLength(
            memberHeaderLength:
                Data("member_count=\(memberCount)\n".utf8).count,
            memberRowLengths: memberRowLengths,
            edgeHeaderLength:
                Data("edge_count=\(edgeCount)\n".utf8).count,
            edgeRowLengths: edgeRowLengths
        )
    }

    static func validateCanonicalSectionLength(
        actual: Int,
        expected: Int
    ) throws {
        guard actual == expected else {
            throw SyntheticRuntimeClosureRecordCollectionFailure
                .canonicalSectionLengthMismatch
        }
    }
}

private extension SyntheticRuntimeClosureRecordCollectionVerifier {
    struct EdgePair: Hashable {
        let parentContentEvidenceID: String
        let loadCommandOrdinal: UInt64
    }

    static func asciiLess(_ lhs: String, _ rhs: String) -> Bool {
        lhs.utf8.lexicographicallyPrecedes(rhs.utf8)
    }

    static func edgePairLess(
        _ lhs: EdgePair,
        _ rhs: EdgePair
    ) -> Bool {
        if lhs.parentContentEvidenceID != rhs.parentContentEvidenceID {
            return asciiLess(
                lhs.parentContentEvidenceID,
                rhs.parentContentEvidenceID
            )
        }
        return lhs.loadCommandOrdinal < rhs.loadCommandOrdinal
    }

    static func checkedCanonicalSectionLength(
        memberHeaderLength: Int,
        memberRowLengths: [Int],
        edgeHeaderLength: Int,
        edgeRowLengths: [Int]
    ) throws -> Int {
        var length = 0
        for value in [memberHeaderLength] +
            memberRowLengths +
            [edgeHeaderLength] +
            edgeRowLengths
        {
            guard value >= 0 else {
                throw SyntheticRuntimeClosureRecordCollectionFailure
                    .canonicalSectionLengthOverflow
            }
            let (next, overflow) = length.addingReportingOverflow(
                value
            )
            guard
                !overflow,
                next <= maximumCanonicalSectionBytes
            else {
                throw SyntheticRuntimeClosureRecordCollectionFailure
                    .canonicalSectionLengthOverflow
            }
            length = next
        }
        return length
    }
}
