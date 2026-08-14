import Foundation

fileprivate enum SyntheticRuntimeClosureGraphConstructionSeal:
    Equatable
{
    case verified
}

fileprivate enum
    SyntheticFileImageRuntimeClosureGraphConstructionSeal:
        Equatable
{
    case verified
}

enum SyntheticRuntimeClosureGraphFailure: Error, Equatable {
    case rootRole(ExecutableContentArtifactRole)
    case rootEvidenceMismatch
    case rootInventoryMismatch
    case collectionMismatch
    case rootMemberCollision
    case unsupportedFileMemberIdentity(memberIndex: Int)
    case memberInventoryCountMismatch
    case duplicateMemberInventory(memberIndex: Int)
    case missingMemberInventory(memberIndex: Int)
    case foreignMemberInventory(inventoryIndex: Int)
    case memberInventoryEvidenceMismatch(memberIndex: Int)
    case edgeParentMissing(edgeIndex: Int)
    case edgeParentEvidenceMismatch(edgeIndex: Int)
    case edgeResolvedMissing(edgeIndex: Int)
    case edgeResolvedEvidenceMismatch(edgeIndex: Int)
    case edgeInventoryEntryMissing(edgeIndex: Int)
    case edgeInventoryKindMismatch(edgeIndex: Int)
    case edgeInventoryInstallNameMismatch(edgeIndex: Int)
    case unresolvedInventoryEntry(
        parentIndex: Int,
        ordinal: UInt64
    )
    case unconsumedInventoryEntry(
        parentIndex: Int,
        ordinal: UInt64
    )
    case unreachableMember(memberIndex: Int)
}

enum SyntheticRuntimeClosureRuntimeDecision:
    String,
    Equatable,
    Sendable
{
    case noGo = "no-go"
}

struct SyntheticRuntimeClosureInventorySummary:
    Equatable,
    Sendable
{
    let parentContentEvidenceID: String
    let sourceLoadCommandsSHA256: String
    let acceptedCommandCount: Int
    fileprivate let constructionSeal:
        SyntheticRuntimeClosureGraphConstructionSeal

    fileprivate init(
        parentContentEvidenceID: String,
        sourceLoadCommandsSHA256: String,
        acceptedCommandCount: Int
    ) {
        self.parentContentEvidenceID = parentContentEvidenceID
        self.sourceLoadCommandsSHA256 = sourceLoadCommandsSHA256
        self.acceptedCommandCount = acceptedCommandCount
        self.constructionSeal = .verified
    }
}

struct SyntheticRuntimeClosureValidatedEdgeProjection:
    Equatable,
    Sendable
{
    let parentContentEvidenceID: String
    let loadCommandOrdinal: UInt64
    let kind: SyntheticRuntimeClosureEdgeKind
    let decodedInstallName: Data
    let resolvedContentEvidenceID: String
    fileprivate let constructionSeal:
        SyntheticRuntimeClosureGraphConstructionSeal

    fileprivate init(
        parentContentEvidenceID: String,
        loadCommandOrdinal: UInt64,
        kind: SyntheticRuntimeClosureEdgeKind,
        decodedInstallName: Data,
        resolvedContentEvidenceID: String
    ) {
        self.parentContentEvidenceID = parentContentEvidenceID
        self.loadCommandOrdinal = loadCommandOrdinal
        self.kind = kind
        self.decodedInstallName = Data(decodedInstallName)
        self.resolvedContentEvidenceID = resolvedContentEvidenceID
        self.constructionSeal = .verified
    }
}

/// Exact comparison evidence for one bounded declared static dependency
/// graph. This does not describe or authorize a dyld runtime closure.
struct SyntheticRuntimeClosureGraphComparison: Equatable {
    let rootContentEvidenceID: String
    let rootRole: ExecutableContentArtifactRole
    let collection: SyntheticRuntimeClosureRecordCollectionComparison
    let inventorySummaries: [SyntheticRuntimeClosureInventorySummary]
    let validatedEdges:
        [SyntheticRuntimeClosureValidatedEdgeProjection]
    let reachableMemberContentEvidenceIDs: [String]
    fileprivate let constructionSeal:
        SyntheticRuntimeClosureGraphConstructionSeal

    let provesGraphMembership = true
    let provesAcceptedCommandCompleteness = true
    let provesBoundedTraversal = true
    let provesRootReachability = true

    let isCompleteRuntimeClosure = false
    let provesRuntimeLaunchability = false
    let runtimeDecision = SyntheticRuntimeClosureRuntimeDecision.noGo

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
        rootContentEvidenceID: String,
        rootRole: ExecutableContentArtifactRole,
        collection:
            SyntheticRuntimeClosureRecordCollectionComparison,
        inventorySummaries:
            [SyntheticRuntimeClosureInventorySummary],
        validatedEdges:
            [SyntheticRuntimeClosureValidatedEdgeProjection],
        reachableMemberContentEvidenceIDs: [String]
    ) {
        self.rootContentEvidenceID = rootContentEvidenceID
        self.rootRole = rootRole
        self.collection = collection
        self.inventorySummaries = inventorySummaries
        self.validatedEdges = validatedEdges
        self.reachableMemberContentEvidenceIDs =
            reachableMemberContentEvidenceIDs
        self.constructionSeal = .verified
    }
}

/// Versioned comparison evidence for one bounded declared static dependency
/// graph whose members may include sealed synthetic D2 file images. This is
/// not convertible to the shared-cache-only v1 result and grants no runtime
/// authority.
struct SyntheticFileImageRuntimeClosureGraphComparison: Equatable {
    let rootContentEvidenceID: String
    let rootRole: ExecutableContentArtifactRole
    let collection: SyntheticRuntimeClosureRecordCollectionComparison
    let inventorySummaries: [SyntheticRuntimeClosureInventorySummary]
    let validatedEdges:
        [SyntheticRuntimeClosureValidatedEdgeProjection]
    let reachableMemberContentEvidenceIDs: [String]
    let fileImageMemberCount: Int
    fileprivate let constructionSeal:
        SyntheticFileImageRuntimeClosureGraphConstructionSeal

    let provesGraphMembership = true
    let provesAcceptedCommandCompleteness = true
    let provesBoundedTraversal = true
    let provesRootReachability = true
    let provesSealedFileImageContinuity = true

    let isCompleteRuntimeClosure = false
    let provesRuntimeLaunchability = false
    let runtimeDecision = SyntheticRuntimeClosureRuntimeDecision.noGo

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
        rootContentEvidenceID: String,
        rootRole: ExecutableContentArtifactRole,
        collection:
            SyntheticRuntimeClosureRecordCollectionComparison,
        inventorySummaries:
            [SyntheticRuntimeClosureInventorySummary],
        validatedEdges:
            [SyntheticRuntimeClosureValidatedEdgeProjection],
        reachableMemberContentEvidenceIDs: [String],
        fileImageMemberCount: Int
    ) {
        self.rootContentEvidenceID = rootContentEvidenceID
        self.rootRole = rootRole
        self.collection = collection
        self.inventorySummaries = inventorySummaries
        self.validatedEdges = validatedEdges
        self.reachableMemberContentEvidenceIDs =
            reachableMemberContentEvidenceIDs
        self.fileImageMemberCount = fileImageMemberCount
        self.constructionSeal = .verified
    }
}

enum SyntheticRuntimeClosureGraphVerifier {
    static func compare(
        root: ExecutableContentIdentityEvidence,
        members: [SyntheticRuntimeClosureMemberRecordComparison],
        edges: [SyntheticRuntimeClosureEdgeRecordComparison],
        collection:
            SyntheticRuntimeClosureRecordCollectionComparison,
        rootInventory:
            SyntheticAcceptedDependencyCommandInventoryComparison,
        memberInventories:
            [SyntheticAcceptedDependencyCommandInventoryComparison]
    ) throws -> SyntheticRuntimeClosureGraphComparison {
        let projection = try SyntheticRuntimeClosureGraphKernel.compare(
            root: root,
            members: members,
            edges: edges,
            collection: collection,
            rootInventory: rootInventory,
            memberInventories: memberInventories,
            memberPolicy: .sharedCacheOnly
        )
        return SyntheticRuntimeClosureGraphComparison(
            rootContentEvidenceID: projection.rootContentEvidenceID,
            rootRole: projection.rootRole,
            collection: projection.collection,
            inventorySummaries: projection.inventorySummaries,
            validatedEdges: projection.validatedEdges,
            reachableMemberContentEvidenceIDs:
                projection.reachableMemberContentEvidenceIDs
        )
    }
}

enum SyntheticFileImageRuntimeClosureGraphVerifier {
    static func compare(
        root: ExecutableContentIdentityEvidence,
        members: [SyntheticRuntimeClosureMemberRecordComparison],
        edges: [SyntheticRuntimeClosureEdgeRecordComparison],
        collection:
            SyntheticRuntimeClosureRecordCollectionComparison,
        rootInventory:
            SyntheticAcceptedDependencyCommandInventoryComparison,
        memberInventories:
            [SyntheticAcceptedDependencyCommandInventoryComparison]
    ) throws -> SyntheticFileImageRuntimeClosureGraphComparison {
        let projection = try SyntheticRuntimeClosureGraphKernel.compare(
            root: root,
            members: members,
            edges: edges,
            collection: collection,
            rootInventory: rootInventory,
            memberInventories: memberInventories,
            memberPolicy: .sharedCacheAndSealedFileImage
        )
        return SyntheticFileImageRuntimeClosureGraphComparison(
            rootContentEvidenceID: projection.rootContentEvidenceID,
            rootRole: projection.rootRole,
            collection: projection.collection,
            inventorySummaries: projection.inventorySummaries,
            validatedEdges: projection.validatedEdges,
            reachableMemberContentEvidenceIDs:
                projection.reachableMemberContentEvidenceIDs,
            fileImageMemberCount: projection.fileImageMemberCount
        )
    }
}

private enum SyntheticRuntimeClosureGraphMemberPolicy {
    case sharedCacheOnly
    case sharedCacheAndSealedFileImage
}

private struct SyntheticRuntimeClosureGraphKernelProjection {
    let rootContentEvidenceID: String
    let rootRole: ExecutableContentArtifactRole
    let collection: SyntheticRuntimeClosureRecordCollectionComparison
    let inventorySummaries: [SyntheticRuntimeClosureInventorySummary]
    let validatedEdges:
        [SyntheticRuntimeClosureValidatedEdgeProjection]
    let reachableMemberContentEvidenceIDs: [String]
    let fileImageMemberCount: Int
}

private enum SyntheticRuntimeClosureGraphKernel {
    static func compare(
        root: ExecutableContentIdentityEvidence,
        members: [SyntheticRuntimeClosureMemberRecordComparison],
        edges: [SyntheticRuntimeClosureEdgeRecordComparison],
        collection:
            SyntheticRuntimeClosureRecordCollectionComparison,
        rootInventory:
            SyntheticAcceptedDependencyCommandInventoryComparison,
        memberInventories:
            [SyntheticAcceptedDependencyCommandInventoryComparison],
        memberPolicy: SyntheticRuntimeClosureGraphMemberPolicy
    ) throws -> SyntheticRuntimeClosureGraphKernelProjection {
        try validateRoot(root)
        try validateRootInventory(rootInventory, root: root)

        let rederivedCollection =
            try SyntheticRuntimeClosureRecordCollectionVerifier
            .derive(members: members, edges: edges)
        guard rederivedCollection == collection else {
            throw SyntheticRuntimeClosureGraphFailure.collectionMismatch
        }

        let rootID = root.contentEvidenceID.sha256
        guard !members.contains(where: {
            $0.contentEvidenceID == rootID
        }) else {
            throw SyntheticRuntimeClosureGraphFailure
                .rootMemberCollision
        }

        var fileImageMemberCount = 0
        for (memberIndex, member) in members.enumerated() {
            switch member.source {
            case .file:
                throw SyntheticRuntimeClosureGraphFailure
                    .unsupportedFileMemberIdentity(
                        memberIndex: memberIndex
                    )
            case .fileImage:
                guard memberPolicy ==
                        .sharedCacheAndSealedFileImage
                else {
                    throw SyntheticRuntimeClosureGraphFailure
                        .unsupportedFileMemberIdentity(
                            memberIndex: memberIndex
                        )
                }
                fileImageMemberCount += 1
            case .sharedCache:
                break
            }
        }

        let memberState = try validateMemberInventories(
            members: members,
            memberInventories: memberInventories,
            memberPolicy: memberPolicy
        )
        let orderedInventories = orderedInventoryBindings(
            rootInventory: rootInventory,
            memberInventoriesByMemberIndex:
                memberState.inventoriesByMemberIndex
        )
        let resolvedEntries = try resolveInventoryEntries(
            orderedInventories,
            memberByInstallName: memberState.memberByInstallName
        )

        try validateEdgeParents(
            edges,
            root: root,
            memberByID: memberState.memberByID
        )
        try validateEdgeResolvedMembers(
            edges,
            memberByID: memberState.memberByID
        )
        let edgeState = try validateEdgesAgainstInventories(
            edges,
            resolvedEntries: resolvedEntries
        )
        try validateAllInventoryEntriesConsumed(
            orderedInventories,
            consumed: edgeState.consumed
        )

        let reachableIDs = try reachableMemberIDs(
            rootID: rootID,
            members: members,
            adjacency: edgeState.adjacency
        )
        let summaries = orderedInventories.map {
            SyntheticRuntimeClosureInventorySummary(
                parentContentEvidenceID:
                    $0.inventory.parentContentEvidenceID,
                sourceLoadCommandsSHA256:
                    $0.inventory.sourceLoadCommandsSHA256,
                acceptedCommandCount: $0.inventory.entries.count
            )
        }

        return SyntheticRuntimeClosureGraphKernelProjection(
            rootContentEvidenceID: rootID,
            rootRole: root.artifactRole,
            collection: collection,
            inventorySummaries: summaries,
            validatedEdges: edgeState.projections,
            reachableMemberContentEvidenceIDs: reachableIDs,
            fileImageMemberCount: fileImageMemberCount
        )
    }
}

private extension SyntheticRuntimeClosureGraphKernel {
    static let maximumAcceptedEntryCount = 4_096

    struct MemberState {
        let memberByID:
            [String: SyntheticRuntimeClosureMemberRecordComparison]
        let memberByInstallName:
            [Data: SyntheticRuntimeClosureMemberRecordComparison]
        let inventoriesByMemberIndex:
            [Int: SyntheticAcceptedDependencyCommandInventoryComparison]
    }

    struct InventoryBinding {
        let inventory:
            SyntheticAcceptedDependencyCommandInventoryComparison
    }

    struct InventoryKey: Hashable {
        let parentContentEvidenceID: String
        let loadCommandOrdinal: UInt64
    }

    struct ResolvedInventoryEntry {
        let parentIndex: Int
        let entry: SyntheticAcceptedDependencyCommand
        let resolvedMember:
            SyntheticRuntimeClosureMemberRecordComparison
    }

    struct EdgeState {
        let consumed: Set<InventoryKey>
        let projections:
            [SyntheticRuntimeClosureValidatedEdgeProjection]
        let adjacency: [String: [String]]
    }

    struct InventoryFailureCandidate {
        let locus: Int
        let rank: Int
        let failure: SyntheticRuntimeClosureGraphFailure
    }

    static func validateRoot(
        _ root: ExecutableContentIdentityEvidence
    ) throws {
        guard root.artifactRole == .git ||
                root.artifactRole == .selfGuard
        else {
            throw SyntheticRuntimeClosureGraphFailure
                .rootRole(root.artifactRole)
        }
        let rederived: ExecutableContentIdentityEvidence
        do {
            rederived = try ExecutableContentIdentityVerifier.derive(
                artifactRole: root.artifactRole,
                comparison: root.comparison
            )
        } catch {
            throw SyntheticRuntimeClosureGraphFailure
                .rootEvidenceMismatch
        }
        guard rederived == root else {
            throw SyntheticRuntimeClosureGraphFailure
                .rootEvidenceMismatch
        }
    }

    static func validateRootInventory(
        _ inventory:
            SyntheticAcceptedDependencyCommandInventoryComparison,
        root: ExecutableContentIdentityEvidence
    ) throws {
        guard case let .root(inventoryRoot) = inventory.source,
              inventoryRoot == root
        else {
            throw SyntheticRuntimeClosureGraphFailure
                .rootInventoryMismatch
        }
        let rederived:
            SyntheticAcceptedDependencyCommandInventoryComparison
        do {
            rederived =
                try SyntheticAcceptedDependencyCommandInventoryVerifier
                .root(root)
        } catch {
            throw SyntheticRuntimeClosureGraphFailure
                .rootInventoryMismatch
        }
        guard rederived == inventory else {
            throw SyntheticRuntimeClosureGraphFailure
                .rootInventoryMismatch
        }
    }

    static func validateMemberInventories(
        members: [SyntheticRuntimeClosureMemberRecordComparison],
        memberInventories:
            [SyntheticAcceptedDependencyCommandInventoryComparison],
        memberPolicy: SyntheticRuntimeClosureGraphMemberPolicy
    ) throws -> MemberState {
        guard memberInventories.count == members.count else {
            throw SyntheticRuntimeClosureGraphFailure
                .memberInventoryCountMismatch
        }

        let memberByID = Dictionary(
            uniqueKeysWithValues: members.map {
                ($0.contentEvidenceID, $0)
            }
        )
        let memberIndexByID = Dictionary(
            uniqueKeysWithValues: members.enumerated().map {
                ($0.element.contentEvidenceID, $0.offset)
            }
        )
        let memberByInstallName = Dictionary(
            uniqueKeysWithValues: members.map {
                ($0.decodedInstallName, $0)
            }
        )

        var candidates: [InventoryFailureCandidate] = []
        var inventoriesByMemberIndex: [
            Int: SyntheticAcceptedDependencyCommandInventoryComparison
        ] = [:]
        inventoriesByMemberIndex.reserveCapacity(members.count)

        for (inventoryIndex, inventory) in
            memberInventories.enumerated()
        {
            guard
                let memberIndex =
                    memberIndexByID[
                        inventory.parentContentEvidenceID
                    ]
            else {
                candidates.append(
                    InventoryFailureCandidate(
                        locus: inventoryIndex,
                        rank: 2,
                        failure: .foreignMemberInventory(
                            inventoryIndex: inventoryIndex
                        )
                    )
                )
                continue
            }
            guard inventoriesByMemberIndex[memberIndex] == nil else {
                candidates.append(
                    InventoryFailureCandidate(
                        locus: memberIndex,
                        rank: 0,
                        failure: .duplicateMemberInventory(
                            memberIndex: memberIndex
                        )
                    )
                )
                continue
            }
            inventoriesByMemberIndex[memberIndex] = inventory
        }

        for memberIndex in members.indices {
            guard
                let inventory =
                    inventoriesByMemberIndex[memberIndex]
            else {
                candidates.append(
                    InventoryFailureCandidate(
                        locus: memberIndex,
                        rank: 1,
                        failure: .missingMemberInventory(
                            memberIndex: memberIndex
                        )
                    )
                )
                continue
            }
            if !memberInventoryMatches(
                inventory,
                member: members[memberIndex],
                memberPolicy: memberPolicy
            ) {
                candidates.append(
                    InventoryFailureCandidate(
                        locus: memberIndex,
                        rank: 3,
                        failure: .memberInventoryEvidenceMismatch(
                            memberIndex: memberIndex
                        )
                    )
                )
            }
        }

        // Equal inventory counts make duplicate, missing, and foreign
        // defects overlap. Choose the earliest bounded input locus, with
        // this rank only breaking ties, so every closed failure remains
        // independently reachable without caller-provided keys.
        if let failure = candidates.min(by: {
            if $0.locus != $1.locus {
                return $0.locus < $1.locus
            }
            return $0.rank < $1.rank
        }) {
            throw failure.failure
        }

        return MemberState(
            memberByID: memberByID,
            memberByInstallName: memberByInstallName,
            inventoriesByMemberIndex: inventoriesByMemberIndex
        )
    }

    static func memberInventoryMatches(
        _ inventory:
            SyntheticAcceptedDependencyCommandInventoryComparison,
        member: SyntheticRuntimeClosureMemberRecordComparison,
        memberPolicy: SyntheticRuntimeClosureGraphMemberPolicy
    ) -> Bool {
        guard
            inventory.parentContentEvidenceID ==
                member.contentEvidenceID,
            inventory.sourceLoadCommandsSHA256 ==
                member.loadCommandsSHA256
        else {
            return false
        }

        switch (member.source, inventory.source) {
        case let (.sharedCache(expectedImage),
                  .sharedCacheMember(snapshot)):
            guard
                snapshot.imageEvidence == expectedImage,
                snapshot.loadCommandsSHA256 ==
                    member.loadCommandsSHA256,
                let rederived = try?
                SyntheticAcceptedDependencyCommandInventoryVerifier
                    .sharedCacheMember(snapshot)
            else {
                return false
            }
            return rederived == inventory

        case let (.fileImage(expectedSnapshot),
                  .fileImageMember(actualSnapshot)):
            guard
                memberPolicy ==
                    .sharedCacheAndSealedFileImage,
                actualSnapshot == expectedSnapshot,
                actualSnapshot.fileImageEvidence.comparison
                    .loadCommandsSHA256 == member.loadCommandsSHA256,
                let rederivedSnapshot = try?
                    SyntheticFileImageRuntimeClosureMemberSnapshotVerifier
                    .derive(
                        fileImageEvidence:
                            actualSnapshot.fileImageEvidence
                    ),
                rederivedSnapshot == actualSnapshot,
                let rederivedInventory = try?
                    SyntheticAcceptedDependencyCommandInventoryVerifier
                    .fileImageMember(actualSnapshot)
            else {
                return false
            }
            return rederivedInventory == inventory

        case (.file, _),
             (.sharedCache, _),
             (.fileImage, _):
            return false
        }
    }

    static func orderedInventoryBindings(
        rootInventory:
            SyntheticAcceptedDependencyCommandInventoryComparison,
        memberInventoriesByMemberIndex:
            [Int: SyntheticAcceptedDependencyCommandInventoryComparison]
    ) -> [InventoryBinding] {
        var result = [InventoryBinding(inventory: rootInventory)]
        for memberIndex in memberInventoriesByMemberIndex.keys.sorted() {
            result.append(
                InventoryBinding(
                    inventory:
                        memberInventoriesByMemberIndex[memberIndex]!
                )
            )
        }
        result.sort {
            $0.inventory.parentContentEvidenceID.utf8
                .lexicographicallyPrecedes(
                    $1.inventory.parentContentEvidenceID.utf8
                )
        }
        return result
    }

    static func resolveInventoryEntries(
        _ inventories: [InventoryBinding],
        memberByInstallName:
            [Data: SyntheticRuntimeClosureMemberRecordComparison]
    ) throws -> [InventoryKey: ResolvedInventoryEntry] {
        var result: [InventoryKey: ResolvedInventoryEntry] = [:]
        result.reserveCapacity(
            min(
                maximumAcceptedEntryCount,
                inventories.reduce(0) {
                    $0 + $1.inventory.entries.count
                }
            )
        )
        var acceptedEntryCount = 0
        for (parentIndex, binding) in inventories.enumerated() {
            for entry in binding.inventory.entries {
                guard acceptedEntryCount < maximumAcceptedEntryCount
                else {
                    throw SyntheticRuntimeClosureGraphFailure
                        .unconsumedInventoryEntry(
                            parentIndex: parentIndex,
                            ordinal: entry.loadCommandOrdinal
                        )
                }
                acceptedEntryCount += 1
                guard
                    let member =
                        memberByInstallName[
                            entry.decodedInstallName
                        ]
                else {
                    throw SyntheticRuntimeClosureGraphFailure
                        .unresolvedInventoryEntry(
                            parentIndex: parentIndex,
                            ordinal: entry.loadCommandOrdinal
                        )
                }
                let key = InventoryKey(
                    parentContentEvidenceID:
                        binding.inventory.parentContentEvidenceID,
                    loadCommandOrdinal: entry.loadCommandOrdinal
                )
                result[key] = ResolvedInventoryEntry(
                    parentIndex: parentIndex,
                    entry: entry,
                    resolvedMember: member
                )
            }
        }
        return result
    }

    static func validateEdgeParents(
        _ edges: [SyntheticRuntimeClosureEdgeRecordComparison],
        root: ExecutableContentIdentityEvidence,
        memberByID:
            [String: SyntheticRuntimeClosureMemberRecordComparison]
    ) throws {
        for (edgeIndex, edge) in edges.enumerated() {
            switch edge.parent {
            case let .root(evidence):
                guard
                    evidence == root,
                    edge.parentContentEvidenceID ==
                        root.contentEvidenceID.sha256
                else {
                    throw SyntheticRuntimeClosureGraphFailure
                        .edgeParentEvidenceMismatch(
                            edgeIndex: edgeIndex
                        )
                }
            case let .member(record):
                guard
                    let expected =
                        memberByID[record.contentEvidenceID]
                else {
                    throw SyntheticRuntimeClosureGraphFailure
                        .edgeParentMissing(edgeIndex: edgeIndex)
                }
                guard
                    expected == record,
                    edge.parentContentEvidenceID ==
                        expected.contentEvidenceID
                else {
                    throw SyntheticRuntimeClosureGraphFailure
                        .edgeParentEvidenceMismatch(
                            edgeIndex: edgeIndex
                        )
                }
            }
        }
    }

    static func validateEdgeResolvedMembers(
        _ edges: [SyntheticRuntimeClosureEdgeRecordComparison],
        memberByID:
            [String: SyntheticRuntimeClosureMemberRecordComparison]
    ) throws {
        for (edgeIndex, edge) in edges.enumerated() {
            guard
                let expected =
                    memberByID[edge.resolvedContentEvidenceID]
            else {
                throw SyntheticRuntimeClosureGraphFailure
                    .edgeResolvedMissing(edgeIndex: edgeIndex)
            }
            guard
                expected == edge.resolved,
                expected.installName == edge.installName,
                expected.decodedInstallName ==
                    edge.decodedInstallName
            else {
                throw SyntheticRuntimeClosureGraphFailure
                    .edgeResolvedEvidenceMismatch(
                        edgeIndex: edgeIndex
                    )
            }
        }
    }

    static func validateEdgesAgainstInventories(
        _ edges: [SyntheticRuntimeClosureEdgeRecordComparison],
        resolvedEntries: [InventoryKey: ResolvedInventoryEntry]
    ) throws -> EdgeState {
        var consumed = Set<InventoryKey>()
        consumed.reserveCapacity(edges.count)
        var projections: [
            SyntheticRuntimeClosureValidatedEdgeProjection
        ] = []
        projections.reserveCapacity(edges.count)
        var adjacency: [String: [String]] = [:]
        adjacency.reserveCapacity(edges.count)

        for (edgeIndex, edge) in edges.enumerated() {
            let key = InventoryKey(
                parentContentEvidenceID:
                    edge.parentContentEvidenceID,
                loadCommandOrdinal: edge.loadCommandOrdinal
            )
            guard let resolvedEntry = resolvedEntries[key] else {
                throw SyntheticRuntimeClosureGraphFailure
                    .edgeInventoryEntryMissing(edgeIndex: edgeIndex)
            }
            guard resolvedEntry.entry.kind == edge.kind else {
                throw SyntheticRuntimeClosureGraphFailure
                    .edgeInventoryKindMismatch(edgeIndex: edgeIndex)
            }
            guard
                resolvedEntry.entry.installName == edge.installName,
                resolvedEntry.entry.decodedInstallName ==
                    edge.decodedInstallName,
                resolvedEntry.resolvedMember.contentEvidenceID ==
                    edge.resolvedContentEvidenceID
            else {
                throw SyntheticRuntimeClosureGraphFailure
                    .edgeInventoryInstallNameMismatch(
                        edgeIndex: edgeIndex
                    )
            }

            consumed.insert(key)
            projections.append(
                SyntheticRuntimeClosureValidatedEdgeProjection(
                    parentContentEvidenceID:
                        edge.parentContentEvidenceID,
                    loadCommandOrdinal: edge.loadCommandOrdinal,
                    kind: edge.kind,
                    decodedInstallName: edge.decodedInstallName,
                    resolvedContentEvidenceID:
                        edge.resolvedContentEvidenceID
                )
            )
            adjacency[edge.parentContentEvidenceID, default: []]
                .append(edge.resolvedContentEvidenceID)
        }

        return EdgeState(
            consumed: consumed,
            projections: projections,
            adjacency: adjacency
        )
    }

    static func validateAllInventoryEntriesConsumed(
        _ inventories: [InventoryBinding],
        consumed: Set<InventoryKey>
    ) throws {
        for (parentIndex, binding) in inventories.enumerated() {
            for entry in binding.inventory.entries {
                let key = InventoryKey(
                    parentContentEvidenceID:
                        binding.inventory.parentContentEvidenceID,
                    loadCommandOrdinal: entry.loadCommandOrdinal
                )
                guard consumed.contains(key) else {
                    throw SyntheticRuntimeClosureGraphFailure
                        .unconsumedInventoryEntry(
                            parentIndex: parentIndex,
                            ordinal: entry.loadCommandOrdinal
                        )
                }
            }
        }
    }

    static func reachableMemberIDs(
        rootID: String,
        members: [SyntheticRuntimeClosureMemberRecordComparison],
        adjacency: [String: [String]]
    ) throws -> [String] {
        var visited = Set<String>()
        visited.reserveCapacity(members.count + 1)
        visited.insert(rootID)
        var queue = [rootID]
        queue.reserveCapacity(members.count + 1)
        var cursor = 0

        while cursor < queue.count {
            let parent = queue[cursor]
            cursor += 1
            for resolved in adjacency[parent] ?? [] {
                if visited.insert(resolved).inserted {
                    queue.append(resolved)
                }
            }
        }

        for (memberIndex, member) in members.enumerated() {
            guard visited.contains(member.contentEvidenceID) else {
                throw SyntheticRuntimeClosureGraphFailure
                    .unreachableMember(memberIndex: memberIndex)
            }
        }
        return members.map(\.contentEvidenceID)
    }
}
