import CryptoKit
import Foundation

struct FileImageRuntimeClosureContentEvidenceID: Equatable, Sendable {
    let sha256: String

    fileprivate init(sha256: String) {
        self.sha256 = sha256
    }
}

enum FileImageRuntimeClosureContentExpectationClaimField:
    Equatable,
    Sendable
{
    case runtimeDecision
    case canExecute
    case canSpawn
    case canAccessNetwork
    case canConsumePack
    case canMutateFileSystem
    case canImportGitObjects
    case canBuild
    case canLoadModel
    case canReserveOutput
    case canPublish
}

enum FileImageRuntimeClosureContentGraphClaimField:
    Equatable,
    Sendable
{
    case rootRole
    case rootContentEvidenceID
    case provesGraphMembership
    case provesAcceptedCommandCompleteness
    case provesBoundedTraversal
    case provesRootReachability
    case provesSealedFileImageContinuity
    case isCompleteRuntimeClosure
    case provesRuntimeLaunchability
    case runtimeDecision
    case canExecute
    case canSpawn
    case canAccessNetwork
    case canConsumePack
    case canMutateFileSystem
    case canImportGitObjects
    case canBuild
    case canLoadModel
    case canReserveOutput
    case canPublish
}

enum FileImageRuntimeClosureContentGraphExpectationField:
    Equatable,
    Sendable
{
    case memberCount
    case edgeCount
    case declaredFileImageMemberCount
    case graphFileImageMemberCount
    case canonicalSectionRange
    case canonicalSectionBytes
}

enum FileImageRuntimeClosureContentIdentityFailure: Error, Equatable {
    case expectationReference(
        FileImageRuntimeClosureExpectationFailure
    )
    case expectationReferenceMismatch
    case expectationClaim(
        FileImageRuntimeClosureContentExpectationClaimField
    )
    case rootRole(
        expected: ExecutableContentArtifactRole,
        actual: ExecutableContentArtifactRole
    )
    case rootEvidence(ExecutableContentIdentityFailure)
    case rootEvidenceMismatch
    case rootManifestIDMismatch
    case dynamicLoaderEvidence(
        DynamicLoaderContentIdentityFailure
    )
    case dynamicLoaderEvidenceMismatch
    case dynamicLoaderManifestIDMismatch
    case graph(SyntheticRuntimeClosureGraphFailure)
    case graphComparisonMismatch
    case graphClaim(
        FileImageRuntimeClosureContentGraphClaimField
    )
    case graphExpectation(
        FileImageRuntimeClosureContentGraphExpectationField
    )
    case memberStorageKind(index: Int)
    case memberExpectation(
        index: Int,
        field: RuntimeClosureExpectationMemberRecordField
    )
    case sharedCacheMemberEvidence(
        index: Int,
        failure: SyntheticSharedCacheImageContentIdentityFailure
    )
    case sharedCacheMemberEvidenceMismatch(index: Int)
    case memberCacheSetMismatch(index: Int)
    case fileImageMemberEvidence(
        index: Int,
        failure:
            SyntheticFileImageRuntimeClosureMemberSnapshotFailure
    )
    case fileImageMemberEvidenceMismatch(index: Int)
}

fileprivate enum FileImageRuntimeClosureContentConstructionSeal:
    Equatable
{
    case verified
}

/// Compact equality evidence for one declared static graph with at least
/// one sealed D2 file-image member. It grants no runtime authority.
struct FileImageRuntimeClosureContentExpectationComparison: Equatable {
    let artifactRole: RuntimeClosureExpectationArtifactRole
    let manifestSHA256: String
    let manifestBytes: UInt64
    let rootExecutableContentEvidenceID: String
    let dynamicLoaderContentEvidenceID: String
    let sharedCacheSetID: String
    let memberCount: Int
    let edgeCount: Int
    let fileImageMemberCount: Int
    let contentEvidenceID: FileImageRuntimeClosureContentEvidenceID

    let provesExpectationAnchorMatch = true
    let provesManifestContentMatch = true
    let provesDeclaredStaticGraphMatch = true
    let provesSealedFileImageContinuity = true
    let isCompleteRuntimeClosure = false
    let provesRuntimeLaunchability = false
    let runtimeResolutionOutcome = "unproved-static-comparison-only"
    let runtimeDecision = RuntimeClosureExpectationRuntimeDecision.noGo

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

    fileprivate let constructionSeal:
        FileImageRuntimeClosureContentConstructionSeal

    fileprivate init(
        artifactRole: RuntimeClosureExpectationArtifactRole,
        manifestSHA256: String,
        manifestBytes: UInt64,
        rootExecutableContentEvidenceID: String,
        dynamicLoaderContentEvidenceID: String,
        sharedCacheSetID: String,
        memberCount: Int,
        edgeCount: Int,
        fileImageMemberCount: Int,
        contentEvidenceID: FileImageRuntimeClosureContentEvidenceID
    ) {
        self.artifactRole = artifactRole
        self.manifestSHA256 = manifestSHA256
        self.manifestBytes = manifestBytes
        self.rootExecutableContentEvidenceID =
            rootExecutableContentEvidenceID
        self.dynamicLoaderContentEvidenceID =
            dynamicLoaderContentEvidenceID
        self.sharedCacheSetID = sharedCacheSetID
        self.memberCount = memberCount
        self.edgeCount = edgeCount
        self.fileImageMemberCount = fileImageMemberCount
        self.contentEvidenceID = contentEvidenceID
        self.constructionSeal = .verified
    }
}

enum FileImageRuntimeClosureContentIdentityVerifier {
    static let identityDomain =
        "fast-mlx-proof-control-file-image-runtime-closure-content-evidence-id-v1"

    static func compare(
        expectationReference:
            FileImageRuntimeClosureExpectationReference,
        currentExpectationAnchor:
            RuntimeClosureExpectationTrustAnchor,
        rootExecutableContentEvidence:
            ExecutableContentIdentityEvidence,
        dynamicLoaderContentEvidence:
            DynamicLoaderContentIdentityEvidence,
        members: [SyntheticRuntimeClosureMemberRecordComparison],
        edges: [SyntheticRuntimeClosureEdgeRecordComparison],
        collection:
            SyntheticRuntimeClosureRecordCollectionComparison,
        rootInventory:
            SyntheticAcceptedDependencyCommandInventoryComparison,
        memberInventories:
            [SyntheticAcceptedDependencyCommandInventoryComparison],
        graphComparison:
            SyntheticFileImageRuntimeClosureGraphComparison
    ) throws -> FileImageRuntimeClosureContentExpectationComparison {
        let currentReference:
            FileImageRuntimeClosureExpectationReference
        do {
            currentReference = try
                FileImageRuntimeClosureExpectationVerifier.reference(
                    anchoredExpectation:
                        expectationReference.anchoredExpectation,
                    currentExpectationAnchor:
                        currentExpectationAnchor
                )
        } catch let failure as
            FileImageRuntimeClosureExpectationFailure
        {
            throw FileImageRuntimeClosureContentIdentityFailure
                .expectationReference(failure)
        }
        guard currentReference == expectationReference else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .expectationReferenceMismatch
        }
        let expectation = currentReference.anchoredExpectation

        let expectedRootRole = rootRole(
            for: expectation.fields.artifactRole
        )
        guard
            rootExecutableContentEvidence.artifactRole ==
                expectedRootRole
        else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .rootRole(
                    expected: expectedRootRole,
                    actual:
                        rootExecutableContentEvidence.artifactRole
                )
        }

        let rederivedRoot: ExecutableContentIdentityEvidence
        do {
            rederivedRoot = try ExecutableContentIdentityVerifier
                .derive(
                    artifactRole: expectedRootRole,
                    comparison:
                        rootExecutableContentEvidence.comparison
                )
        } catch let failure as ExecutableContentIdentityFailure {
            throw FileImageRuntimeClosureContentIdentityFailure
                .rootEvidence(failure)
        }
        guard rederivedRoot == rootExecutableContentEvidence else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .rootEvidenceMismatch
        }
        guard
            rederivedRoot.contentEvidenceID.sha256 ==
                expectation.fields.rootExecutableContentEvidenceID
        else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .rootManifestIDMismatch
        }

        let rederivedLoader: DynamicLoaderContentIdentityEvidence
        do {
            rederivedLoader = try
                DynamicLoaderContentIdentityVerifier.derive(
                    comparison:
                        dynamicLoaderContentEvidence.comparison
                )
        } catch let failure as DynamicLoaderContentIdentityFailure {
            throw FileImageRuntimeClosureContentIdentityFailure
                .dynamicLoaderEvidence(failure)
        }
        guard rederivedLoader == dynamicLoaderContentEvidence else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .dynamicLoaderEvidenceMismatch
        }
        guard
            rederivedLoader.contentEvidenceID.sha256 ==
                expectation.fields.dynamicLoaderContentEvidenceID
        else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .dynamicLoaderManifestIDMismatch
        }

        let rederivedGraph:
            SyntheticFileImageRuntimeClosureGraphComparison
        do {
            rederivedGraph = try
                SyntheticFileImageRuntimeClosureGraphVerifier.compare(
                    root: rederivedRoot,
                    members: members,
                    edges: edges,
                    collection: collection,
                    rootInventory: rootInventory,
                    memberInventories: memberInventories
                )
        } catch let failure as SyntheticRuntimeClosureGraphFailure {
            throw FileImageRuntimeClosureContentIdentityFailure
                .graph(failure)
        }
        guard rederivedGraph == graphComparison else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .graphComparisonMismatch
        }
        try validateGraphClaims(
            rederivedGraph,
            rootRole: expectedRootRole,
            rootContentEvidenceID:
                rederivedRoot.contentEvidenceID.sha256
        )

        guard members.count == expectation.fields.members.count else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .graphExpectation(.memberCount)
        }
        guard edges.count == expectation.fields.edges.count else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .graphExpectation(.edgeCount)
        }
        var actualFileImageMemberCount = 0
        for member in members {
            guard case .fileImage = member.source else {
                continue
            }
            let next = actualFileImageMemberCount
                .addingReportingOverflow(1)
            guard !next.overflow else {
                throw FileImageRuntimeClosureContentIdentityFailure
                    .graphExpectation(
                        .declaredFileImageMemberCount
                    )
            }
            actualFileImageMemberCount = next.partialValue
        }
        guard
            actualFileImageMemberCount ==
                currentReference.declaredFileImageMemberCount
        else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .graphExpectation(.declaredFileImageMemberCount)
        }
        guard
            rederivedGraph.fileImageMemberCount ==
                actualFileImageMemberCount
        else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .graphExpectation(.graphFileImageMemberCount)
        }

        for (index, member) in members.enumerated() {
            let expectedMember = expectation.fields.members[index]
            try validateStorageAndEvidence(
                member,
                expected: expectedMember,
                sharedCacheSetEvidence:
                    expectation.sharedCacheSetEvidence,
                index: index
            )
            try validateMemberExpectation(
                member,
                expected: expectedMember,
                index: index
            )
        }

        try validateGraphSection(expectation, collection: collection)
        try validateExpectationClaims(currentReference)
        try validateExpectationClaims(expectation)

        let sharedCacheSetID = expectation.sharedCacheSetEvidence
            .sharedCacheSetID.sha256
        let preimage = identityPreimage(
            artifactRole: expectation.fields.artifactRole,
            manifestSHA256: expectation.documentSHA256,
            manifestBytes: expectation.documentBytes,
            rootExecutableContentEvidenceID:
                rederivedRoot.contentEvidenceID.sha256,
            dynamicLoaderContentEvidenceID:
                rederivedLoader.contentEvidenceID.sha256,
            sharedCacheSetID: sharedCacheSetID,
            fileImageMemberCount: actualFileImageMemberCount
        )
        let contentEvidenceID =
            FileImageRuntimeClosureContentEvidenceID(
                sha256: sha256Hex(preimage)
            )
        return FileImageRuntimeClosureContentExpectationComparison(
            artifactRole: expectation.fields.artifactRole,
            manifestSHA256: expectation.documentSHA256,
            manifestBytes: expectation.documentBytes,
            rootExecutableContentEvidenceID:
                rederivedRoot.contentEvidenceID.sha256,
            dynamicLoaderContentEvidenceID:
                rederivedLoader.contentEvidenceID.sha256,
            sharedCacheSetID: sharedCacheSetID,
            memberCount: members.count,
            edgeCount: edges.count,
            fileImageMemberCount: actualFileImageMemberCount,
            contentEvidenceID: contentEvidenceID
        )
    }
}

private extension FileImageRuntimeClosureContentIdentityVerifier {
    static func rootRole(
        for role: RuntimeClosureExpectationArtifactRole
    ) -> ExecutableContentArtifactRole {
        switch role {
        case .git:
            .git
        case .selfGuard:
            .selfGuard
        }
    }

    static func validateGraphClaims(
        _ graph: SyntheticFileImageRuntimeClosureGraphComparison,
        rootRole: ExecutableContentArtifactRole,
        rootContentEvidenceID: String
    ) throws {
        guard graph.rootRole == rootRole else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .graphClaim(.rootRole)
        }
        guard graph.rootContentEvidenceID == rootContentEvidenceID else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .graphClaim(.rootContentEvidenceID)
        }
        guard graph.provesGraphMembership else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .graphClaim(.provesGraphMembership)
        }
        guard graph.provesAcceptedCommandCompleteness else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .graphClaim(.provesAcceptedCommandCompleteness)
        }
        guard graph.provesBoundedTraversal else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .graphClaim(.provesBoundedTraversal)
        }
        guard graph.provesRootReachability else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .graphClaim(.provesRootReachability)
        }
        guard graph.provesSealedFileImageContinuity else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .graphClaim(.provesSealedFileImageContinuity)
        }
        guard !graph.isCompleteRuntimeClosure else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .graphClaim(.isCompleteRuntimeClosure)
        }
        guard !graph.provesRuntimeLaunchability else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .graphClaim(.provesRuntimeLaunchability)
        }
        guard graph.runtimeDecision == .noGo else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .graphClaim(.runtimeDecision)
        }
        try requireFalse(graph.canExecute, graphField: .canExecute)
        try requireFalse(graph.canSpawn, graphField: .canSpawn)
        try requireFalse(
            graph.canAccessNetwork,
            graphField: .canAccessNetwork
        )
        try requireFalse(
            graph.canConsumePack,
            graphField: .canConsumePack
        )
        try requireFalse(
            graph.canMutateFileSystem,
            graphField: .canMutateFileSystem
        )
        try requireFalse(
            graph.canImportGitObjects,
            graphField: .canImportGitObjects
        )
        try requireFalse(graph.canBuild, graphField: .canBuild)
        try requireFalse(graph.canLoadModel, graphField: .canLoadModel)
        try requireFalse(
            graph.canReserveOutput,
            graphField: .canReserveOutput
        )
        try requireFalse(graph.canPublish, graphField: .canPublish)
    }

    static func requireFalse(
        _ value: Bool,
        graphField: FileImageRuntimeClosureContentGraphClaimField
    ) throws {
        guard !value else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .graphClaim(graphField)
        }
    }

    static func validateStorageAndEvidence(
        _ member: SyntheticRuntimeClosureMemberRecordComparison,
        expected: RuntimeClosureExpectationMemberFields,
        sharedCacheSetEvidence:
            SyntheticSharedCacheSetIdentityEvidence,
        index: Int
    ) throws {
        switch (expected.storage, member.storage, member.source) {
        case let (.sharedCache, .sharedCache, .sharedCache(image)):
            let rederived:
                SyntheticSharedCacheImageContentIdentityEvidence
            do {
                rederived = try
                    SyntheticSharedCacheImageContentIdentityVerifier
                    .derive(
                        cacheSetEvidence: image.cacheSetEvidence,
                        facts: image.facts
                    )
            } catch let failure as
                SyntheticSharedCacheImageContentIdentityFailure
            {
                throw FileImageRuntimeClosureContentIdentityFailure
                    .sharedCacheMemberEvidence(
                        index: index,
                        failure: failure
                    )
            }
            guard rederived == image else {
                throw FileImageRuntimeClosureContentIdentityFailure
                    .sharedCacheMemberEvidenceMismatch(index: index)
            }
            guard image.cacheSetEvidence == sharedCacheSetEvidence else {
                throw FileImageRuntimeClosureContentIdentityFailure
                    .memberCacheSetMismatch(index: index)
            }

        case let (.file, .file, .fileImage(snapshot)):
            let rederived:
                SyntheticFileImageRuntimeClosureMemberSnapshot
            do {
                rederived = try
                    SyntheticFileImageRuntimeClosureMemberSnapshotVerifier
                    .derive(
                        fileImageEvidence: snapshot.fileImageEvidence
                    )
            } catch let failure as
                SyntheticFileImageRuntimeClosureMemberSnapshotFailure
            {
                throw FileImageRuntimeClosureContentIdentityFailure
                    .fileImageMemberEvidence(
                        index: index,
                        failure: failure
                    )
            }
            guard rederived == snapshot else {
                throw FileImageRuntimeClosureContentIdentityFailure
                    .fileImageMemberEvidenceMismatch(index: index)
            }

        default:
            throw FileImageRuntimeClosureContentIdentityFailure
                .memberStorageKind(index: index)
        }
    }

    static func validateMemberExpectation(
        _ member: SyntheticRuntimeClosureMemberRecordComparison,
        expected: RuntimeClosureExpectationMemberFields,
        index: Int
    ) throws {
        guard expected.contentEvidenceID == member.contentEvidenceID else {
            throw memberFailure(index, .contentEvidenceID)
        }
        guard expected.storage == member.storage else {
            throw memberFailure(index, .storage)
        }
        guard expected.installName.bytes == member.installName.bytes else {
            throw memberFailure(index, .installNameBytes)
        }
        guard
            expected.installName.base64URL ==
                member.installName.base64URL
        else {
            throw memberFailure(index, .installNameBase64URL)
        }
        guard expected.decodedInstallName == member.decodedInstallName else {
            throw memberFailure(index, .installNameSyntax)
        }
        guard expected.machOUUID == member.machOUUID else {
            throw memberFailure(index, .machOUUID)
        }
        guard
            expected.primaryCodeDirectoryBlobSHA256 ==
                member.primaryCodeDirectoryBlobSHA256
        else {
            throw memberFailure(
                index,
                .primaryCodeDirectoryBlobSHA256
            )
        }
        guard
            expected.loadCommandsSHA256 ==
                member.loadCommandsSHA256
        else {
            throw memberFailure(index, .loadCommandsSHA256)
        }
    }

    static func memberFailure(
        _ index: Int,
        _ field: RuntimeClosureExpectationMemberRecordField
    ) -> FileImageRuntimeClosureContentIdentityFailure {
        .memberExpectation(index: index, field: field)
    }

    static func validateGraphSection(
        _ expectation: AnchoredRuntimeClosureExpectationDocument,
        collection: SyntheticRuntimeClosureRecordCollectionComparison
    ) throws {
        let bytes = expectation.expectationFile.bytes
        let range = expectation.canonicalGraphSectionRange
        guard
            !range.isEmpty,
            range.lowerBound >= bytes.startIndex,
            range.upperBound <= bytes.endIndex,
            range.count == collection.canonicalSectionBytes.count
        else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .graphExpectation(.canonicalSectionRange)
        }
        for offset in 0..<range.count {
            let expectationIndex = bytes.index(
                range.lowerBound,
                offsetBy: offset
            )
            let collectionIndex = collection.canonicalSectionBytes.index(
                collection.canonicalSectionBytes.startIndex,
                offsetBy: offset
            )
            guard
                bytes[expectationIndex] ==
                    collection.canonicalSectionBytes[collectionIndex]
            else {
                throw FileImageRuntimeClosureContentIdentityFailure
                    .graphExpectation(.canonicalSectionBytes)
            }
        }
    }

    static func validateExpectationClaims(
        _ reference: FileImageRuntimeClosureExpectationReference
    ) throws {
        guard reference.runtimeDecision == .noGo else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .expectationClaim(.runtimeDecision)
        }
        try validateFalseExpectationClaims(
            canExecute: reference.canExecute,
            canSpawn: reference.canSpawn,
            canAccessNetwork: reference.canAccessNetwork,
            canConsumePack: reference.canConsumePack,
            canMutateFileSystem: reference.canMutateFileSystem,
            canImportGitObjects: reference.canImportGitObjects,
            canBuild: reference.canBuild,
            canLoadModel: reference.canLoadModel,
            canReserveOutput: reference.canReserveOutput,
            canPublish: reference.canPublish
        )
    }

    static func validateExpectationClaims(
        _ expectation: AnchoredRuntimeClosureExpectationDocument
    ) throws {
        guard expectation.runtimeDecision == .noGo else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .expectationClaim(.runtimeDecision)
        }
        try validateFalseExpectationClaims(
            canExecute: expectation.canExecute,
            canSpawn: expectation.canSpawn,
            canAccessNetwork: expectation.canAccessNetwork,
            canConsumePack: expectation.canConsumePack,
            canMutateFileSystem: expectation.canMutateFileSystem,
            canImportGitObjects: expectation.canImportGitObjects,
            canBuild: expectation.canBuild,
            canLoadModel: expectation.canLoadModel,
            canReserveOutput: expectation.canReserveOutput,
            canPublish: expectation.canPublish
        )
    }

    static func validateFalseExpectationClaims(
        canExecute: Bool,
        canSpawn: Bool,
        canAccessNetwork: Bool,
        canConsumePack: Bool,
        canMutateFileSystem: Bool,
        canImportGitObjects: Bool,
        canBuild: Bool,
        canLoadModel: Bool,
        canReserveOutput: Bool,
        canPublish: Bool
    ) throws {
        try requireFalse(canExecute, expectationField: .canExecute)
        try requireFalse(canSpawn, expectationField: .canSpawn)
        try requireFalse(
            canAccessNetwork,
            expectationField: .canAccessNetwork
        )
        try requireFalse(
            canConsumePack,
            expectationField: .canConsumePack
        )
        try requireFalse(
            canMutateFileSystem,
            expectationField: .canMutateFileSystem
        )
        try requireFalse(
            canImportGitObjects,
            expectationField: .canImportGitObjects
        )
        try requireFalse(canBuild, expectationField: .canBuild)
        try requireFalse(canLoadModel, expectationField: .canLoadModel)
        try requireFalse(
            canReserveOutput,
            expectationField: .canReserveOutput
        )
        try requireFalse(canPublish, expectationField: .canPublish)
    }

    static func requireFalse(
        _ value: Bool,
        expectationField:
            FileImageRuntimeClosureContentExpectationClaimField
    ) throws {
        guard !value else {
            throw FileImageRuntimeClosureContentIdentityFailure
                .expectationClaim(expectationField)
        }
    }

    static func identityPreimage(
        artifactRole: RuntimeClosureExpectationArtifactRole,
        manifestSHA256: String,
        manifestBytes: UInt64,
        rootExecutableContentEvidenceID: String,
        dynamicLoaderContentEvidenceID: String,
        sharedCacheSetID: String,
        fileImageMemberCount: Int
    ) -> Data {
        Data(
            ([
                identityDomain,
                "artifact_role=\(artifactRole.rawValue)",
                "manifest_sha256=\(manifestSHA256)",
                "manifest_bytes=\(manifestBytes)",
                "root_executable_content_evidence_id=" +
                    rootExecutableContentEvidenceID,
                "dynamic_loader_content_evidence_id=" +
                    dynamicLoaderContentEvidenceID,
                "shared_cache_set_id=\(sharedCacheSetID)",
                "file_image_member_count=\(fileImageMemberCount)",
            ].joined(separator: "\n") + "\n").utf8
        )
    }

    static func sha256Hex(_ bytes: Data) -> String {
        SHA256.hash(data: bytes)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}
