import CryptoKit
import Foundation

struct RuntimeClosureContentEvidenceID: Equatable, Sendable {
  let sha256: String

  fileprivate init(sha256: String) {
    self.sha256 = sha256
  }
}

enum RuntimeClosureContentExpectationClaimField:
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

enum RuntimeClosureContentGraphClaimField:
  Equatable,
  Sendable
{
  case rootRole
  case rootContentEvidenceID
  case provesGraphMembership
  case provesAcceptedCommandCompleteness
  case provesBoundedTraversal
  case provesRootReachability
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

enum RuntimeClosureContentGraphExpectationField:
  Equatable,
  Sendable
{
  case memberCount
  case edgeCount
  case canonicalSectionRange
  case canonicalSectionBytes
}

enum RuntimeClosureContentIdentityFailure: Error, Equatable {
  case expectationReanchor(RuntimeClosureExpectationError)
  case expectationEvidenceMismatch
  case expectationClaim(RuntimeClosureContentExpectationClaimField)
  case rootRole(
    expected: ExecutableContentArtifactRole,
    actual: ExecutableContentArtifactRole
  )
  case rootEvidence(ExecutableContentIdentityFailure)
  case rootEvidenceMismatch
  case rootManifestIDMismatch
  case dynamicLoaderEvidence(DynamicLoaderContentIdentityFailure)
  case dynamicLoaderEvidenceMismatch
  case dynamicLoaderManifestIDMismatch
  case graph(SyntheticRuntimeClosureGraphFailure)
  case graphComparisonMismatch
  case graphClaim(RuntimeClosureContentGraphClaimField)
  case graphExpectation(RuntimeClosureContentGraphExpectationField)
  case sharedCacheMemberEvidence(
    index: Int,
    failure: SyntheticSharedCacheImageContentIdentityFailure
  )
  case sharedCacheMemberEvidenceMismatch(index: Int)
  case memberCacheSetMismatch(index: Int)
}

private enum RuntimeClosureContentConstructionSeal: Equatable {
  case verified
}

/// Compact equality evidence for one declared static runtime-closure graph.
/// This value does not prove launchability or authorize runtime operations.
struct RuntimeClosureContentExpectationComparison: Equatable {
  let artifactRole: RuntimeClosureExpectationArtifactRole
  let manifestSHA256: String
  let manifestBytes: UInt64
  let rootExecutableContentEvidenceID: String
  let dynamicLoaderContentEvidenceID: String
  let sharedCacheSetID: String
  let memberCount: Int
  let edgeCount: Int
  let contentEvidenceID: RuntimeClosureContentEvidenceID

  let provesExpectationAnchorMatch = true
  let provesManifestContentMatch = true
  let provesDeclaredStaticGraphMatch = true
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

  fileprivate let constructionSeal: RuntimeClosureContentConstructionSeal

  fileprivate init(
    artifactRole: RuntimeClosureExpectationArtifactRole,
    manifestSHA256: String,
    manifestBytes: UInt64,
    rootExecutableContentEvidenceID: String,
    dynamicLoaderContentEvidenceID: String,
    sharedCacheSetID: String,
    memberCount: Int,
    edgeCount: Int,
    contentEvidenceID: RuntimeClosureContentEvidenceID
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
    self.contentEvidenceID = contentEvidenceID
    self.constructionSeal = .verified
  }
}

enum RuntimeClosureContentIdentityVerifier {
  static let identityDomain =
    "fast-mlx-proof-control-runtime-closure-content-evidence-id-v1"

  static func compare(
    anchoredExpectation: AnchoredRuntimeClosureExpectationDocument,
    currentExpectationAnchor: RuntimeClosureExpectationTrustAnchor,
    rootExecutableContentEvidence: ExecutableContentIdentityEvidence,
    dynamicLoaderContentEvidence: DynamicLoaderContentIdentityEvidence,
    members: [SyntheticRuntimeClosureMemberRecordComparison],
    edges: [SyntheticRuntimeClosureEdgeRecordComparison],
    collection: SyntheticRuntimeClosureRecordCollectionComparison,
    rootInventory:
      SyntheticAcceptedDependencyCommandInventoryComparison,
    memberInventories: [SyntheticAcceptedDependencyCommandInventoryComparison],
    graphComparison: SyntheticRuntimeClosureGraphComparison
  ) throws -> RuntimeClosureContentExpectationComparison {
    let currentExpectation: AnchoredRuntimeClosureExpectationDocument
    do {
      currentExpectation =
        try RuntimeClosureExpectationVerifier
        .anchor(
          expectationFile:
            anchoredExpectation.expectationFile,
          trustAnchor: currentExpectationAnchor
        )
    } catch let failure as RuntimeClosureExpectationError {
      throw
        RuntimeClosureContentIdentityFailure
        .expectationReanchor(failure)
    }
    guard currentExpectation == anchoredExpectation else {
      throw RuntimeClosureContentIdentityFailure
        .expectationEvidenceMismatch
    }

    let expectedRootRole = rootRole(
      for: currentExpectation.fields.artifactRole
    )
    guard rootExecutableContentEvidence.artifactRole == expectedRootRole
    else {
      throw RuntimeClosureContentIdentityFailure.rootRole(
        expected: expectedRootRole,
        actual: rootExecutableContentEvidence.artifactRole
      )
    }

    let rederivedRoot: ExecutableContentIdentityEvidence
    do {
      rederivedRoot = try ExecutableContentIdentityVerifier.derive(
        artifactRole: expectedRootRole,
        comparison: rootExecutableContentEvidence.comparison
      )
    } catch let failure as ExecutableContentIdentityFailure {
      throw
        RuntimeClosureContentIdentityFailure
        .rootEvidence(failure)
    }
    guard rederivedRoot == rootExecutableContentEvidence else {
      throw RuntimeClosureContentIdentityFailure
        .rootEvidenceMismatch
    }
    guard
      rederivedRoot.contentEvidenceID.sha256
        == currentExpectation.fields.rootExecutableContentEvidenceID
    else {
      throw RuntimeClosureContentIdentityFailure
        .rootManifestIDMismatch
    }

    let rederivedLoader: DynamicLoaderContentIdentityEvidence
    do {
      rederivedLoader =
        try DynamicLoaderContentIdentityVerifier
        .derive(
          comparison:
            dynamicLoaderContentEvidence.comparison
        )
    } catch let failure as DynamicLoaderContentIdentityFailure {
      throw
        RuntimeClosureContentIdentityFailure
        .dynamicLoaderEvidence(failure)
    }
    guard rederivedLoader == dynamicLoaderContentEvidence else {
      throw RuntimeClosureContentIdentityFailure
        .dynamicLoaderEvidenceMismatch
    }
    guard
      rederivedLoader.contentEvidenceID.sha256
        == currentExpectation.fields.dynamicLoaderContentEvidenceID
    else {
      throw RuntimeClosureContentIdentityFailure
        .dynamicLoaderManifestIDMismatch
    }

    let rederivedGraph: SyntheticRuntimeClosureGraphComparison
    do {
      rederivedGraph =
        try SyntheticRuntimeClosureGraphVerifier
        .compare(
          root: rederivedRoot,
          members: members,
          edges: edges,
          collection: collection,
          rootInventory: rootInventory,
          memberInventories: memberInventories
        )
    } catch let failure as SyntheticRuntimeClosureGraphFailure {
      throw RuntimeClosureContentIdentityFailure.graph(failure)
    }
    guard rederivedGraph == graphComparison else {
      throw RuntimeClosureContentIdentityFailure
        .graphComparisonMismatch
    }
    try validateGraphClaims(
      rederivedGraph,
      rootRole: expectedRootRole,
      rootContentEvidenceID:
        rederivedRoot.contentEvidenceID.sha256
    )

    guard members.count == currentExpectation.fields.members.count
    else {
      throw
        RuntimeClosureContentIdentityFailure
        .graphExpectation(.memberCount)
    }
    guard edges.count == currentExpectation.fields.edges.count else {
      throw
        RuntimeClosureContentIdentityFailure
        .graphExpectation(.edgeCount)
    }

    for (index, member) in members.enumerated() {
      guard case .sharedCache(let imageEvidence) = member.source
      else {
        throw RuntimeClosureContentIdentityFailure.graph(
          .unsupportedFileMemberIdentity(
            memberIndex: index
          )
        )
      }
      let rederivedImage: SyntheticSharedCacheImageContentIdentityEvidence
      do {
        rederivedImage =
          try SyntheticSharedCacheImageContentIdentityVerifier
          .derive(
            cacheSetEvidence:
              imageEvidence.cacheSetEvidence,
            facts: imageEvidence.facts
          )
      } catch let failure as SyntheticSharedCacheImageContentIdentityFailure {
        throw
          RuntimeClosureContentIdentityFailure
          .sharedCacheMemberEvidence(
            index: index,
            failure: failure
          )
      }
      guard rederivedImage == imageEvidence else {
        throw
          RuntimeClosureContentIdentityFailure
          .sharedCacheMemberEvidenceMismatch(index: index)
      }
      guard imageEvidence.cacheSetEvidence == currentExpectation.sharedCacheSetEvidence
      else {
        throw
          RuntimeClosureContentIdentityFailure
          .memberCacheSetMismatch(index: index)
      }
    }

    try validateGraphSection(
      currentExpectation,
      collection: collection
    )
    try validateExpectationClaims(currentExpectation)

    let sharedCacheSetID = currentExpectation
      .sharedCacheSetEvidence.sharedCacheSetID.sha256
    let identityPreimage = identityPreimage(
      artifactRole: currentExpectation.fields.artifactRole,
      manifestSHA256: currentExpectation.documentSHA256,
      manifestBytes: currentExpectation.documentBytes,
      rootExecutableContentEvidenceID:
        rederivedRoot.contentEvidenceID.sha256,
      dynamicLoaderContentEvidenceID:
        rederivedLoader.contentEvidenceID.sha256,
      sharedCacheSetID: sharedCacheSetID
    )
    let contentEvidenceID = RuntimeClosureContentEvidenceID(
      sha256: sha256Hex(identityPreimage)
    )

    return RuntimeClosureContentExpectationComparison(
      artifactRole: currentExpectation.fields.artifactRole,
      manifestSHA256: currentExpectation.documentSHA256,
      manifestBytes: currentExpectation.documentBytes,
      rootExecutableContentEvidenceID:
        rederivedRoot.contentEvidenceID.sha256,
      dynamicLoaderContentEvidenceID:
        rederivedLoader.contentEvidenceID.sha256,
      sharedCacheSetID: sharedCacheSetID,
      memberCount: members.count,
      edgeCount: edges.count,
      contentEvidenceID: contentEvidenceID
    )
  }
}

extension RuntimeClosureContentIdentityVerifier {
  fileprivate static func rootRole(
    for role: RuntimeClosureExpectationArtifactRole
  ) -> ExecutableContentArtifactRole {
    switch role {
    case .git:
      .git
    case .selfGuard:
      .selfGuard
    }
  }

  fileprivate static func validateGraphClaims(
    _ graph: SyntheticRuntimeClosureGraphComparison,
    rootRole: ExecutableContentArtifactRole,
    rootContentEvidenceID: String
  ) throws {
    guard graph.rootRole == rootRole else {
      throw
        RuntimeClosureContentIdentityFailure
        .graphClaim(.rootRole)
    }
    guard graph.rootContentEvidenceID == rootContentEvidenceID
    else {
      throw
        RuntimeClosureContentIdentityFailure
        .graphClaim(.rootContentEvidenceID)
    }
    guard graph.provesGraphMembership else {
      throw
        RuntimeClosureContentIdentityFailure
        .graphClaim(.provesGraphMembership)
    }
    guard graph.provesAcceptedCommandCompleteness else {
      throw
        RuntimeClosureContentIdentityFailure
        .graphClaim(.provesAcceptedCommandCompleteness)
    }
    guard graph.provesBoundedTraversal else {
      throw
        RuntimeClosureContentIdentityFailure
        .graphClaim(.provesBoundedTraversal)
    }
    guard graph.provesRootReachability else {
      throw
        RuntimeClosureContentIdentityFailure
        .graphClaim(.provesRootReachability)
    }
    guard !graph.isCompleteRuntimeClosure else {
      throw
        RuntimeClosureContentIdentityFailure
        .graphClaim(.isCompleteRuntimeClosure)
    }
    guard !graph.provesRuntimeLaunchability else {
      throw
        RuntimeClosureContentIdentityFailure
        .graphClaim(.provesRuntimeLaunchability)
    }
    guard graph.runtimeDecision == .noGo else {
      throw
        RuntimeClosureContentIdentityFailure
        .graphClaim(.runtimeDecision)
    }
    try requireFalse(
      graph.canExecute,
      field: RuntimeClosureContentGraphClaimField.canExecute
    )
    try requireFalse(
      graph.canSpawn,
      field: RuntimeClosureContentGraphClaimField.canSpawn
    )
    try requireFalse(
      graph.canAccessNetwork,
      field:
        RuntimeClosureContentGraphClaimField.canAccessNetwork
    )
    try requireFalse(
      graph.canConsumePack,
      field: RuntimeClosureContentGraphClaimField.canConsumePack
    )
    try requireFalse(
      graph.canMutateFileSystem,
      field:
        RuntimeClosureContentGraphClaimField.canMutateFileSystem
    )
    try requireFalse(
      graph.canImportGitObjects,
      field:
        RuntimeClosureContentGraphClaimField.canImportGitObjects
    )
    try requireFalse(
      graph.canBuild,
      field: RuntimeClosureContentGraphClaimField.canBuild
    )
    try requireFalse(
      graph.canLoadModel,
      field: RuntimeClosureContentGraphClaimField.canLoadModel
    )
    try requireFalse(
      graph.canReserveOutput,
      field:
        RuntimeClosureContentGraphClaimField.canReserveOutput
    )
    try requireFalse(
      graph.canPublish,
      field: RuntimeClosureContentGraphClaimField.canPublish
    )
  }

  fileprivate static func requireFalse(
    _ value: Bool,
    field: RuntimeClosureContentGraphClaimField
  ) throws {
    guard !value else {
      throw
        RuntimeClosureContentIdentityFailure
        .graphClaim(field)
    }
  }

  fileprivate static func validateGraphSection(
    _ expectation: AnchoredRuntimeClosureExpectationDocument,
    collection: SyntheticRuntimeClosureRecordCollectionComparison
  ) throws {
    let bytes = expectation.expectationFile.bytes
    let range = expectation.canonicalGraphSectionRange
    guard !range.isEmpty,
      range.lowerBound >= bytes.startIndex,
      range.upperBound <= bytes.endIndex,
      range.count == collection.canonicalSectionBytes.count
    else {
      throw
        RuntimeClosureContentIdentityFailure
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
      guard bytes[expectationIndex] == collection.canonicalSectionBytes[collectionIndex]
      else {
        throw
          RuntimeClosureContentIdentityFailure
          .graphExpectation(.canonicalSectionBytes)
      }
    }
  }

  fileprivate static func validateExpectationClaims(
    _ expectation: AnchoredRuntimeClosureExpectationDocument
  ) throws {
    guard expectation.runtimeDecision == .noGo else {
      throw
        RuntimeClosureContentIdentityFailure
        .expectationClaim(.runtimeDecision)
    }
    try requireFalse(
      expectation.canExecute,
      field:
        RuntimeClosureContentExpectationClaimField.canExecute
    )
    try requireFalse(
      expectation.canSpawn,
      field:
        RuntimeClosureContentExpectationClaimField.canSpawn
    )
    try requireFalse(
      expectation.canAccessNetwork,
      field:
        RuntimeClosureContentExpectationClaimField
        .canAccessNetwork
    )
    try requireFalse(
      expectation.canConsumePack,
      field:
        RuntimeClosureContentExpectationClaimField.canConsumePack
    )
    try requireFalse(
      expectation.canMutateFileSystem,
      field:
        RuntimeClosureContentExpectationClaimField
        .canMutateFileSystem
    )
    try requireFalse(
      expectation.canImportGitObjects,
      field:
        RuntimeClosureContentExpectationClaimField
        .canImportGitObjects
    )
    try requireFalse(
      expectation.canBuild,
      field:
        RuntimeClosureContentExpectationClaimField.canBuild
    )
    try requireFalse(
      expectation.canLoadModel,
      field:
        RuntimeClosureContentExpectationClaimField.canLoadModel
    )
    try requireFalse(
      expectation.canReserveOutput,
      field:
        RuntimeClosureContentExpectationClaimField
        .canReserveOutput
    )
    try requireFalse(
      expectation.canPublish,
      field:
        RuntimeClosureContentExpectationClaimField.canPublish
    )
  }

  fileprivate static func requireFalse(
    _ value: Bool,
    field: RuntimeClosureContentExpectationClaimField
  ) throws {
    guard !value else {
      throw
        RuntimeClosureContentIdentityFailure
        .expectationClaim(field)
    }
  }

  fileprivate static func identityPreimage(
    artifactRole: RuntimeClosureExpectationArtifactRole,
    manifestSHA256: String,
    manifestBytes: UInt64,
    rootExecutableContentEvidenceID: String,
    dynamicLoaderContentEvidenceID: String,
    sharedCacheSetID: String
  ) -> Data {
    Data(
      ([
        identityDomain,
        "artifact_role=\(artifactRole.rawValue)",
        "manifest_sha256=\(manifestSHA256)",
        "manifest_bytes=\(manifestBytes)",
        "root_executable_content_evidence_id=" + rootExecutableContentEvidenceID,
        "dynamic_loader_content_evidence_id=" + dynamicLoaderContentEvidenceID,
        "shared_cache_set_id=\(sharedCacheSetID)",
      ].joined(separator: "\n") + "\n").utf8
    )
  }

  fileprivate static func sha256Hex(_ bytes: Data) -> String {
    SHA256.hash(data: bytes)
      .map { String(format: "%02x", $0) }
      .joined()
  }
}
