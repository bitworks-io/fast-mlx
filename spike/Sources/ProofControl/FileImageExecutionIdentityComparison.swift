import CryptoKit
import Foundation

struct FileImageExecutionIdentityAnchorContextID: Equatable, Sendable {
  let sha256: String

  fileprivate init(sha256: String) {
    self.sha256 = sha256
  }
}

enum FileImageExecutionIdentityComparisonFailure: Error, Equatable, Sendable {
  enum AnchorSlot: Equatable, Sendable {
    case gitToolPolicy
    case runtimeDenial
    case gitExecutableExpectation
    case selfGuardExecutableExpectation
    case gitClosureExpectation
    case selfGuardClosureExpectation
  }

  enum AnchorScalarField: Equatable, Sendable {
    case digest
    case byteCount
    case minimumGeneration
  }

  enum RoleSlot: Equatable, Sendable {
    case gitExecutableAnchor
    case gitExecutableEvidence
    case selfGuardExecutableAnchor
    case selfGuardExecutableEvidence
    case gitClosureAnchor
    case gitClosureEvidence
    case gitClosureReference
    case gitClosureComparison
    case selfGuardClosureAnchor
    case selfGuardClosureEvidence
    case selfGuardClosureReference
    case selfGuardClosureComparison
  }

  enum ClosureField: Equatable, Sendable {
    case manifestDigest
    case manifestBytes
    case rootID
    case dynamicLoaderID
    case cacheSetID
    case memberCount
    case edgeCount
    case fileImageMemberCount
    case scopedComparisonClaims
    case runtimeTerminal
    case authorityClaims
    case toolContentEvidenceID
    case denialContentEvidenceID
  }

  enum ExecutableField: Equatable, Sendable {
    case fileDigest
    case fileBytes
    case machOUUID
    case primaryCodeDirectoryDigest
    case authorityClaims
  }

  enum SharedEnvironmentField: Equatable, Sendable {
    case dynamicLoaderContentID
    case sharedCacheSetID
  }

  enum PlatformField: Equatable, Sendable {
    case architecture
    case hardwareModel
    case osVersion
    case osBuild
  }

  enum Predecessor: Equatable, Sendable {
    case gitToolAnchor
    case gitToolReference
    case runtimeDenialAnchor
    case runtimeDenialReference
    case gitExecutableAnchor
    case gitExecutableContent
    case gitExecutableMatch
    case selfGuardExecutableAnchor
    case selfGuardExecutableContent
    case selfGuardExecutableMatch
    case gitClosureAnchor
    case selfGuardClosureAnchor
  }

  case invalidAnchorScalar(
    slot: AnchorSlot,
    field: AnchorScalarField
  )
  case roleSubstitution(slot: RoleSlot)
  case anchorEvaluationTimeMismatch(AnchorSlot)

  case gitToolPolicyReanchor(GitToolPolicyV2Error)
  case gitToolPolicyEvidenceMismatch
  case signedClaimToolReference(GitToolPolicyV2ClaimReferenceError)
  case signedClaimToolReferenceMismatch

  case runtimeDenialReanchor(GitRuntimePolicyDenialV2Error)
  case runtimeDenialEvidenceMismatch
  case runtimeDenialReference(GitRuntimePolicyDenialV2Error)
  case predecessorContractViolation(Predecessor)

  case executableExpectation(
    role: RuntimeClosureExpectationArtifactRole,
    failure: ExecutableIdentityExpectationError
  )
  case executableExpectationMismatch(
    role: RuntimeClosureExpectationArtifactRole
  )
  case executableEvidence(
    role: RuntimeClosureExpectationArtifactRole,
    failure: ExecutableContentIdentityFailure
  )
  case executableEvidenceMismatch(
    role: RuntimeClosureExpectationArtifactRole
  )
  case executableComparison(
    role: RuntimeClosureExpectationArtifactRole,
    failure: ExecutableContentIdentityFailure
  )
  case executableComparisonMismatch(
    role: RuntimeClosureExpectationArtifactRole
  )

  case closureExpectation(
    role: RuntimeClosureExpectationArtifactRole,
    failure: RuntimeClosureExpectationError
  )
  case closureExpectationMismatch(
    role: RuntimeClosureExpectationArtifactRole
  )
  case closureReference(
    role: RuntimeClosureExpectationArtifactRole
  )
  case closureReferenceMismatch(
    role: RuntimeClosureExpectationArtifactRole
  )
  case closureContinuity(
    role: RuntimeClosureExpectationArtifactRole,
    field: ClosureField
  )
  case executablePolicyContinuity(
    role: RuntimeClosureExpectationArtifactRole,
    field: ExecutableField
  )
  case sharedEnvironmentMismatch(SharedEnvironmentField)
  case runtimePolicyContinuity
  case platformMismatch(PlatformField)
}

/// Compact singleton equality evidence only. This value is deliberately not
/// self-revalidating and grants no runtime or source-import authority.
struct FileImageExecutionIdentityComparisonEvidence: Equatable, Sendable {
  let runClaimID: String
  let anchorContextID: FileImageExecutionIdentityAnchorContextID
  let gitToolPolicySHA256: String
  let runtimeDenialPolicySHA256: String
  let gitExecutableExpectationSHA256: String
  let gitExecutableContentEvidenceID: String
  let selfGuardExecutableExpectationSHA256: String
  let selfGuardExecutableContentEvidenceID: String
  let gitClosureManifestSHA256: String
  let gitFileImageRuntimeClosureContentEvidenceID: String
  let selfGuardClosureManifestSHA256: String
  let selfGuardFileImageRuntimeClosureContentEvidenceID: String
  let comparisonSHA256: String

  let provesSignedClaimToolPolicyMatch = true
  let provesRuntimeDenialMatch = true
  let provesExecutableIdentityMatch = true
  let provesFileImageRuntimeClosureIdentityMatch = true
  let provesSingleCurrentAnchorContext = true
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

  fileprivate init(
    runClaimID: String,
    anchorContextID: FileImageExecutionIdentityAnchorContextID,
    gitToolPolicySHA256: String,
    runtimeDenialPolicySHA256: String,
    gitExecutableExpectationSHA256: String,
    gitExecutableContentEvidenceID: String,
    selfGuardExecutableExpectationSHA256: String,
    selfGuardExecutableContentEvidenceID: String,
    gitClosureManifestSHA256: String,
    gitFileImageRuntimeClosureContentEvidenceID: String,
    selfGuardClosureManifestSHA256: String,
    selfGuardFileImageRuntimeClosureContentEvidenceID: String,
    comparisonSHA256: String
  ) {
    self.runClaimID = runClaimID
    self.anchorContextID = anchorContextID
    self.gitToolPolicySHA256 = gitToolPolicySHA256
    self.runtimeDenialPolicySHA256 = runtimeDenialPolicySHA256
    self.gitExecutableExpectationSHA256 =
      gitExecutableExpectationSHA256
    self.gitExecutableContentEvidenceID =
      gitExecutableContentEvidenceID
    self.selfGuardExecutableExpectationSHA256 =
      selfGuardExecutableExpectationSHA256
    self.selfGuardExecutableContentEvidenceID =
      selfGuardExecutableContentEvidenceID
    self.gitClosureManifestSHA256 = gitClosureManifestSHA256
    self.gitFileImageRuntimeClosureContentEvidenceID =
      gitFileImageRuntimeClosureContentEvidenceID
    self.selfGuardClosureManifestSHA256 =
      selfGuardClosureManifestSHA256
    self.selfGuardFileImageRuntimeClosureContentEvidenceID =
      selfGuardFileImageRuntimeClosureContentEvidenceID
    self.comparisonSHA256 = comparisonSHA256
  }
}

enum FileImageExecutionIdentityComparisonVerifier {
  static let anchorContextDomain =
    "fast-mlx-proof-control-file-image-execution-identity-anchor-context-id-v1"
  static let comparisonDomain =
    "fast-mlx-proof-control-file-image-execution-identity-comparison-id-v1"

  static func compare(
    signedClaimToolPolicyReference: SignedClaimGitToolPolicyV2Reference,
    currentGitToolPolicyAnchor: GitToolPolicyV2TrustAnchor,
    runtimeDenialDocument: AnchoredGitRuntimePolicyDenialV2Document,
    currentRuntimeDenialAnchor: GitRuntimePolicyDenialV2TrustAnchor,
    gitExecutableComparison: ExecutableContentExpectationComparison,
    currentGitExecutableAnchor: ExecutableIdentityExpectationTrustAnchor,
    selfGuardExecutableComparison: ExecutableContentExpectationComparison,
    currentSelfGuardExecutableAnchor: ExecutableIdentityExpectationTrustAnchor,
    gitClosureExpectation: AnchoredRuntimeClosureExpectationDocument,
    currentGitClosureAnchor: RuntimeClosureExpectationTrustAnchor,
    gitClosureReference: FileImageRuntimeClosureExpectationReference,
    gitClosureComparison:
      FileImageRuntimeClosureContentExpectationComparison,
    selfGuardClosureExpectation: AnchoredRuntimeClosureExpectationDocument,
    currentSelfGuardClosureAnchor: RuntimeClosureExpectationTrustAnchor,
    selfGuardClosureReference:
      FileImageRuntimeClosureExpectationReference,
    selfGuardClosureComparison:
      FileImageRuntimeClosureContentExpectationComparison
  ) throws -> FileImageExecutionIdentityComparisonEvidence {
    try validateAnchorScalars(
      gitTool: currentGitToolPolicyAnchor,
      denial: currentRuntimeDenialAnchor,
      gitExecutable: currentGitExecutableAnchor,
      selfGuardExecutable: currentSelfGuardExecutableAnchor,
      gitClosure: currentGitClosureAnchor,
      selfGuardClosure: currentSelfGuardClosureAnchor
    )
    try validateCommonTime(
      gitTool: currentGitToolPolicyAnchor,
      denial: currentRuntimeDenialAnchor,
      gitExecutable: currentGitExecutableAnchor,
      selfGuardExecutable: currentSelfGuardExecutableAnchor,
      gitClosure: currentGitClosureAnchor,
      selfGuardClosure: currentSelfGuardClosureAnchor
    )

    let toolReference = try revalidateToolReference(
      signedClaimToolPolicyReference,
      anchor: currentGitToolPolicyAnchor
    )
    let denial = try revalidateRuntimeDenial(
      runtimeDenialDocument,
      anchor: currentRuntimeDenialAnchor,
      toolReference: toolReference
    )
    let gitExecutable = try revalidateExecutable(
      gitExecutableComparison,
      anchor: currentGitExecutableAnchor,
      role: .git
    )
    let selfGuardExecutable = try revalidateExecutable(
      selfGuardExecutableComparison,
      anchor: currentSelfGuardExecutableAnchor,
      role: .selfGuard
    )
    let gitClosure = try revalidateClosure(
      gitClosureExpectation,
      anchor: currentGitClosureAnchor,
      role: .git
    )
    let selfGuardClosure = try revalidateClosure(
      selfGuardClosureExpectation,
      anchor: currentSelfGuardClosureAnchor,
      role: .selfGuard
    )
    let gitReference = try revalidateClosureReference(
      gitClosureReference,
      expectation: gitClosure,
      anchor: currentGitClosureAnchor,
      role: .git
    )
    let selfGuardReference = try revalidateClosureReference(
      selfGuardClosureReference,
      expectation: selfGuardClosure,
      anchor: currentSelfGuardClosureAnchor,
      role: .selfGuard
    )
    try validateClosureComparison(
      gitClosureComparison,
      expectation: gitClosure,
      reference: gitReference,
      role: .git
    )
    try validateClosureComparison(
      selfGuardClosureComparison,
      expectation: selfGuardClosure,
      reference: selfGuardReference,
      role: .selfGuard
    )

    try validateContinuity(
      toolReference: toolReference,
      denial: denial,
      gitExecutable: gitExecutable,
      selfGuardExecutable: selfGuardExecutable,
      gitClosure: gitClosure,
      gitClosureComparison: gitClosureComparison,
      selfGuardClosure: selfGuardClosure,
      selfGuardClosureComparison: selfGuardClosureComparison
    )

    let anchorContextID = FileImageExecutionIdentityAnchorContextID(
      sha256: sha256Hex(
        anchorContextPreimage(
          gitTool: currentGitToolPolicyAnchor,
          denial: currentRuntimeDenialAnchor,
          gitExecutable: currentGitExecutableAnchor,
          selfGuardExecutable: currentSelfGuardExecutableAnchor,
          gitClosure: currentGitClosureAnchor,
          selfGuardClosure: currentSelfGuardClosureAnchor
        )
      )
    )
    let runClaimID = toolReference.signedClaim.claimID.rawValue
    let comparisonSHA256 = sha256Hex(
      comparisonPreimage(
        runClaimID: runClaimID,
        anchorContextID: anchorContextID,
        toolPolicy: toolReference.policyDocument,
        denial: denial,
        gitExecutable: gitExecutable,
        selfGuardExecutable: selfGuardExecutable,
        gitClosure: gitClosure,
        gitClosureComparison: gitClosureComparison,
        selfGuardClosure: selfGuardClosure,
        selfGuardClosureComparison: selfGuardClosureComparison
      )
    )

    return FileImageExecutionIdentityComparisonEvidence(
      runClaimID: runClaimID,
      anchorContextID: anchorContextID,
      gitToolPolicySHA256: toolReference.policyDocument.policySHA256,
      runtimeDenialPolicySHA256: denial.policySHA256,
      gitExecutableExpectationSHA256:
        gitExecutable.expectation.documentSHA256,
      gitExecutableContentEvidenceID:
        gitExecutable.evidence.contentEvidenceID.sha256,
      selfGuardExecutableExpectationSHA256:
        selfGuardExecutable.expectation.documentSHA256,
      selfGuardExecutableContentEvidenceID:
        selfGuardExecutable.evidence.contentEvidenceID.sha256,
      gitClosureManifestSHA256: gitClosure.documentSHA256,
      gitFileImageRuntimeClosureContentEvidenceID:
        gitClosureComparison.contentEvidenceID.sha256,
      selfGuardClosureManifestSHA256:
        selfGuardClosure.documentSHA256,
      selfGuardFileImageRuntimeClosureContentEvidenceID:
        selfGuardClosureComparison.contentEvidenceID.sha256,
      comparisonSHA256: comparisonSHA256
    )
  }
}

private extension FileImageExecutionIdentityComparisonVerifier {
  struct RevalidatedExecutable {
    let expectation: AnchoredExecutableIdentityExpectationDocument
    let evidence: ExecutableContentIdentityEvidence
    let comparison: ExecutableContentExpectationComparison
  }

  static func validateAnchorScalars(
    gitTool: GitToolPolicyV2TrustAnchor,
    denial: GitRuntimePolicyDenialV2TrustAnchor,
    gitExecutable: ExecutableIdentityExpectationTrustAnchor,
    selfGuardExecutable: ExecutableIdentityExpectationTrustAnchor,
    gitClosure: RuntimeClosureExpectationTrustAnchor,
    selfGuardClosure: RuntimeClosureExpectationTrustAnchor
  ) throws {
    try validateAnchor(
      slot: .gitToolPolicy,
      digest: gitTool.expectedCurrentPolicySHA256,
      bytes: gitTool.expectedCurrentPolicyBytes,
      generation: gitTool.minimumPolicyGeneration
    )
    try validateAnchor(
      slot: .runtimeDenial,
      digest: denial.expectedCurrentPolicySHA256,
      bytes: denial.expectedCurrentPolicyBytes,
      generation: denial.minimumPolicyGeneration
    )
    try validateAnchor(
      slot: .gitExecutableExpectation,
      digest: gitExecutable.expectedCurrentDocumentSHA256,
      bytes: gitExecutable.expectedCurrentDocumentBytes,
      generation: gitExecutable.minimumEvidenceGeneration
    )
    guard gitExecutable.expectedArtifactRole == .git else {
      throw FileImageExecutionIdentityComparisonFailure.roleSubstitution(
        slot: .gitExecutableAnchor
      )
    }
    try validateAnchor(
      slot: .selfGuardExecutableExpectation,
      digest: selfGuardExecutable.expectedCurrentDocumentSHA256,
      bytes: selfGuardExecutable.expectedCurrentDocumentBytes,
      generation: selfGuardExecutable.minimumEvidenceGeneration
    )
    guard selfGuardExecutable.expectedArtifactRole == .selfGuard else {
      throw FileImageExecutionIdentityComparisonFailure.roleSubstitution(
        slot: .selfGuardExecutableAnchor
      )
    }
    try validateAnchor(
      slot: .gitClosureExpectation,
      digest: gitClosure.expectedCurrentDocumentSHA256,
      bytes: gitClosure.expectedCurrentDocumentBytes,
      generation: gitClosure.minimumEvidenceGeneration
    )
    guard gitClosure.expectedArtifactRole == .git else {
      throw FileImageExecutionIdentityComparisonFailure.roleSubstitution(
        slot: .gitClosureAnchor
      )
    }
    try validateAnchor(
      slot: .selfGuardClosureExpectation,
      digest: selfGuardClosure.expectedCurrentDocumentSHA256,
      bytes: selfGuardClosure.expectedCurrentDocumentBytes,
      generation: selfGuardClosure.minimumEvidenceGeneration
    )
    guard selfGuardClosure.expectedArtifactRole == .selfGuard else {
      throw FileImageExecutionIdentityComparisonFailure.roleSubstitution(
        slot: .selfGuardClosureAnchor
      )
    }
  }

  static func validateAnchor(
    slot: FileImageExecutionIdentityComparisonFailure.AnchorSlot,
    digest: String,
    bytes: UInt64,
    generation: UInt64
  ) throws {
    guard isLowercaseHex(digest, count: 64) else {
      throw FileImageExecutionIdentityComparisonFailure.invalidAnchorScalar(
        slot: slot,
        field: .digest
      )
    }
    guard bytes > 0 else {
      throw FileImageExecutionIdentityComparisonFailure.invalidAnchorScalar(
        slot: slot,
        field: .byteCount
      )
    }
    guard generation > 0 else {
      throw FileImageExecutionIdentityComparisonFailure.invalidAnchorScalar(
        slot: slot,
        field: .minimumGeneration
      )
    }
  }

  static func validateCommonTime(
    gitTool: GitToolPolicyV2TrustAnchor,
    denial: GitRuntimePolicyDenialV2TrustAnchor,
    gitExecutable: ExecutableIdentityExpectationTrustAnchor,
    selfGuardExecutable: ExecutableIdentityExpectationTrustAnchor,
    gitClosure: RuntimeClosureExpectationTrustAnchor,
    selfGuardClosure: RuntimeClosureExpectationTrustAnchor
  ) throws {
    let expected = gitTool.verificationUnixSeconds
    let values: [(
      FileImageExecutionIdentityComparisonFailure.AnchorSlot, UInt64
    )] = [
      (.runtimeDenial, denial.verificationUnixSeconds),
      (.gitExecutableExpectation, gitExecutable.verificationUnixSeconds),
      (
        .selfGuardExecutableExpectation,
        selfGuardExecutable.verificationUnixSeconds
      ),
      (.gitClosureExpectation, gitClosure.verificationUnixSeconds),
      (
        .selfGuardClosureExpectation,
        selfGuardClosure.verificationUnixSeconds
      ),
    ]
    for (slot, actual) in values where actual != expected {
      throw FileImageExecutionIdentityComparisonFailure
        .anchorEvaluationTimeMismatch(slot)
    }
  }

  static func revalidateToolReference(
    _ reference: SignedClaimGitToolPolicyV2Reference,
    anchor: GitToolPolicyV2TrustAnchor
  ) throws -> SignedClaimGitToolPolicyV2Reference {
    let policy: AnchoredGitToolPolicyV2Document
    do {
      policy = try GitToolPolicyV2Verifier.anchor(
        policyFile: reference.policyDocument.policyFile,
        trustAnchor: anchor
      )
    } catch let failure as GitToolPolicyV2Error {
      throw FileImageExecutionIdentityComparisonFailure
        .gitToolPolicyReanchor(failure)
    } catch {
      throw FileImageExecutionIdentityComparisonFailure
        .predecessorContractViolation(.gitToolAnchor)
    }
    guard policy == reference.policyDocument else {
      throw FileImageExecutionIdentityComparisonFailure
        .gitToolPolicyEvidenceMismatch
    }

    let rederived: SignedClaimGitToolPolicyV2Reference
    do {
      rederived = try GitToolPolicyV2Verifier.reference(
        signedClaim: reference.signedClaim,
        policyDocument: policy
      )
    } catch let failure as GitToolPolicyV2ClaimReferenceError {
      throw FileImageExecutionIdentityComparisonFailure
        .signedClaimToolReference(failure)
    } catch {
      throw FileImageExecutionIdentityComparisonFailure
        .predecessorContractViolation(.gitToolReference)
    }
    guard rederived == reference else {
      throw FileImageExecutionIdentityComparisonFailure
        .signedClaimToolReferenceMismatch
    }
    return rederived
  }

  static func revalidateRuntimeDenial(
    _ document: AnchoredGitRuntimePolicyDenialV2Document,
    anchor: GitRuntimePolicyDenialV2TrustAnchor,
    toolReference: SignedClaimGitToolPolicyV2Reference
  ) throws -> AnchoredGitRuntimePolicyDenialV2Document {
    let reanchored: AnchoredGitRuntimePolicyDenialV2Document
    do {
      reanchored = try GitRuntimePolicyDenialV2Verifier.anchor(
        policyFile: document.policyFile,
        trustAnchor: anchor
      )
    } catch let failure as GitRuntimePolicyDenialV2Error {
      throw FileImageExecutionIdentityComparisonFailure
        .runtimeDenialReanchor(failure)
    } catch {
      throw FileImageExecutionIdentityComparisonFailure
        .predecessorContractViolation(.runtimeDenialAnchor)
    }
    guard reanchored == document else {
      throw FileImageExecutionIdentityComparisonFailure
        .runtimeDenialEvidenceMismatch
    }

    do {
      try GitRuntimePolicyDenialV2Verifier.requireUnavailable(
        policyDocument: reanchored,
        toolPolicyReference: toolReference
      )
    } catch let failure as GitRuntimePolicyDenialV2Error {
      guard failure == .supportedSandboxMechanismUnavailable else {
        throw FileImageExecutionIdentityComparisonFailure
          .runtimeDenialReference(failure)
      }
    } catch {
      throw FileImageExecutionIdentityComparisonFailure
        .predecessorContractViolation(.runtimeDenialReference)
    }
    return reanchored
  }

  static func revalidateExecutable(
    _ comparison: ExecutableContentExpectationComparison,
    anchor: ExecutableIdentityExpectationTrustAnchor,
    role: RuntimeClosureExpectationArtifactRole
  ) throws -> RevalidatedExecutable {
    let expectedExecutableRole = executableRole(for: role)
    let roleSlot: FileImageExecutionIdentityComparisonFailure.RoleSlot =
      role == .git ? .gitExecutableEvidence : .selfGuardExecutableEvidence
    guard comparison.expectation.fields.artifactRole == expectedExecutableRole,
      comparison.contentEvidence.artifactRole == contentRole(for: role)
    else {
      throw FileImageExecutionIdentityComparisonFailure
        .roleSubstitution(slot: roleSlot)
    }

    let expectation: AnchoredExecutableIdentityExpectationDocument
    do {
      expectation = try ExecutableIdentityExpectationVerifier.anchor(
        expectationFile: comparison.expectation.expectationFile,
        trustAnchor: anchor
      )
    } catch let failure as ExecutableIdentityExpectationError {
      throw FileImageExecutionIdentityComparisonFailure.executableExpectation(
        role: role,
        failure: failure
      )
    } catch {
      throw FileImageExecutionIdentityComparisonFailure
        .predecessorContractViolation(executableAnchorPredecessor(for: role))
    }
    guard expectation == comparison.expectation else {
      throw FileImageExecutionIdentityComparisonFailure
        .executableExpectationMismatch(role: role)
    }

    let evidence: ExecutableContentIdentityEvidence
    do {
      evidence = try ExecutableContentIdentityVerifier.derive(
        artifactRole: contentRole(for: role),
        comparison: comparison.contentEvidence.comparison
      )
    } catch let failure as ExecutableContentIdentityFailure {
      throw FileImageExecutionIdentityComparisonFailure.executableEvidence(
        role: role,
        failure: failure
      )
    } catch {
      throw FileImageExecutionIdentityComparisonFailure
        .predecessorContractViolation(executableContentPredecessor(for: role))
    }
    guard evidence == comparison.contentEvidence else {
      throw FileImageExecutionIdentityComparisonFailure
        .executableEvidenceMismatch(role: role)
    }

    let rederived: ExecutableContentExpectationComparison
    do {
      rederived = try ExecutableContentIdentityVerifier.match(
        evidence: evidence,
        expectation: expectation
      )
    } catch let failure as ExecutableContentIdentityFailure {
      throw FileImageExecutionIdentityComparisonFailure.executableComparison(
        role: role,
        failure: failure
      )
    } catch {
      throw FileImageExecutionIdentityComparisonFailure
        .predecessorContractViolation(executableMatchPredecessor(for: role))
    }
    guard rederived == comparison else {
      throw FileImageExecutionIdentityComparisonFailure
        .executableComparisonMismatch(role: role)
    }
    guard hasNoAuthority(comparison) else {
      throw FileImageExecutionIdentityComparisonFailure.executablePolicyContinuity(
        role: role,
        field: .authorityClaims
      )
    }
    return RevalidatedExecutable(
      expectation: expectation,
      evidence: evidence,
      comparison: rederived
    )
  }

  static func revalidateClosure(
    _ expectation: AnchoredRuntimeClosureExpectationDocument,
    anchor: RuntimeClosureExpectationTrustAnchor,
    role: RuntimeClosureExpectationArtifactRole
  ) throws -> AnchoredRuntimeClosureExpectationDocument {
    let roleSlot: FileImageExecutionIdentityComparisonFailure.RoleSlot =
      role == .git ? .gitClosureEvidence : .selfGuardClosureEvidence
    guard expectation.fields.artifactRole == role else {
      throw FileImageExecutionIdentityComparisonFailure
        .roleSubstitution(slot: roleSlot)
    }

    let reanchored: AnchoredRuntimeClosureExpectationDocument
    do {
      reanchored = try RuntimeClosureExpectationVerifier.anchor(
        expectationFile: expectation.expectationFile,
        trustAnchor: anchor
      )
    } catch let failure as RuntimeClosureExpectationError {
      throw FileImageExecutionIdentityComparisonFailure.closureExpectation(
        role: role,
        failure: failure
      )
    } catch {
      throw FileImageExecutionIdentityComparisonFailure
        .predecessorContractViolation(closureAnchorPredecessor(for: role))
    }
    guard reanchored == expectation else {
      throw FileImageExecutionIdentityComparisonFailure
        .closureExpectationMismatch(role: role)
    }
    return reanchored
  }

  static func revalidateClosureReference(
    _ reference: FileImageRuntimeClosureExpectationReference,
    expectation: AnchoredRuntimeClosureExpectationDocument,
    anchor: RuntimeClosureExpectationTrustAnchor,
    role: RuntimeClosureExpectationArtifactRole
  ) throws -> FileImageRuntimeClosureExpectationReference {
    let roleSlot: FileImageExecutionIdentityComparisonFailure.RoleSlot =
      role == .git ? .gitClosureReference : .selfGuardClosureReference
    guard reference.anchoredExpectation.fields.artifactRole == role else {
      throw FileImageExecutionIdentityComparisonFailure
        .roleSubstitution(slot: roleSlot)
    }

    let rederived: FileImageRuntimeClosureExpectationReference
    do {
      rederived = try FileImageRuntimeClosureExpectationVerifier.reference(
        anchoredExpectation: expectation,
        currentExpectationAnchor: anchor
      )
    } catch {
      throw FileImageExecutionIdentityComparisonFailure
        .closureReference(role: role)
    }
    guard rederived == reference else {
      throw FileImageExecutionIdentityComparisonFailure
        .closureReferenceMismatch(role: role)
    }
    return rederived
  }

  static func validateClosureComparison(
    _ comparison: FileImageRuntimeClosureContentExpectationComparison,
    expectation: AnchoredRuntimeClosureExpectationDocument,
    reference: FileImageRuntimeClosureExpectationReference,
    role: RuntimeClosureExpectationArtifactRole
  ) throws {
    let roleSlot: FileImageExecutionIdentityComparisonFailure.RoleSlot =
      role == .git ? .gitClosureComparison : .selfGuardClosureComparison
    guard comparison.artifactRole == role else {
      throw FileImageExecutionIdentityComparisonFailure
        .roleSubstitution(slot: roleSlot)
    }

    try require(
      comparison.manifestSHA256 == expectation.documentSHA256,
      role: role,
      field: .manifestDigest
    )
    try require(
      comparison.manifestBytes == expectation.documentBytes,
      role: role,
      field: .manifestBytes
    )
    try require(
      comparison.rootExecutableContentEvidenceID
        == expectation.fields.rootExecutableContentEvidenceID,
      role: role,
      field: .rootID
    )
    try require(
      comparison.dynamicLoaderContentEvidenceID
        == expectation.fields.dynamicLoaderContentEvidenceID,
      role: role,
      field: .dynamicLoaderID
    )
    try require(
      comparison.sharedCacheSetID
        == expectation.sharedCacheSetEvidence.sharedCacheSetID.sha256,
      role: role,
      field: .cacheSetID
    )
    try require(
      comparison.memberCount == expectation.fields.members.count,
      role: role,
      field: .memberCount
    )
    try require(
      comparison.edgeCount == expectation.fields.edges.count,
      role: role,
      field: .edgeCount
    )
    try require(
      comparison.fileImageMemberCount
        == reference.declaredFileImageMemberCount,
      role: role,
      field: .fileImageMemberCount
    )
    try require(
      comparison.provesExpectationAnchorMatch
        && comparison.provesManifestContentMatch
        && comparison.provesDeclaredStaticGraphMatch
        && comparison.provesSealedFileImageContinuity,
      role: role,
      field: .scopedComparisonClaims
    )
    try require(
      !comparison.isCompleteRuntimeClosure
        && !comparison.provesRuntimeLaunchability
        && comparison.runtimeResolutionOutcome
          == "unproved-static-comparison-only"
        && comparison.runtimeDecision == .noGo,
      role: role,
      field: .runtimeTerminal
    )
    try require(
      hasNoAuthority(comparison),
      role: role,
      field: .authorityClaims
    )
  }

  static func validateContinuity(
    toolReference: SignedClaimGitToolPolicyV2Reference,
    denial: AnchoredGitRuntimePolicyDenialV2Document,
    gitExecutable: RevalidatedExecutable,
    selfGuardExecutable: RevalidatedExecutable,
    gitClosure: AnchoredRuntimeClosureExpectationDocument,
    gitClosureComparison: FileImageRuntimeClosureContentExpectationComparison,
    selfGuardClosure: AnchoredRuntimeClosureExpectationDocument,
    selfGuardClosureComparison: FileImageRuntimeClosureContentExpectationComparison
  ) throws {
    let tool = toolReference.policyDocument.fields
    let git = gitExecutable.expectation.fields
    try requireExecutable(
      git.fileSHA256 == tool.executableSHA256,
      role: .git,
      field: .fileDigest
    )
    try requireExecutable(
      git.fileBytes == tool.executableBytes,
      role: .git,
      field: .fileBytes
    )
    try requireExecutable(
      git.machOUUID == tool.executableMachOUUID,
      role: .git,
      field: .machOUUID
    )
    try requireExecutable(
      git.codeDirectories.first?.blobSHA256
        == tool.executableCodeDirectorySHA256,
      role: .git,
      field: .primaryCodeDirectoryDigest
    )

    let guardFields = selfGuardExecutable.expectation.fields
    try requireExecutable(
      guardFields.fileSHA256 == denial.fields.selfGuardSHA256,
      role: .selfGuard,
      field: .fileDigest
    )
    try requireExecutable(
      guardFields.fileBytes == denial.fields.selfGuardBytes,
      role: .selfGuard,
      field: .fileBytes
    )
    try requireExecutable(
      guardFields.machOUUID == denial.fields.selfGuardMachOUUID,
      role: .selfGuard,
      field: .machOUUID
    )
    try requireExecutable(
      guardFields.codeDirectories.first?.blobSHA256
        == denial.fields.selfGuardCodeDirectorySHA256,
      role: .selfGuard,
      field: .primaryCodeDirectoryDigest
    )

    try require(
      gitClosure.documentSHA256 == tool.gitRuntimeClosureManifestSHA256,
      role: .git,
      field: .manifestDigest
    )
    try require(
      gitClosure.documentBytes == tool.gitRuntimeClosureManifestBytes,
      role: .git,
      field: .manifestBytes
    )
    try require(
      gitClosureComparison.contentEvidenceID.sha256
        == tool.gitFileImageRuntimeClosureContentEvidenceID,
      role: .git,
      field: .toolContentEvidenceID
    )
    try require(
      gitClosureComparison.contentEvidenceID.sha256
        == denial.fields.gitFileImageRuntimeClosureContentEvidenceID,
      role: .git,
      field: .denialContentEvidenceID
    )
    try require(
      selfGuardClosureComparison.contentEvidenceID.sha256
        == denial.fields.selfGuardFileImageRuntimeClosureContentEvidenceID,
      role: .selfGuard,
      field: .denialContentEvidenceID
    )

    guard toolReference.policyDocument.runtimePolicySHA256
      == toolReference.signedClaim.fields.policies.runtimeSHA256,
      toolReference.policyDocument.runtimePolicySHA256
        == denial.policySHA256
    else {
      throw FileImageExecutionIdentityComparisonFailure.runtimePolicyContinuity
    }

    guard gitClosureComparison.dynamicLoaderContentEvidenceID
      == selfGuardClosureComparison.dynamicLoaderContentEvidenceID
    else {
      throw FileImageExecutionIdentityComparisonFailure.sharedEnvironmentMismatch(
        .dynamicLoaderContentID
      )
    }
    guard gitClosureComparison.sharedCacheSetID
      == selfGuardClosureComparison.sharedCacheSetID
    else {
      throw FileImageExecutionIdentityComparisonFailure.sharedEnvironmentMismatch(
        .sharedCacheSetID
      )
    }

    try requirePlatform(
      gitClosure.fields.platformArchitecture == "arm64"
        && selfGuardClosure.fields.platformArchitecture
          == gitClosure.fields.platformArchitecture,
      field: .architecture
    )
    try requirePlatform(
      gitClosure.fields.platformHardwareModel == "Mac15,14"
        && selfGuardClosure.fields.platformHardwareModel
          == gitClosure.fields.platformHardwareModel,
      field: .hardwareModel
    )
    try requirePlatform(
      gitClosure.fields.platformOSVersion == "26.5.2"
        && selfGuardClosure.fields.platformOSVersion
          == gitClosure.fields.platformOSVersion,
      field: .osVersion
    )
    try requirePlatform(
      gitClosure.fields.platformOSBuild == "25F84"
        && selfGuardClosure.fields.platformOSBuild
          == gitClosure.fields.platformOSBuild,
      field: .osBuild
    )
  }

  static func anchorContextPreimage(
    gitTool: GitToolPolicyV2TrustAnchor,
    denial: GitRuntimePolicyDenialV2TrustAnchor,
    gitExecutable: ExecutableIdentityExpectationTrustAnchor,
    selfGuardExecutable: ExecutableIdentityExpectationTrustAnchor,
    gitClosure: RuntimeClosureExpectationTrustAnchor,
    selfGuardClosure: RuntimeClosureExpectationTrustAnchor
  ) -> Data {
    Data(
      ([
        anchorContextDomain,
        "git_tool_policy_v2_expected_sha256="
          + gitTool.expectedCurrentPolicySHA256,
        "git_tool_policy_v2_expected_bytes="
          + "\(gitTool.expectedCurrentPolicyBytes)",
        "git_tool_policy_v2_minimum_generation="
          + "\(gitTool.minimumPolicyGeneration)",
        "git_tool_policy_v2_verification_unix_seconds="
          + "\(gitTool.verificationUnixSeconds)",
        "runtime_denial_v2_expected_sha256="
          + denial.expectedCurrentPolicySHA256,
        "runtime_denial_v2_expected_bytes="
          + "\(denial.expectedCurrentPolicyBytes)",
        "runtime_denial_v2_minimum_generation="
          + "\(denial.minimumPolicyGeneration)",
        "runtime_denial_v2_verification_unix_seconds="
          + "\(denial.verificationUnixSeconds)",
        "git_executable_expectation_sha256="
          + gitExecutable.expectedCurrentDocumentSHA256,
        "git_executable_expectation_bytes="
          + "\(gitExecutable.expectedCurrentDocumentBytes)",
        "git_executable_minimum_generation="
          + "\(gitExecutable.minimumEvidenceGeneration)",
        "git_executable_verification_unix_seconds="
          + "\(gitExecutable.verificationUnixSeconds)",
        "self_guard_executable_expectation_sha256="
          + selfGuardExecutable.expectedCurrentDocumentSHA256,
        "self_guard_executable_expectation_bytes="
          + "\(selfGuardExecutable.expectedCurrentDocumentBytes)",
        "self_guard_executable_minimum_generation="
          + "\(selfGuardExecutable.minimumEvidenceGeneration)",
        "self_guard_executable_verification_unix_seconds="
          + "\(selfGuardExecutable.verificationUnixSeconds)",
        "git_closure_expectation_sha256="
          + gitClosure.expectedCurrentDocumentSHA256,
        "git_closure_expectation_bytes="
          + "\(gitClosure.expectedCurrentDocumentBytes)",
        "git_closure_minimum_generation="
          + "\(gitClosure.minimumEvidenceGeneration)",
        "git_closure_verification_unix_seconds="
          + "\(gitClosure.verificationUnixSeconds)",
        "self_guard_closure_expectation_sha256="
          + selfGuardClosure.expectedCurrentDocumentSHA256,
        "self_guard_closure_expectation_bytes="
          + "\(selfGuardClosure.expectedCurrentDocumentBytes)",
        "self_guard_closure_minimum_generation="
          + "\(selfGuardClosure.minimumEvidenceGeneration)",
        "self_guard_closure_verification_unix_seconds="
          + "\(selfGuardClosure.verificationUnixSeconds)",
      ].joined(separator: "\n") + "\n").utf8
    )
  }

  static func comparisonPreimage(
    runClaimID: String,
    anchorContextID: FileImageExecutionIdentityAnchorContextID,
    toolPolicy: AnchoredGitToolPolicyV2Document,
    denial: AnchoredGitRuntimePolicyDenialV2Document,
    gitExecutable: RevalidatedExecutable,
    selfGuardExecutable: RevalidatedExecutable,
    gitClosure: AnchoredRuntimeClosureExpectationDocument,
    gitClosureComparison: FileImageRuntimeClosureContentExpectationComparison,
    selfGuardClosure: AnchoredRuntimeClosureExpectationDocument,
    selfGuardClosureComparison: FileImageRuntimeClosureContentExpectationComparison
  ) -> Data {
    Data(
      ([
        comparisonDomain,
        "run_claim_id=\(runClaimID)",
        "anchor_context_id=\(anchorContextID.sha256)",
        "git_tool_policy_v2_sha256=\(toolPolicy.policySHA256)",
        "runtime_denial_policy_v2_sha256=\(denial.policySHA256)",
        "git_executable_expectation_sha256="
          + gitExecutable.expectation.documentSHA256,
        "git_executable_content_evidence_id="
          + gitExecutable.evidence.contentEvidenceID.sha256,
        "self_guard_executable_expectation_sha256="
          + selfGuardExecutable.expectation.documentSHA256,
        "self_guard_executable_content_evidence_id="
          + selfGuardExecutable.evidence.contentEvidenceID.sha256,
        "git_closure_manifest_sha256=\(gitClosure.documentSHA256)",
        "git_file_image_runtime_closure_content_evidence_id="
          + gitClosureComparison.contentEvidenceID.sha256,
        "self_guard_closure_manifest_sha256="
          + selfGuardClosure.documentSHA256,
        "self_guard_file_image_runtime_closure_content_evidence_id="
          + selfGuardClosureComparison.contentEvidenceID.sha256,
        "runtime_decision=no-go",
      ].joined(separator: "\n") + "\n").utf8
    )
  }

  static func require(
    _ condition: Bool,
    role: RuntimeClosureExpectationArtifactRole,
    field: FileImageExecutionIdentityComparisonFailure.ClosureField
  ) throws {
    guard condition else {
      throw FileImageExecutionIdentityComparisonFailure.closureContinuity(
        role: role,
        field: field
      )
    }
  }

  static func requireExecutable(
    _ condition: Bool,
    role: RuntimeClosureExpectationArtifactRole,
    field: FileImageExecutionIdentityComparisonFailure.ExecutableField
  ) throws {
    guard condition else {
      throw FileImageExecutionIdentityComparisonFailure.executablePolicyContinuity(
        role: role,
        field: field
      )
    }
  }

  static func requirePlatform(
    _ condition: Bool,
    field: FileImageExecutionIdentityComparisonFailure.PlatformField
  ) throws {
    guard condition else {
      throw FileImageExecutionIdentityComparisonFailure.platformMismatch(field)
    }
  }

  static func executableRole(
    for role: RuntimeClosureExpectationArtifactRole
  ) -> ExecutableIdentityArtifactRole {
    role == .git ? .git : .selfGuard
  }

  static func contentRole(
    for role: RuntimeClosureExpectationArtifactRole
  ) -> ExecutableContentArtifactRole {
    role == .git ? .git : .selfGuard
  }

  static func executableAnchorPredecessor(
    for role: RuntimeClosureExpectationArtifactRole
  ) -> FileImageExecutionIdentityComparisonFailure.Predecessor {
    role == .git ? .gitExecutableAnchor : .selfGuardExecutableAnchor
  }

  static func executableContentPredecessor(
    for role: RuntimeClosureExpectationArtifactRole
  ) -> FileImageExecutionIdentityComparisonFailure.Predecessor {
    role == .git ? .gitExecutableContent : .selfGuardExecutableContent
  }

  static func executableMatchPredecessor(
    for role: RuntimeClosureExpectationArtifactRole
  ) -> FileImageExecutionIdentityComparisonFailure.Predecessor {
    role == .git ? .gitExecutableMatch : .selfGuardExecutableMatch
  }

  static func closureAnchorPredecessor(
    for role: RuntimeClosureExpectationArtifactRole
  ) -> FileImageExecutionIdentityComparisonFailure.Predecessor {
    role == .git ? .gitClosureAnchor : .selfGuardClosureAnchor
  }

  static func hasNoAuthority(
    _ comparison: ExecutableContentExpectationComparison
  ) -> Bool {
    !comparison.canExecute && !comparison.canSpawn
      && !comparison.canAccessNetwork && !comparison.canConsumePack
      && !comparison.canMutateFileSystem
      && !comparison.canImportGitObjects && !comparison.canBuild
      && !comparison.canLoadModel && !comparison.canReserveOutput
      && !comparison.canPublish
  }

  static func hasNoAuthority(
    _ comparison: FileImageRuntimeClosureContentExpectationComparison
  ) -> Bool {
    !comparison.canExecute && !comparison.canSpawn
      && !comparison.canAccessNetwork && !comparison.canConsumePack
      && !comparison.canMutateFileSystem
      && !comparison.canImportGitObjects && !comparison.canBuild
      && !comparison.canLoadModel && !comparison.canReserveOutput
      && !comparison.canPublish
  }

  static func isLowercaseHex(_ value: String, count: Int) -> Bool {
    let bytes = value.utf8
    guard bytes.count == count else { return false }
    return bytes.allSatisfy {
      (UInt8(ascii: "0")...UInt8(ascii: "9")).contains($0)
        || (UInt8(ascii: "a")...UInt8(ascii: "f")).contains($0)
    }
  }

  static func sha256Hex(_ bytes: Data) -> String {
    SHA256.hash(data: bytes)
      .map { String(format: "%02x", $0) }
      .joined()
  }
}
