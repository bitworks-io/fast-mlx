import CryptoKit
import Foundation
import XCTest

@testable import ProofControl

final class ExecutionIdentityComparisonTests: XCTestCase {
  func testPrimaryStageC3TypesExist() {
    _ = ExecutionIdentityAnchorContextID.self
    _ = ExecutionIdentityComparisonEvidence.self
    _ = ExecutionIdentityComparisonFailure.self
    _ = ExecutionIdentityComparisonVerifier.self
  }

  func testCanonicalSingletonDerivesExactCompactNoGoEvidence() throws {
    let fixture = try Self.fixture()
    let result = try Self.compare(fixture)
    let repeated = try Self.compare(fixture)
    let context = Self.contextPreimage(fixture)
    let final = Self.finalPreimage(fixture, contextID: Self.sha256Hex(context))
    let expectedContext = Data(
      ("""
      fast-mlx-proof-control-execution-identity-anchor-context-id-v1
      git_tool_policy_expected_sha256=6f7ca3d829fe846fe03a742cb49ce196b7ae059e7f358b73e5d66987df35882e
      git_tool_policy_expected_bytes=1784
      git_tool_policy_minimum_generation=7
      git_tool_policy_verification_unix_seconds=2000000000
      runtime_denial_expected_sha256=cd6eba353f148cad266741326c7d5984c09a6dd35e51480ee8333cb4d43fca27
      runtime_denial_expected_bytes=2393
      runtime_denial_minimum_generation=7
      runtime_denial_verification_unix_seconds=2000000000
      git_executable_expectation_sha256=e843ae1af1c9b248c60db7ae90617b77a54191c80436a30ec88cacdfe96c6ae1
      git_executable_expectation_bytes=1446
      git_executable_minimum_generation=9
      git_executable_verification_unix_seconds=2000000000
      self_guard_executable_expectation_sha256=0ec1ca27fd24ec29422362de6c3af9b242447bb25ccfd0934d53db03b5f6517f
      self_guard_executable_expectation_bytes=1463
      self_guard_executable_minimum_generation=9
      self_guard_executable_verification_unix_seconds=2000000000
      git_closure_expectation_sha256=a49347c7ced37396281b66a1f6bac2f17720376f40b86a4f68d4d6c2fe76ee79
      git_closure_expectation_bytes=1999
      git_closure_minimum_generation=9
      git_closure_verification_unix_seconds=2000000000
      self_guard_closure_expectation_sha256=31026c03d74aae99a3da5473f5619ef8a1220f5df25c58146276ae6d73bfb99f
      self_guard_closure_expectation_bytes=2006
      self_guard_closure_minimum_generation=9
      self_guard_closure_verification_unix_seconds=2000000000
      """ + "\n").utf8
    )
    let expectedFinal = Data(
      ("""
      fast-mlx-proof-control-execution-identity-comparison-id-v1
      run_claim_id=f4a00fa58933b5397ee2b52d4660de78518f01d83b38d261321702da5d2d5579
      anchor_context_id=a02cc962d79fc0f8b0af7de490bbc25cc47de447da89520c90bd2b8de372304e
      git_tool_policy_sha256=6f7ca3d829fe846fe03a742cb49ce196b7ae059e7f358b73e5d66987df35882e
      runtime_denial_policy_sha256=cd6eba353f148cad266741326c7d5984c09a6dd35e51480ee8333cb4d43fca27
      git_executable_expectation_sha256=e843ae1af1c9b248c60db7ae90617b77a54191c80436a30ec88cacdfe96c6ae1
      git_executable_content_evidence_id=48926011d3c1e00791bb3f8400f4c6325f334f52112b48b4fbe755c8059b2743
      self_guard_executable_expectation_sha256=0ec1ca27fd24ec29422362de6c3af9b242447bb25ccfd0934d53db03b5f6517f
      self_guard_executable_content_evidence_id=603851004262bf63df89ca19bec1deda73781785c9e89837cdf2004a9bb657e3
      git_closure_manifest_sha256=a49347c7ced37396281b66a1f6bac2f17720376f40b86a4f68d4d6c2fe76ee79
      git_closure_content_evidence_id=9a380befb74183284691595e4f9959feb6a43249552a86063c7b45cdf7c92506
      self_guard_closure_manifest_sha256=31026c03d74aae99a3da5473f5619ef8a1220f5df25c58146276ae6d73bfb99f
      self_guard_closure_content_evidence_id=ac05a369e3f2620e48217c6d0b3bab53fb0d90bd6a49fa4c64b1bd1a017d8ca5
      runtime_decision=no-go
      """ + "\n").utf8
    )

    XCTAssertEqual(result, repeated)
    XCTAssertEqual(context, expectedContext)
    XCTAssertEqual(final, expectedFinal)
    XCTAssertEqual(result.anchorContextID.sha256, Self.sha256Hex(context))
    XCTAssertEqual(result.comparisonSHA256, Self.sha256Hex(final))
    XCTAssertEqual(
      result.anchorContextID.sha256,
      "a02cc962d79fc0f8b0af7de490bbc25cc47de447da89520c90bd2b8de372304e"
    )
    XCTAssertEqual(
      result.comparisonSHA256,
      "1d242a19f417c676f5c68e8239624ba4248dd90c22303bfc78157ea03c0cb132"
    )
    XCTAssertEqual(result.runClaimID, fixture.reference.signedClaim.claimID.rawValue)
    XCTAssertEqual(result.gitRole, .git)
    XCTAssertEqual(result.selfGuardRole, .selfGuard)
    XCTAssertEqual(result.runtimeDecision, .noGo)
    XCTAssertFalse(result.canExecute)
    XCTAssertFalse(result.canSpawn)
    XCTAssertFalse(result.canAccessNetwork)
    XCTAssertFalse(result.canConsumePack)
    XCTAssertFalse(result.canMutateFileSystem)
    XCTAssertFalse(result.canImportGitObjects)
    XCTAssertFalse(result.canBuild)
    XCTAssertFalse(result.canLoadModel)
    XCTAssertFalse(result.canReserveOutput)
    XCTAssertFalse(result.canPublish)
    XCTAssertEqual(Self.effects, .zero)

    let labels = Set(Mirror(reflecting: result).children.compactMap(\.label))
    for forbidden in [
      "signedClaim", "policyDocument", "trustAnchor", "expectationFile",
      "contentEvidence", "comparison", "graph", "members", "edges",
      "path", "fileDescriptor", "argv", "environment", "process",
      "sandbox", "capability", "nonce", "receipt", "resultRoot",
    ] {
      XCTAssertFalse(labels.contains(forbidden))
    }
  }

  func testAnchorScalarsRolesAndCommonTimeWinBeforePredecessors() throws {
    let fixture = try Self.fixture()

    Self.assertFailure(
      fixture,
      gitToolAnchor: GitToolPolicyTrustAnchor(
        expectedCurrentPolicySHA256: fixture.tool.policySHA256.uppercased(),
        expectedCurrentPolicyBytes: fixture.tool.policyBytes,
        minimumPolicyGeneration: 7,
        verificationUnixSeconds: Self.commonTime
      ),
      expected: .invalidAnchorScalar(slot: .gitToolPolicy, field: .digest)
    )
    Self.assertFailure(
      fixture,
      runtimeDenialAnchor: GitRuntimePolicyDenialTrustAnchor(
        expectedCurrentPolicySHA256: fixture.denial.policySHA256,
        expectedCurrentPolicyBytes: 0,
        minimumPolicyGeneration: 7,
        verificationUnixSeconds: Self.commonTime
      ),
      expected: .invalidAnchorScalar(slot: .runtimeDenial, field: .byteCount)
    )
    Self.assertFailure(
      fixture,
      gitExecutableAnchor: ExecutableIdentityExpectationTrustAnchor(
        expectedCurrentDocumentSHA256:
          fixture.gitExecutable.expectation.documentSHA256,
        expectedCurrentDocumentBytes:
          fixture.gitExecutable.expectation.documentBytes,
        minimumEvidenceGeneration: 0,
        verificationUnixSeconds: Self.commonTime,
        expectedArtifactRole: .git
      ),
      expected: .invalidAnchorScalar(
        slot: .gitExecutableExpectation,
        field: .minimumGeneration
      )
    )
    Self.assertFailure(
      fixture,
      selfGuardClosureAnchor: RuntimeClosureExpectationTrustAnchor(
        expectedCurrentDocumentSHA256:
          fixture.selfGuardClosure.expectation.documentSHA256,
        expectedCurrentDocumentBytes:
          fixture.selfGuardClosure.expectation.documentBytes,
        minimumEvidenceGeneration: 9,
        verificationUnixSeconds: Self.commonTime,
        expectedArtifactRole: .git
      ),
      expected: .roleSubstitution(slot: .selfGuardClosureAnchor)
    )

    let expiredAndMixed = GitRuntimePolicyDenialTrustAnchor(
      expectedCurrentPolicySHA256: fixture.denial.policySHA256,
      expectedCurrentPolicyBytes: fixture.denial.policyBytes,
      minimumPolicyGeneration: 7,
      verificationUnixSeconds: 2_100_000_001
    )
    Self.assertFailure(
      fixture,
      runtimeDenialAnchor: expiredAndMixed,
      expected: .anchorEvaluationTimeMismatch(.runtimeDenial)
    )
    XCTAssertEqual(Self.effects, .zero)
  }

  func testEveryAnchorRejectsDigestBytesAndGenerationBeforeEvidence() throws {
    let fixture = try Self.fixture()
    let invalidDigest = String(repeating: "A", count: 64)

    Self.assertFailure(
      fixture,
      gitToolAnchor: Self.toolAnchor(fixture, digest: invalidDigest),
      expected: .invalidAnchorScalar(slot: .gitToolPolicy, field: .digest)
    )
    Self.assertFailure(
      fixture,
      gitToolAnchor: Self.toolAnchor(fixture, bytes: 0),
      expected: .invalidAnchorScalar(slot: .gitToolPolicy, field: .byteCount)
    )
    Self.assertFailure(
      fixture,
      gitToolAnchor: Self.toolAnchor(fixture, generation: 0),
      expected: .invalidAnchorScalar(
        slot: .gitToolPolicy,
        field: .minimumGeneration
      )
    )

    Self.assertFailure(
      fixture,
      runtimeDenialAnchor: Self.denialAnchor(
        fixture,
        digest: invalidDigest
      ),
      expected: .invalidAnchorScalar(slot: .runtimeDenial, field: .digest)
    )
    Self.assertFailure(
      fixture,
      runtimeDenialAnchor: Self.denialAnchor(fixture, bytes: 0),
      expected: .invalidAnchorScalar(slot: .runtimeDenial, field: .byteCount)
    )
    Self.assertFailure(
      fixture,
      runtimeDenialAnchor: Self.denialAnchor(fixture, generation: 0),
      expected: .invalidAnchorScalar(
        slot: .runtimeDenial,
        field: .minimumGeneration
      )
    )

    Self.assertExecutableAnchorScalarFailures(
      fixture,
      original: fixture.gitExecutableAnchor,
      slot: .gitExecutableExpectation,
      git: true,
      invalidDigest: invalidDigest
    )
    Self.assertExecutableAnchorScalarFailures(
      fixture,
      original: fixture.selfGuardExecutableAnchor,
      slot: .selfGuardExecutableExpectation,
      git: false,
      invalidDigest: invalidDigest
    )
    Self.assertClosureAnchorScalarFailures(
      fixture,
      original: fixture.gitClosure.anchor,
      slot: .gitClosureExpectation,
      git: true,
      invalidDigest: invalidDigest
    )
    Self.assertClosureAnchorScalarFailures(
      fixture,
      original: fixture.selfGuardClosure.anchor,
      slot: .selfGuardClosureExpectation,
      git: false,
      invalidDigest: invalidDigest
    )
    XCTAssertEqual(Self.effects, .zero)
  }

  func testEveryRoleBearingAnchorRejectsSubstitution() throws {
    let fixture = try Self.fixture()
    Self.assertFailure(
      fixture,
      gitExecutableAnchor: Self.executableAnchor(
        fixture.gitExecutableAnchor,
        role: .selfGuard
      ),
      expected: .roleSubstitution(slot: .gitExecutableAnchor)
    )
    Self.assertFailure(
      fixture,
      selfGuardExecutableAnchor: Self.executableAnchor(
        fixture.selfGuardExecutableAnchor,
        role: .git
      ),
      expected: .roleSubstitution(slot: .selfGuardExecutableAnchor)
    )
    Self.assertFailure(
      fixture,
      gitClosureAnchor: Self.closureAnchor(
        fixture.gitClosure.anchor,
        role: .selfGuard
      ),
      expected: .roleSubstitution(slot: .gitClosureAnchor)
    )
    Self.assertFailure(
      fixture,
      selfGuardClosureAnchor: Self.closureAnchor(
        fixture.selfGuardClosure.anchor,
        role: .git
      ),
      expected: .roleSubstitution(slot: .selfGuardClosureAnchor)
    )
    XCTAssertEqual(Self.effects, .zero)
  }

  func testMixedDenialExecutableAndClosureTimesPrecedeValidity() throws {
    let fixture = try Self.fixture()
    let expired: UInt64 = 2_100_000_001
    Self.assertFailure(
      fixture,
      runtimeDenialAnchor: Self.denialAnchor(fixture, time: expired),
      expected: .anchorEvaluationTimeMismatch(.runtimeDenial)
    )
    Self.assertFailure(
      fixture,
      gitExecutableAnchor: Self.executableAnchor(
        fixture.gitExecutableAnchor,
        time: expired
      ),
      expected: .anchorEvaluationTimeMismatch(.gitExecutableExpectation)
    )
    Self.assertFailure(
      fixture,
      gitClosureAnchor: Self.closureAnchor(
        fixture.gitClosure.anchor,
        time: expired
      ),
      expected: .anchorEvaluationTimeMismatch(.gitClosureExpectation)
    )
    XCTAssertEqual(Self.effects, .zero)
  }

  func testCurrentEvidenceReanchorsAndPreservesTypedFailures() throws {
    let fixture = try Self.fixture()
    let badToolAnchor = GitToolPolicyTrustAnchor(
      expectedCurrentPolicySHA256: String(repeating: "f", count: 64),
      expectedCurrentPolicyBytes: fixture.tool.policyBytes,
      minimumPolicyGeneration: 7,
      verificationUnixSeconds: Self.commonTime
    )
    Self.assertFailure(
      fixture,
      gitToolAnchor: badToolAnchor,
      expected: .gitToolPolicyReanchor(.policyDigestMismatch)
    )

    let changedExecutableAnchor = ExecutableIdentityExpectationTrustAnchor(
      expectedCurrentDocumentSHA256:
        fixture.gitExecutable.expectation.documentSHA256,
      expectedCurrentDocumentBytes:
        fixture.gitExecutable.expectation.documentBytes,
      minimumEvidenceGeneration: 9,
      verificationUnixSeconds: Self.commonTime + 1,
      expectedArtifactRole: .git
    )
    Self.assertFailure(
      fixture,
      allTimes: Self.commonTime + 1,
      gitExecutableAnchor: changedExecutableAnchor,
      expected: .executableExpectationMismatch(role: .git)
    )
    XCTAssertEqual(Self.effects, .zero)
  }

  func testRuntimeDenialRejectsDigestAndResourceDiscontinuity() throws {
    let fixture = try Self.fixture()
    let changedDenial = try Self.fixture(selfGuardCacheSeed: "2")
    Self.assertFailure(
      Self.replacing(fixture, denialWith: changedDenial),
      expected: .runtimeDenialReference(.runtimePolicyDigestMismatch)
    )

    let resourceMismatch = try Self.fixture(
      denialLimits: Self.phase1Limits(
        maxOpenFiles:
          GitToolPolicyVerifier.phase1ResourceCeilings.maxOpenFiles - 1
      )
    )
    Self.assertFailure(
      resourceMismatch,
      expected: .runtimeDenialReference(
        .toolPolicyResourceMismatch(.maxOpenFiles)
      )
    )
    XCTAssertEqual(Self.effects, .zero)
  }

  func testSealedRoleAndPrefixSubstitutionRefusesBeforeJoins() throws {
    let fixture = try Self.fixture()
    let changedGit = try Self.fixture(gitRootUUIDSeed: 0x11)

    Self.assertFailure(
      Self.replacing(
        fixture,
        gitExecutableComparisonWith: fixture.selfGuardExecutable
      ),
      expected: .roleSubstitution(slot: .gitExecutableEvidence)
    )
    Self.assertFailure(
      Self.replacing(
        fixture,
        gitClosureWith: fixture.selfGuardClosure,
        keepGitClosureAnchor: true
      ),
      expected: .roleSubstitution(slot: .gitClosureEvidence)
    )

    let mixedClosure = Self.mixingClosure(
      expectationFrom: fixture.gitClosure,
      comparisonFrom: changedGit.gitClosure
    )
    Self.assertFailure(
      Self.replacing(fixture, gitClosureWith: mixedClosure),
      expected: .closureContinuity(role: .git, field: .manifestDigest)
    )
    XCTAssertEqual(Self.effects, .zero)
  }

  func testExecutableManifestAndDirectionalDenialJoinsRefuseDrift() throws {
    let fixture = try Self.fixture()
    let changedGit = try Self.fixture(gitRootUUIDSeed: 0x11)
    let changedSelfGuard = try Self.fixture(selfGuardRootUUIDSeed: 0x41)

    Self.assertFailure(
      Self.replacing(fixture, gitExecutableWith: changedGit),
      expected: .executablePolicyContinuity(
        role: .git,
        field: .fileDigest
      )
    )
    Self.assertFailure(
      try Self.fixture(
        toolExecutableBytes:
          fixture.gitExecutable.expectation.fields.fileBytes + 1
      ),
      expected: .executablePolicyContinuity(
        role: .git,
        field: .fileBytes
      )
    )
    Self.assertFailure(
      try Self.fixture(
        toolExecutableMachOUUID: String(repeating: "e", count: 32)
      ),
      expected: .executablePolicyContinuity(
        role: .git,
        field: .machOUUID
      )
    )
    Self.assertFailure(
      try Self.fixture(
        toolExecutableCodeDirectorySHA256:
          String(repeating: "e", count: 64)
      ),
      expected: .executablePolicyContinuity(
        role: .git,
        field: .primaryCodeDirectoryDigest
      )
    )
    Self.assertFailure(
      Self.replacing(
        fixture,
        selfGuardExecutableWith: changedSelfGuard
      ),
      expected: .executablePolicyContinuity(
        role: .selfGuard,
        field: .fileDigest
      )
    )
    Self.assertFailure(
      try Self.fixture(
        denialSelfGuardBytes:
          fixture.selfGuardExecutable.expectation.fields.fileBytes + 1
      ),
      expected: .executablePolicyContinuity(
        role: .selfGuard,
        field: .fileBytes
      )
    )
    Self.assertFailure(
      try Self.fixture(
        denialSelfGuardMachOUUID: String(repeating: "e", count: 32)
      ),
      expected: .executablePolicyContinuity(
        role: .selfGuard,
        field: .machOUUID
      )
    )
    Self.assertFailure(
      try Self.fixture(
        denialSelfGuardCodeDirectorySHA256:
          String(repeating: "e", count: 64)
      ),
      expected: .executablePolicyContinuity(
        role: .selfGuard,
        field: .primaryCodeDirectoryDigest
      )
    )
    Self.assertFailure(
      Self.replacing(fixture, gitClosureWith: changedGit.gitClosure),
      expected: .closureContinuity(role: .git, field: .manifestDigest)
    )
    Self.assertFailure(
      try Self.fixture(
        denialGitClosureID: String(repeating: "e", count: 64)
      ),
      expected: .closureContinuity(role: .git, field: .denialClosureID)
    )
    Self.assertFailure(
      try Self.fixture(
        denialSelfGuardClosureID: String(repeating: "e", count: 64)
      ),
      expected: .closureContinuity(
        role: .selfGuard,
        field: .denialClosureID
      )
    )
    XCTAssertEqual(Self.effects, .zero)
  }

  func testSharedCacheEnvironmentMismatchFailsBeforeIdentity() throws {
    let fixture = try Self.fixture(selfGuardCacheSeed: "2")
    Self.assertFailure(
      fixture,
      expected: .sharedEnvironmentMismatch(.sharedCacheSetID)
    )
    XCTAssertEqual(Self.effects, .zero)
  }

  func testDynamicLoaderEnvironmentMismatchFailsBeforeIdentity() throws {
    let fixture = try Self.fixture(selfGuardLoaderSeed: 0x21)
    Self.assertFailure(
      fixture,
      expected: .sharedEnvironmentMismatch(.dynamicLoaderContentID)
    )
    XCTAssertEqual(Self.effects, .zero)
  }

  func testCallerOwnedByteMutationCannotChangeRetainedEvidence() throws {
    let fixture = try Self.fixture()
    let before = try Self.compare(fixture)
    let retained = fixture.gitClosure.expectation.expectationFile.bytes
    var callerOwned = retained
    callerOwned[callerOwned.startIndex] ^= 0xff

    XCTAssertNotEqual(callerOwned, retained)
    XCTAssertEqual(
      fixture.gitClosure.expectation.expectationFile.bytes,
      retained
    )
    XCTAssertEqual(try Self.compare(fixture), before)
    XCTAssertEqual(Self.effects, .zero)
  }

  func testEveryContextAndFinalPayloadMutationChangesItsID() throws {
    let fixture = try Self.fixture()
    let context = Self.contextPreimage(fixture)
    let contextLines = String(decoding: context, as: UTF8.self)
      .split(separator: "\n", omittingEmptySubsequences: false)
      .map(String.init)
    XCTAssertEqual(contextLines.count, 26)
    XCTAssertTrue(contextLines.last?.isEmpty == true)
    let contextID = Self.sha256Hex(context)
    for index in 1...24 {
      var changed = contextLines
      changed[index] += "0"
      XCTAssertNotEqual(
        Self.sha256Hex(Data(changed.joined(separator: "\n").utf8)),
        contextID
      )
    }

    let final = Self.finalPreimage(fixture, contextID: contextID)
    let finalLines = String(decoding: final, as: UTF8.self)
      .split(separator: "\n", omittingEmptySubsequences: false)
      .map(String.init)
    XCTAssertEqual(finalLines.count, 15)
    XCTAssertTrue(finalLines.last?.isEmpty == true)
    let finalID = Self.sha256Hex(final)
    for index in 1...12 {
      var changed = finalLines
      changed[index] += "0"
      XCTAssertNotEqual(
        Self.sha256Hex(Data(changed.joined(separator: "\n").utf8)),
        finalID
      )
    }
    var changedTerminal = finalLines
    changedTerminal[13] = "runtime_decision=go"
    XCTAssertNotEqual(
      Self.sha256Hex(Data(changedTerminal.joined(separator: "\n").utf8)),
      finalID
    )
    XCTAssertEqual(Self.effects, .zero)
  }
}

private extension ExecutionIdentityComparisonTests {
  struct Effects: Equatable {
    var processSpawns = 0
    var fileSystemMutations = 0
    var networkAccesses = 0
    var packConsumptions = 0
    var sourceMutations = 0
    var buildOperations = 0
    var modelLoads = 0
    var outputReservations = 0
    var publications = 0

    static let zero = Self()
  }

  struct ClosureFixture {
    let role: RuntimeClosureExpectationArtifactRole
    let root: ExecutableContentIdentityEvidence
    let loader: DynamicLoaderContentIdentityEvidence
    let cacheSet: SyntheticSharedCacheSetIdentityEvidence
    let members: [SyntheticRuntimeClosureMemberRecordComparison]
    let edges: [SyntheticRuntimeClosureEdgeRecordComparison]
    let collection: SyntheticRuntimeClosureRecordCollectionComparison
    let rootInventory: SyntheticAcceptedDependencyCommandInventoryComparison
    let memberInventories:
      [SyntheticAcceptedDependencyCommandInventoryComparison]
    let graph: SyntheticRuntimeClosureGraphComparison
    let anchor: RuntimeClosureExpectationTrustAnchor
    let expectation: AnchoredRuntimeClosureExpectationDocument
    let comparison: RuntimeClosureContentExpectationComparison
  }

  struct Fixture {
    let reference: SignedClaimGitToolPolicyReference
    let toolAnchor: GitToolPolicyTrustAnchor
    let tool: AnchoredGitToolPolicyDocument
    let denialAnchor: GitRuntimePolicyDenialTrustAnchor
    let denial: AnchoredGitRuntimePolicyDenialDocument
    let gitExecutableAnchor: ExecutableIdentityExpectationTrustAnchor
    let gitExecutable: ExecutableContentExpectationComparison
    let selfGuardExecutableAnchor:
      ExecutableIdentityExpectationTrustAnchor
    let selfGuardExecutable: ExecutableContentExpectationComparison
    let gitClosure: ClosureFixture
    let selfGuardClosure: ClosureFixture
  }

  struct TestKey {
    // gitleaks:allow -- deterministic synthetic unit-test material only.
    let privateKey: Curve25519.Signing.PrivateKey  // gitleaks:allow
    let publicKeyBase64: String
    let keyID: String
  }

  static let commonTime: UInt64 = 2_000_000_000
  static let installNameValue = "/usr/lib/libStageC3Fixture.dylib"
  static let effects = Effects()

  static func fixture(
    selfGuardCacheSeed: Character = "1",
    selfGuardLoaderSeed: UInt8 = 0x20,
    gitRootUUIDSeed: UInt8? = nil,
    selfGuardRootUUIDSeed: UInt8? = nil,
    toolExecutableBytes: UInt64? = nil,
    toolExecutableMachOUUID: String? = nil,
    toolExecutableCodeDirectorySHA256: String? = nil,
    denialSelfGuardBytes: UInt64? = nil,
    denialSelfGuardMachOUUID: String? = nil,
    denialSelfGuardCodeDirectorySHA256: String? = nil,
    denialGitClosureID: String? = nil,
    denialSelfGuardClosureID: String? = nil,
    denialLimits: GitToolPolicyResourceLimits? = nil
  ) throws -> Fixture {
    let gitLoader = try dynamicLoaderEvidence(seed: 0x20)
    let selfGuardLoader = try dynamicLoaderEvidence(
      seed: selfGuardLoaderSeed
    )
    let gitCache = try cacheSet(seed: "1")
    let selfGuardCache = try cacheSet(seed: selfGuardCacheSeed)
    let gitClosure = try closure(
      role: .git,
      cacheSet: gitCache,
      loader: gitLoader,
      rootUUIDSeed: gitRootUUIDSeed
    )
    let selfGuardClosure = try closure(
      role: .selfGuard,
      cacheSet: selfGuardCache,
      loader: selfGuardLoader,
      rootUUIDSeed: selfGuardRootUUIDSeed
    )
    let (gitExecutable, gitExecutableAnchor) =
      try executableComparison(
        evidence: gitClosure.root,
        role: .git
      )
    let (selfGuardExecutable, selfGuardExecutableAnchor) =
      try executableComparison(
        evidence: selfGuardClosure.root,
        role: .selfGuard
      )
    let selfGuardPrimaryCodeDirectorySHA256 = try
      denialSelfGuardCodeDirectorySHA256
      ?? XCTUnwrap(
        selfGuardExecutable.expectation.fields
          .codeDirectories.first?.blobSHA256
      )
    let gitPrimaryCodeDirectorySHA256 = try
      toolExecutableCodeDirectorySHA256
      ?? XCTUnwrap(
        gitExecutable.expectation.fields
          .codeDirectories.first?.blobSHA256
      )

    let denialFields = GitRuntimePolicyDenialFields(
      policyGeneration: 7,
      validFromUnixSeconds: 1_900_000_000,
      validUntilUnixSeconds: 2_100_000_000,
      selfGuardSHA256:
        selfGuardExecutable.expectation.fields.fileSHA256,
      selfGuardBytes:
        denialSelfGuardBytes
          ?? selfGuardExecutable.expectation.fields.fileBytes,
      selfGuardMachOUUID:
        denialSelfGuardMachOUUID
          ?? selfGuardExecutable.expectation.fields.machOUUID,
      selfGuardCodeDirectorySHA256:
        selfGuardPrimaryCodeDirectorySHA256,
      selfGuardRuntimeClosureID:
        denialSelfGuardClosureID
          ?? selfGuardClosure.comparison.contentEvidenceID.sha256,
      gitRuntimeClosureID:
        denialGitClosureID
          ?? gitClosure.comparison.contentEvidenceID.sha256,
      limits: denialLimits ?? GitToolPolicyVerifier.phase1ResourceCeilings,
      terminationGraceMilliseconds: 2_000,
      reapTimeoutMilliseconds: 5_000
    )
    let denialFile = admittedFile(
      try GitRuntimePolicyDenialVerifier.policyBytes(
        fields: denialFields
      )
    )
    let denialAnchor = GitRuntimePolicyDenialTrustAnchor(
      expectedCurrentPolicySHA256: denialFile.sha256,
      expectedCurrentPolicyBytes: UInt64(denialFile.bytes.count),
      minimumPolicyGeneration: 7,
      verificationUnixSeconds: commonTime
    )
    let denial = try GitRuntimePolicyDenialVerifier.anchor(
      policyFile: denialFile,
      trustAnchor: denialAnchor
    )

    let toolFields = GitToolPolicyFields(
      policyGeneration: 7,
      validFromUnixSeconds: 1_900_000_000,
      validUntilUnixSeconds: 2_100_000_000,
      executableSHA256:
        gitExecutable.expectation.fields.fileSHA256,
      executableBytes:
        toolExecutableBytes
          ?? gitExecutable.expectation.fields.fileBytes,
      executableMachOUUID:
        toolExecutableMachOUUID
          ?? gitExecutable.expectation.fields.machOUUID,
      executableCodeDirectorySHA256:
        gitPrimaryCodeDirectorySHA256,
      runtimeClosureManifestSHA256:
        gitClosure.expectation.documentSHA256,
      runtimeClosureManifestBytes:
        gitClosure.expectation.documentBytes,
      runtimePolicySHA256: denial.policySHA256,
      limits: GitToolPolicyVerifier.phase1ResourceCeilings
    )
    let toolFile = admittedFile(
      try GitToolPolicyVerifier.policyBytes(fields: toolFields)
    )
    let toolAnchor = GitToolPolicyTrustAnchor(
      expectedCurrentPolicySHA256: toolFile.sha256,
      expectedCurrentPolicyBytes: UInt64(toolFile.bytes.count),
      minimumPolicyGeneration: 7,
      verificationUnixSeconds: commonTime
    )
    let tool = try GitToolPolicyVerifier.anchor(
      policyFile: toolFile,
      trustAnchor: toolAnchor
    )
    let claim = try signedClaim(
      toolManifestSHA256: tool.policySHA256,
      toolManifestBytes: tool.policyBytes,
      runtimePolicySHA256: denial.policySHA256,
      usePinnedSignature:
        selfGuardCacheSeed == "1" && selfGuardLoaderSeed == 0x20
        && gitRootUUIDSeed == nil && selfGuardRootUUIDSeed == nil
        && toolExecutableBytes == nil && toolExecutableMachOUUID == nil
        && toolExecutableCodeDirectorySHA256 == nil
        && denialSelfGuardBytes == nil
        && denialSelfGuardMachOUUID == nil
        && denialSelfGuardCodeDirectorySHA256 == nil
        && denialGitClosureID == nil && denialSelfGuardClosureID == nil
        && denialLimits == nil
    )
    let reference = try GitToolPolicyVerifier.reference(
      signedClaim: claim,
      policyDocument: tool
    )

    return Fixture(
      reference: reference,
      toolAnchor: toolAnchor,
      tool: tool,
      denialAnchor: denialAnchor,
      denial: denial,
      gitExecutableAnchor: gitExecutableAnchor,
      gitExecutable: gitExecutable,
      selfGuardExecutableAnchor: selfGuardExecutableAnchor,
      selfGuardExecutable: selfGuardExecutable,
      gitClosure: gitClosure,
      selfGuardClosure: selfGuardClosure
    )
  }

  static func compare(
    _ fixture: Fixture,
    allTimes: UInt64? = nil,
    gitToolAnchor: GitToolPolicyTrustAnchor? = nil,
    runtimeDenialAnchor: GitRuntimePolicyDenialTrustAnchor? = nil,
    gitExecutableAnchor: ExecutableIdentityExpectationTrustAnchor? = nil,
    selfGuardExecutableAnchor:
      ExecutableIdentityExpectationTrustAnchor? = nil,
    gitClosureAnchor: RuntimeClosureExpectationTrustAnchor? = nil,
    selfGuardClosureAnchor: RuntimeClosureExpectationTrustAnchor? = nil
  ) throws -> ExecutionIdentityComparisonEvidence {
    let time = allTimes
    let resolvedToolAnchor = gitToolAnchor ?? time.map {
      GitToolPolicyTrustAnchor(
        expectedCurrentPolicySHA256: fixture.toolAnchor
          .expectedCurrentPolicySHA256,
        expectedCurrentPolicyBytes: fixture.toolAnchor
          .expectedCurrentPolicyBytes,
        minimumPolicyGeneration: fixture.toolAnchor.minimumPolicyGeneration,
        verificationUnixSeconds: $0
      )
    } ?? fixture.toolAnchor
    let resolvedDenialAnchor = runtimeDenialAnchor ?? time.map {
      GitRuntimePolicyDenialTrustAnchor(
        expectedCurrentPolicySHA256: fixture.denialAnchor
          .expectedCurrentPolicySHA256,
        expectedCurrentPolicyBytes: fixture.denialAnchor
          .expectedCurrentPolicyBytes,
        minimumPolicyGeneration: fixture.denialAnchor
          .minimumPolicyGeneration,
        verificationUnixSeconds: $0
      )
    } ?? fixture.denialAnchor
    let resolvedGitExecutableAnchor = gitExecutableAnchor ?? time.map {
      executableAnchor(fixture.gitExecutable.expectation, time: $0)
    } ?? fixture.gitExecutableAnchor
    let resolvedSelfGuardExecutableAnchor = selfGuardExecutableAnchor
      ?? time.map {
        executableAnchor(
          fixture.selfGuardExecutable.expectation,
          time: $0
        )
      } ?? fixture.selfGuardExecutableAnchor
    let resolvedGitClosureAnchor = gitClosureAnchor ?? time.map {
      closureAnchor(fixture.gitClosure.expectation, time: $0)
    } ?? fixture.gitClosure.anchor
    let resolvedSelfGuardClosureAnchor = selfGuardClosureAnchor ?? time.map {
      closureAnchor(fixture.selfGuardClosure.expectation, time: $0)
    } ?? fixture.selfGuardClosure.anchor

    return try ExecutionIdentityComparisonVerifier.compare(
      signedClaimToolPolicyReference: fixture.reference,
      currentGitToolPolicyAnchor: resolvedToolAnchor,
      runtimeDenialDocument: fixture.denial,
      currentRuntimeDenialAnchor: resolvedDenialAnchor,
      gitExecutableComparison: fixture.gitExecutable,
      currentGitExecutableAnchor: resolvedGitExecutableAnchor,
      selfGuardExecutableComparison: fixture.selfGuardExecutable,
      currentSelfGuardExecutableAnchor: resolvedSelfGuardExecutableAnchor,
      gitClosureExpectation: fixture.gitClosure.expectation,
      currentGitClosureAnchor: resolvedGitClosureAnchor,
      gitClosureComparison: fixture.gitClosure.comparison,
      selfGuardClosureExpectation: fixture.selfGuardClosure.expectation,
      currentSelfGuardClosureAnchor: resolvedSelfGuardClosureAnchor,
      selfGuardClosureComparison: fixture.selfGuardClosure.comparison
    )
  }

  static func assertFailure(
    _ fixture: Fixture,
    allTimes: UInt64? = nil,
    gitToolAnchor: GitToolPolicyTrustAnchor? = nil,
    runtimeDenialAnchor: GitRuntimePolicyDenialTrustAnchor? = nil,
    gitExecutableAnchor: ExecutableIdentityExpectationTrustAnchor? = nil,
    selfGuardExecutableAnchor:
      ExecutableIdentityExpectationTrustAnchor? = nil,
    gitClosureAnchor: RuntimeClosureExpectationTrustAnchor? = nil,
    selfGuardClosureAnchor: RuntimeClosureExpectationTrustAnchor? = nil,
    expected: ExecutionIdentityComparisonFailure,
    file: StaticString = #filePath,
    line: UInt = #line
  ) {
    XCTAssertThrowsError(
      try compare(
        fixture,
        allTimes: allTimes,
        gitToolAnchor: gitToolAnchor,
        runtimeDenialAnchor: runtimeDenialAnchor,
        gitExecutableAnchor: gitExecutableAnchor,
        selfGuardExecutableAnchor: selfGuardExecutableAnchor,
        gitClosureAnchor: gitClosureAnchor,
        selfGuardClosureAnchor: selfGuardClosureAnchor
      ),
      file: file,
      line: line
    ) {
      XCTAssertEqual(
        $0 as? ExecutionIdentityComparisonFailure,
        expected,
        file: file,
        line: line
      )
    }
  }

  static func assertExecutableAnchorScalarFailures(
    _ fixture: Fixture,
    original: ExecutableIdentityExpectationTrustAnchor,
    slot: ExecutionIdentityAnchorSlot,
    git: Bool,
    invalidDigest: String,
    file: StaticString = #filePath,
    line: UInt = #line
  ) {
    let cases: [(
      ExecutableIdentityExpectationTrustAnchor,
      ExecutionIdentityAnchorScalarField
    )] = [
      (executableAnchor(original, digest: invalidDigest), .digest),
      (executableAnchor(original, bytes: 0), .byteCount),
      (executableAnchor(original, generation: 0), .minimumGeneration),
    ]
    for (anchor, field) in cases {
      if git {
        assertFailure(
          fixture,
          gitExecutableAnchor: anchor,
          expected: .invalidAnchorScalar(slot: slot, field: field),
          file: file,
          line: line
        )
      } else {
        assertFailure(
          fixture,
          selfGuardExecutableAnchor: anchor,
          expected: .invalidAnchorScalar(slot: slot, field: field),
          file: file,
          line: line
        )
      }
    }
  }

  static func assertClosureAnchorScalarFailures(
    _ fixture: Fixture,
    original: RuntimeClosureExpectationTrustAnchor,
    slot: ExecutionIdentityAnchorSlot,
    git: Bool,
    invalidDigest: String,
    file: StaticString = #filePath,
    line: UInt = #line
  ) {
    let cases: [(
      RuntimeClosureExpectationTrustAnchor,
      ExecutionIdentityAnchorScalarField
    )] = [
      (closureAnchor(original, digest: invalidDigest), .digest),
      (closureAnchor(original, bytes: 0), .byteCount),
      (closureAnchor(original, generation: 0), .minimumGeneration),
    ]
    for (anchor, field) in cases {
      if git {
        assertFailure(
          fixture,
          gitClosureAnchor: anchor,
          expected: .invalidAnchorScalar(slot: slot, field: field),
          file: file,
          line: line
        )
      } else {
        assertFailure(
          fixture,
          selfGuardClosureAnchor: anchor,
          expected: .invalidAnchorScalar(slot: slot, field: field),
          file: file,
          line: line
        )
      }
    }
  }

  static func replacing(
    _ base: Fixture,
    denialWith denialSource: Fixture? = nil,
    gitExecutableWith gitExecutableSource: Fixture? = nil,
    selfGuardExecutableWith selfGuardExecutableSource: Fixture? = nil,
    gitExecutableComparisonWith gitExecutableComparison:
      ExecutableContentExpectationComparison? = nil,
    gitClosureWith gitClosure: ClosureFixture? = nil,
    keepGitClosureAnchor: Bool = false
  ) -> Fixture {
    let denial = denialSource ?? base
    let gitExecutable = gitExecutableSource ?? base
    let selfGuardExecutable = selfGuardExecutableSource ?? base
    let resolvedGitClosure = gitClosure ?? base.gitClosure
    return Fixture(
      reference: base.reference,
      toolAnchor: base.toolAnchor,
      tool: base.tool,
      denialAnchor: denial.denialAnchor,
      denial: denial.denial,
      gitExecutableAnchor: gitExecutable.gitExecutableAnchor,
      gitExecutable:
        gitExecutableComparison ?? gitExecutable.gitExecutable,
      selfGuardExecutableAnchor:
        selfGuardExecutable.selfGuardExecutableAnchor,
      selfGuardExecutable: selfGuardExecutable.selfGuardExecutable,
      gitClosure: ClosureFixture(
        role: resolvedGitClosure.role,
        root: resolvedGitClosure.root,
        loader: resolvedGitClosure.loader,
        cacheSet: resolvedGitClosure.cacheSet,
        members: resolvedGitClosure.members,
        edges: resolvedGitClosure.edges,
        collection: resolvedGitClosure.collection,
        rootInventory: resolvedGitClosure.rootInventory,
        memberInventories: resolvedGitClosure.memberInventories,
        graph: resolvedGitClosure.graph,
        anchor: keepGitClosureAnchor
          ? base.gitClosure.anchor
          : resolvedGitClosure.anchor,
        expectation: resolvedGitClosure.expectation,
        comparison: resolvedGitClosure.comparison
      ),
      selfGuardClosure: base.selfGuardClosure
    )
  }

  static func mixingClosure(
    expectationFrom current: ClosureFixture,
    comparisonFrom changed: ClosureFixture
  ) -> ClosureFixture {
    ClosureFixture(
      role: current.role,
      root: current.root,
      loader: current.loader,
      cacheSet: current.cacheSet,
      members: current.members,
      edges: current.edges,
      collection: current.collection,
      rootInventory: current.rootInventory,
      memberInventories: current.memberInventories,
      graph: current.graph,
      anchor: current.anchor,
      expectation: current.expectation,
      comparison: changed.comparison
    )
  }

  static func phase1Limits(
    maxOpenFiles: UInt64
  ) -> GitToolPolicyResourceLimits {
    let limits = GitToolPolicyVerifier.phase1ResourceCeilings
    return GitToolPolicyResourceLimits(
      maxPackBytes: limits.maxPackBytes,
      maxPackObjects: limits.maxPackObjects,
      maxCommitBytes: limits.maxCommitBytes,
      maxSingleInflatedObjectBytes: limits.maxSingleInflatedObjectBytes,
      maxTotalInflatedBytes: limits.maxTotalInflatedBytes,
      maxCompressionRatio: limits.maxCompressionRatio,
      maxTreeDepth: limits.maxTreeDepth,
      maxTreeCount: limits.maxTreeCount,
      maxObjectDatabaseBytes: limits.maxObjectDatabaseBytes,
      maxStdoutBytes: limits.maxStdoutBytes,
      maxStderrBytes: limits.maxStderrBytes,
      wallTimeoutMilliseconds: limits.wallTimeoutMilliseconds,
      cpuTimeoutSeconds: limits.cpuTimeoutSeconds,
      maxAddressSpaceBytes: limits.maxAddressSpaceBytes,
      maxFileSizeBytes: limits.maxFileSizeBytes,
      maxOpenFiles: maxOpenFiles
    )
  }

  static func closure(
    role: RuntimeClosureExpectationArtifactRole,
    cacheSet: SyntheticSharedCacheSetIdentityEvidence,
    loader: DynamicLoaderContentIdentityEvidence,
    rootUUIDSeed: UInt8?
  ) throws -> ClosureFixture {
    let name = installName(installNameValue)
    let image = try SyntheticSharedCacheImageContentIdentityVerifier.derive(
      cacheSetEvidence: cacheSet,
      facts: SyntheticSharedCacheImageContentFacts(
        installNameBytes: name.bytes,
        installNameBase64URL: name.base64URL,
        machOUUID: fixedHex(seed: 0x30, count: 16),
        primaryCodeDirectory: .absent,
        loadCommandsSHA256: sha256Hex(Data())
      )
    )
    let snapshot =
      try SyntheticSharedCacheImageLoadCommandSnapshotVerifier.derive(
        imageEvidence: image,
        loadCommandBytes: Data()
      )
    let member = try SyntheticRuntimeClosureRecordSchemaVerifier.member(
      index: 0,
      source: .sharedCache(image),
      installName: name
    )
    let memberInventory =
      try SyntheticAcceptedDependencyCommandInventoryVerifier
      .sharedCacheMember(snapshot)
    let root = try rootEvidence(
      role: role == .git ? .git : .selfGuard,
      loadCommands: [dylibCommand(name: installNameValue)],
      uuidSeed: rootUUIDSeed
    )
    let rootInventory =
      try SyntheticAcceptedDependencyCommandInventoryVerifier.root(root)
    let rootEntry = try XCTUnwrap(rootInventory.entries.first)
    let edge = try SyntheticRuntimeClosureRecordSchemaVerifier.edge(
      index: 0,
      parent: .root(root),
      loadCommandOrdinal: rootEntry.loadCommandOrdinal,
      kind: rootEntry.kind,
      installName: member.installName,
      resolved: member
    )
    let members = [member]
    let edges = [edge]
    let collection =
      try SyntheticRuntimeClosureRecordCollectionVerifier.derive(
        members: members,
        edges: edges
      )
    let graph = try SyntheticRuntimeClosureGraphVerifier.compare(
      root: root,
      members: members,
      edges: edges,
      collection: collection,
      rootInventory: rootInventory,
      memberInventories: [memberInventory]
    )
    let expectationFile = admittedFile(
      renderClosureExpectation(
        role: role,
        rootID: root.contentEvidenceID.sha256,
        loaderID: loader.contentEvidenceID.sha256,
        cacheSet: cacheSet,
        members: members,
        edges: edges
      )
    )
    let anchor = RuntimeClosureExpectationTrustAnchor(
      expectedCurrentDocumentSHA256: expectationFile.sha256,
      expectedCurrentDocumentBytes: UInt64(expectationFile.bytes.count),
      minimumEvidenceGeneration: 9,
      verificationUnixSeconds: commonTime,
      expectedArtifactRole: role
    )
    let expectation = try RuntimeClosureExpectationVerifier.anchor(
      expectationFile: expectationFile,
      trustAnchor: anchor
    )
    let comparison = try RuntimeClosureContentIdentityVerifier.compare(
      anchoredExpectation: expectation,
      currentExpectationAnchor: anchor,
      rootExecutableContentEvidence: root,
      dynamicLoaderContentEvidence: loader,
      members: members,
      edges: edges,
      collection: collection,
      rootInventory: rootInventory,
      memberInventories: [memberInventory],
      graphComparison: graph
    )
    return ClosureFixture(
      role: role,
      root: root,
      loader: loader,
      cacheSet: cacheSet,
      members: members,
      edges: edges,
      collection: collection,
      rootInventory: rootInventory,
      memberInventories: [memberInventory],
      graph: graph,
      anchor: anchor,
      expectation: expectation,
      comparison: comparison
    )
  }

  static func executableComparison(
    evidence: ExecutableContentIdentityEvidence,
    role: ExecutableIdentityArtifactRole
  ) throws -> (
    ExecutableContentExpectationComparison,
    ExecutableIdentityExpectationTrustAnchor
  ) {
    let fields = executableFields(
      evidence.comparison,
      role: role
    )
    let file = admittedFile(
      try ExecutableIdentityExpectationVerifier.documentBytes(
        fields: fields
      )
    )
    let anchor = ExecutableIdentityExpectationTrustAnchor(
      expectedCurrentDocumentSHA256: file.sha256,
      expectedCurrentDocumentBytes: UInt64(file.bytes.count),
      minimumEvidenceGeneration: 9,
      verificationUnixSeconds: commonTime,
      expectedArtifactRole: role
    )
    let expectation = try ExecutableIdentityExpectationVerifier.anchor(
      expectationFile: file,
      trustAnchor: anchor
    )
    let comparison = try ExecutableContentIdentityVerifier.match(
      evidence: evidence,
      expectation: expectation
    )
    return (comparison, anchor)
  }

  static func executableFields(
    _ comparison: SyntheticMachOIdentityComparison,
    role: ExecutableIdentityArtifactRole
  ) -> ExecutableIdentityExpectationFields {
    ExecutableIdentityExpectationFields(
      evidenceGeneration: 9,
      validFromUnixSeconds: 1_900_000_000,
      validUntilUnixSeconds: 2_100_000_000,
      artifactRole: role,
      fileSHA256: comparison.fileSHA256,
      fileBytes: UInt64(comparison.retainedFileBytes.count),
      cpuSubtype: hex8(comparison.cpuSubtype),
      headerFlags: hex8(comparison.headerFlags),
      loadCommandCount: UInt64(comparison.loadCommandCount),
      loadCommandBytes: UInt64(comparison.loadCommandBytes.count),
      loadCommandsSHA256: comparison.loadCommandsSHA256,
      machOUUID: hex(comparison.machOUUID),
      codeSignatureRegionSHA256:
        comparison.codeSignatureRegionSHA256,
      codeSignatureRegionBytes:
        UInt64(comparison.codeSignatureRegion.count),
      codeDirectories: comparison.codeDirectories.map {
        ExecutableIdentityCodeDirectoryExpectation(
          slot: UInt64($0.slot),
          blobSHA256: $0.blobSHA256,
          blobBytes: UInt64($0.blob.count),
          hashType: UInt64($0.hashType),
          flags: hex8($0.flags),
          signingIdentifierBytes: UInt64($0.signingIdentifier.count),
          signingIdentifierBase64URL: base64URL($0.signingIdentifier),
          teamIdentifierBytes: UInt64($0.teamIdentifier.count),
          teamIdentifierBase64URL: base64URL($0.teamIdentifier)
        )
      },
      cmsBlobSHA256: comparison.cmsBlobSHA256,
      cmsBlobBytes: UInt64(comparison.cmsBlob?.count ?? 0)
    )
  }

  static func signedClaim(
    toolManifestSHA256: String,
    toolManifestBytes: UInt64,
    runtimePolicySHA256: String,
    usePinnedSignature: Bool
  ) throws -> OperatorSignedRunClaim {
    let rootKey = try testKey(seed: 0x10)
    let runKey = try testKey(seed: 0x30)
    let authorizationKey = try testKey(seed: 0x50)
    let worker = try authorizedFile(
      Data("worker-stage-c3\n".utf8),
      purpose: .workerBytes,
      key: authorizationKey,
      // gitleaks:allow -- deterministic synthetic unit-test signature only.
      signatureBase64:
        "snhEEEl8XOOpC2BefJL9gVh96AIg4GjO2wbBt3z+p3Bup0+phwtCBVT4jmiSVyaA"
        + "c6qP0GRflGLI9wRXI/ZtDw=="
    )
    let baseline = try authorizedFile(
      Data("baseline-stage-c3\n".utf8),
      purpose: .sourceManifest,
      key: authorizationKey,
      // gitleaks:allow -- deterministic synthetic unit-test signature only.
      signatureBase64:
        "kQHJT43I7e874p/2HEDUoi+Yq6VKxlOdYtXxp/boMtALfMM5eQZxclwOZLCZ1EV7"
        + "3Y81jR27fBa9Kdek23ahAA=="
    )
    let candidate = try authorizedFile(
      Data("candidate-stage-c3\n".utf8),
      purpose: .sourceManifest,
      key: authorizationKey,
      // gitleaks:allow -- deterministic synthetic unit-test signature only.
      signatureBase64:
        "tuX5DenUQTWOXIXZIfb+p1ti6T04ERNr/2xRN2aRHM3YrWzUhXBNfLQBhlvKXy11"
        + "dSLvMOBPk0YbMt12ZO/9Cg=="
    )
    let inputs = try RunAuthorizedInputs(
      worker: worker,
      baselineSourceManifest: baseline,
      candidateSourceManifest: candidate
    )

    let keyFields = OperatorKeyPolicyFields(
      rootKeyID: rootKey.keyID,
      policyGeneration: 7,
      validFromUnixSeconds: 1_900_000_000,
      validUntilUnixSeconds: 2_100_000_000,
      activeOperatorKeyID: runKey.keyID,
      activeOperatorPublicKeyBase64: runKey.publicKeyBase64,
      activeOperatorScope: .runClaim,
      allowedClaimSubject: .absorbedMLALoadedResultPair,
      revokedOperatorKeyIDs: []
    )
    let keyPolicyFile = admittedFile(
      try OperatorKeyPolicyVerifier.policyBytes(fields: keyFields)
    )
    // gitleaks:allow -- deterministic synthetic unit-test signature only.
    let keyPolicySignature =
      "PTOXcmv0GTOCFYoUJ1ll7b+1nNbG8f/8/oZZ+wpsdNJ1Mil7nWrRzim+bnK4ws40"
      + "m5cL0C4i74ceLsCeeZ4JBA=="
    let keyPolicy = try OperatorKeyPolicyVerifier.admit(
      policyFile: keyPolicyFile,
      rootSignatureBase64: keyPolicySignature,
      trustAnchor: OperatorKeyPolicyTrustAnchor(
        rootPublicKeyBase64: rootKey.publicKeyBase64,
        rootKeyID: rootKey.keyID,
        expectedCurrentPolicySHA256: keyPolicyFile.sha256,
        minimumPolicyGeneration: 7,
        verificationUnixSeconds: commonTime
      )
    )
    let runner = admittedFile(Data("runner-stage-c3\n".utf8))
    let expectations = OperatorRunClaimAdmissionExpectations(
      keyPolicy: keyPolicy,
      hostAdmissionID: hexByte(0x04),
      runner: runner,
      resultPairID: hexByte(0x05),
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
      worker: authorizedReference(worker),
      policies: OperatorRunClaimPolicyReferences(
        sourceSHA256: hexByte(0x10),
        dependencySHA256: hexByte(0x11),
        buildSHA256: hexByte(0x12),
        runtimeSHA256: runtimePolicySHA256,
        preflightSHA256: hexByte(0x14),
        publicationSHA256: hexByte(0x15)
      ),
      toolManifest: OperatorRunClaimByteIdentity(
        sha256: toolManifestSHA256,
        byteCount: toolManifestBytes
      ),
      baseline: OperatorRunClaimSourceReference(
        role: .baseline,
        sourceManifest: authorizedReference(baseline),
        gitCommitSHA1: String(repeating: "a", count: 40),
        gitTreeSHA1: String(repeating: "b", count: 40),
        route: .decompressedDeepSeekV3,
        slot: .baseline,
        buildReceiptID: hexByte(0x17),
        binary: OperatorRunClaimByteIdentity(
          sha256: hexByte(0x18),
          byteCount: 1_001
        )
      ),
      candidate: OperatorRunClaimSourceReference(
        role: .candidate,
        sourceManifest: authorizedReference(candidate),
        gitCommitSHA1: String(repeating: "c", count: 40),
        gitTreeSHA1: String(repeating: "d", count: 40),
        route: .absorbedMLADeepSeekV3Explicit,
        slot: .candidate,
        buildReceiptID: hexByte(0x19),
        binary: OperatorRunClaimByteIdentity(
          sha256: hexByte(0x20),
          byteCount: 1_002
        )
      ),
      model: OperatorRunClaimAuthorizedPayloadReference(
        authorizationID: hexByte(0x21),
        payload: OperatorRunClaimByteIdentity(
          sha256: hexByte(0x22),
          byteCount: 55
        )
      ),
      tokenizer: OperatorRunClaimAuthorizedPayloadReference(
        authorizationID: hexByte(0x23),
        payload: OperatorRunClaimByteIdentity(
          sha256: hexByte(0x24),
          byteCount: 66
        )
      ),
      workload: OperatorRunClaimAuthorizedPayloadReference(
        authorizationID: hexByte(0x25),
        payload: OperatorRunClaimByteIdentity(
          sha256: hexByte(0x26),
          byteCount: 77
        )
      ),
      resultPairID: expectations.resultPairID
    )
    let claimBytes = try OperatorSignedRunClaimVerifier.claimBytes(
      fields: fields
    )
    let signature: String
    if usePinnedSignature {
      // gitleaks:allow -- deterministic synthetic unit-test signature only.
      signature =
        "XNUPQG2Ww1oEL0b1j0x2ugWjyEp0TluGp315WGT1o/SIK202Hq8w/lwepz3XljnI"
        + "kHUmiz5ONM/iH4kLGFY0Cg=="
    } else {
      signature = try runKey.privateKey.signature(
        for: claimBytes
      ).base64EncodedString()
    }
    return try OperatorSignedRunClaimVerifier.verify(
      claimBytes: claimBytes,
      signatureBase64: signature,
      expectations: expectations
    )
  }

  static func authorizedFile(
    _ bytes: Data,
    purpose: OperatorAuthorizationPurpose,
    key: TestKey,
    signatureBase64: String
  ) throws -> OperatorAuthorizedFile {
    let file = admittedFile(bytes)
    let claim = try OperatorAuthorization.claimBytes(
      purpose: purpose,
      payloadSHA256: file.sha256,
      payloadByteCount: UInt64(file.bytes.count)
    )
    return try OperatorAuthorization.verify(
      admittedFile: file,
      expectedPurpose: purpose,
      claimBytes: claim,
      signatureBase64: signatureBase64,
      publicKeyBase64: key.publicKeyBase64,
      allowedKeyID: key.keyID
    )
  }

  static func authorizedReference(
    _ file: OperatorAuthorizedFile
  ) -> OperatorRunClaimAuthorizedPayloadReference {
    OperatorRunClaimAuthorizedPayloadReference(
      authorizationID: file.authorizationID.rawValue,
      payload: OperatorRunClaimByteIdentity(
        sha256: file.file.sha256,
        byteCount: UInt64(file.file.bytes.count)
      )
    )
  }

  static func testKey(seed: UInt8) throws -> TestKey {
    // gitleaks:allow -- deterministic synthetic unit-test material only.
    let key = try Curve25519.Signing.PrivateKey(
      rawRepresentation: Data((0..<32).map { seed &+ UInt8($0) })
    )
    let publicBytes = key.publicKey.rawRepresentation
    return TestKey(
      privateKey: key,
      publicKeyBase64: publicBytes.base64EncodedString(),
      keyID: sha256Hex(publicBytes)
    )
  }

  static func cacheSet(
    seed: Character
  ) throws -> SyntheticSharedCacheSetIdentityEvidence {
    try SyntheticSharedCacheSetIdentityVerifier.derive(
      records: [
        SyntheticSharedCacheFileRecord(
          suffixBytes: 0,
          suffixBase64URL: "",
          fileSHA256: String(repeating: seed, count: 64),
          fileBytes: 4_096,
          headerUUID: String(repeating: seed, count: 32)
        )
      ]
    )
  }

  static func rootEvidence(
    role: ExecutableContentArtifactRole,
    loadCommands: [Data],
    uuidSeed: UInt8? = nil
  ) throws -> ExecutableContentIdentityEvidence {
    let signature = superBlob(
      entries: [
        (
          0,
          codeDirectory(
            signingIdentifier:
              Data("com.example.c3.\(role.rawValue)".utf8)
          )
        )
      ]
    )
    let comparison = try SyntheticMachOIdentityParser.parse(
      executeMachO(
        loadCommands: loadCommands,
        signatureRegion: signature,
        uuidSeed: uuidSeed ?? (role == .git ? 0x10 : 0x40)
      )
    )
    return try ExecutableContentIdentityVerifier.derive(
      artifactRole: role,
      comparison: comparison
    )
  }

  static func dynamicLoaderEvidence(
    seed: UInt8
  ) throws -> DynamicLoaderContentIdentityEvidence {
    let signature = superBlob(
      entries: [
        (
          0,
          codeDirectory(
            signingIdentifier: Data("com.example.c3.dyld".utf8)
          )
        )
      ]
    )
    let comparison = try SyntheticDynamicLoaderMachOIdentityParser.parse(
      dynamicLoaderMachO(
        signatureRegion: signature,
        uuidSeed: seed
      )
    )
    return try DynamicLoaderContentIdentityVerifier.derive(
      comparison: comparison
    )
  }

  static func executeMachO(
    loadCommands extraCommands: [Data],
    signatureRegion: Data,
    uuidSeed: UInt8
  ) -> Data {
    var commands = uuidCommand(seed: uuidSeed)
    for command in extraCommands { commands.append(command) }
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
      fileType: 2,
      commandCount: extraCommands.count + 2,
      commands: commands,
      signatureRegion: signatureRegion
    )
  }

  static func dynamicLoaderMachO(
    signatureRegion: Data,
    uuidSeed: UInt8
  ) -> Data {
    var commands = uuidCommand(seed: uuidSeed)
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
    while !result.count.isMultiple(of: 8) { result.append(0) }
    writeUInt32LE(&result, at: 4, value: UInt32(result.count))
    return result
  }

  static func dylibCommand(name: String) -> Data {
    var result = Data()
    appendUInt32LE(&result, 0x0c)
    appendUInt32LE(&result, 0)
    appendUInt32LE(&result, 24)
    appendUInt32LE(&result, 0)
    appendUInt32LE(&result, 0)
    appendUInt32LE(&result, 0)
    result.append(Data(name.utf8))
    result.append(0)
    while !result.count.isMultiple(of: 8) { result.append(0) }
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
    for (_, blob) in entries { result.append(blob) }
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

  static func renderClosureExpectation(
    role: RuntimeClosureExpectationArtifactRole,
    rootID: String,
    loaderID: String,
    cacheSet: SyntheticSharedCacheSetIdentityEvidence,
    members: [SyntheticRuntimeClosureMemberRecordComparison],
    edges: [SyntheticRuntimeClosureEdgeRecordComparison]
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
      lines.append("\(prefix)suffix_base64url=\(record.suffixBase64URL)")
      lines.append("\(prefix)sha256=\(record.fileSHA256)")
      lines.append("\(prefix)bytes=\(record.fileBytes)")
      lines.append("\(prefix)header_uuid=\(record.headerUUID)")
    }
    lines.append("shared_cache_set_id=\(cacheSet.sharedCacheSetID.sha256)")
    lines.append("member_count=\(members.count)")
    for (index, member) in members.enumerated() {
      let prefix = "member_\(fourDigit(index))_"
      lines.append("\(prefix)content_evidence_id=\(member.contentEvidenceID)")
      lines.append("\(prefix)storage=\(member.storage.rawValue)")
      lines.append("\(prefix)install_name_bytes=\(member.installName.bytes)")
      lines.append("\(prefix)install_name_base64url=\(member.installName.base64URL)")
      lines.append("\(prefix)macho_uuid=\(member.machOUUID)")
      lines.append(
        "\(prefix)primary_code_directory_blob_sha256="
          + member.primaryCodeDirectoryBlobSHA256
      )
      lines.append("\(prefix)load_commands_sha256=\(member.loadCommandsSHA256)")
    }
    lines.append("edge_count=\(edges.count)")
    for (index, edge) in edges.enumerated() {
      let prefix = "edge_\(fourDigit(index))_"
      lines.append(
        "\(prefix)parent_content_evidence_id="
          + edge.parentContentEvidenceID
      )
      lines.append("\(prefix)load_command_ordinal=\(edge.loadCommandOrdinal)")
      lines.append("\(prefix)kind=\(edge.kind.rawValue)")
      lines.append("\(prefix)install_name_bytes=\(edge.installName.bytes)")
      lines.append("\(prefix)install_name_base64url=\(edge.installName.base64URL)")
      lines.append(
        "\(prefix)resolved_content_evidence_id="
          + edge.resolvedContentEvidenceID
      )
    }
    lines.append("runtime_resolution_outcome=unproved-static-comparison-only")
    lines.append("runtime_authority=none")
    return Data((lines.joined(separator: "\n") + "\n").utf8)
  }

  static func contextPreimage(_ fixture: Fixture) -> Data {
    let anchors = (
      fixture.toolAnchor,
      fixture.denialAnchor,
      fixture.gitExecutableAnchor,
      fixture.selfGuardExecutableAnchor,
      fixture.gitClosure.anchor,
      fixture.selfGuardClosure.anchor
    )
    return Data(
      ([
        "fast-mlx-proof-control-execution-identity-anchor-context-id-v1",
        "git_tool_policy_expected_sha256="
          + anchors.0.expectedCurrentPolicySHA256,
        "git_tool_policy_expected_bytes=\(anchors.0.expectedCurrentPolicyBytes)",
        "git_tool_policy_minimum_generation=\(anchors.0.minimumPolicyGeneration)",
        "git_tool_policy_verification_unix_seconds=\(anchors.0.verificationUnixSeconds)",
        "runtime_denial_expected_sha256="
          + anchors.1.expectedCurrentPolicySHA256,
        "runtime_denial_expected_bytes=\(anchors.1.expectedCurrentPolicyBytes)",
        "runtime_denial_minimum_generation=\(anchors.1.minimumPolicyGeneration)",
        "runtime_denial_verification_unix_seconds=\(anchors.1.verificationUnixSeconds)",
        "git_executable_expectation_sha256="
          + anchors.2.expectedCurrentDocumentSHA256,
        "git_executable_expectation_bytes=\(anchors.2.expectedCurrentDocumentBytes)",
        "git_executable_minimum_generation=\(anchors.2.minimumEvidenceGeneration)",
        "git_executable_verification_unix_seconds=\(anchors.2.verificationUnixSeconds)",
        "self_guard_executable_expectation_sha256="
          + anchors.3.expectedCurrentDocumentSHA256,
        "self_guard_executable_expectation_bytes=\(anchors.3.expectedCurrentDocumentBytes)",
        "self_guard_executable_minimum_generation=\(anchors.3.minimumEvidenceGeneration)",
        "self_guard_executable_verification_unix_seconds=\(anchors.3.verificationUnixSeconds)",
        "git_closure_expectation_sha256="
          + anchors.4.expectedCurrentDocumentSHA256,
        "git_closure_expectation_bytes=\(anchors.4.expectedCurrentDocumentBytes)",
        "git_closure_minimum_generation=\(anchors.4.minimumEvidenceGeneration)",
        "git_closure_verification_unix_seconds=\(anchors.4.verificationUnixSeconds)",
        "self_guard_closure_expectation_sha256="
          + anchors.5.expectedCurrentDocumentSHA256,
        "self_guard_closure_expectation_bytes=\(anchors.5.expectedCurrentDocumentBytes)",
        "self_guard_closure_minimum_generation=\(anchors.5.minimumEvidenceGeneration)",
        "self_guard_closure_verification_unix_seconds=\(anchors.5.verificationUnixSeconds)",
      ].joined(separator: "\n") + "\n").utf8
    )
  }

  static func finalPreimage(
    _ fixture: Fixture,
    contextID: String
  ) -> Data {
    Data(
      ([
        "fast-mlx-proof-control-execution-identity-comparison-id-v1",
        "run_claim_id=\(fixture.reference.signedClaim.claimID.rawValue)",
        "anchor_context_id=\(contextID)",
        "git_tool_policy_sha256=\(fixture.tool.policySHA256)",
        "runtime_denial_policy_sha256=\(fixture.denial.policySHA256)",
        "git_executable_expectation_sha256="
          + fixture.gitExecutable.expectation.documentSHA256,
        "git_executable_content_evidence_id="
          + fixture.gitExecutable.contentEvidence.contentEvidenceID.sha256,
        "self_guard_executable_expectation_sha256="
          + fixture.selfGuardExecutable.expectation.documentSHA256,
        "self_guard_executable_content_evidence_id="
          + fixture.selfGuardExecutable.contentEvidence.contentEvidenceID.sha256,
        "git_closure_manifest_sha256="
          + fixture.gitClosure.expectation.documentSHA256,
        "git_closure_content_evidence_id="
          + fixture.gitClosure.comparison.contentEvidenceID.sha256,
        "self_guard_closure_manifest_sha256="
          + fixture.selfGuardClosure.expectation.documentSHA256,
        "self_guard_closure_content_evidence_id="
          + fixture.selfGuardClosure.comparison.contentEvidenceID.sha256,
        "runtime_decision=no-go",
      ].joined(separator: "\n") + "\n").utf8
    )
  }

  static func executableAnchor(
    _ expectation: AnchoredExecutableIdentityExpectationDocument,
    time: UInt64
  ) -> ExecutableIdentityExpectationTrustAnchor {
    ExecutableIdentityExpectationTrustAnchor(
      expectedCurrentDocumentSHA256: expectation.documentSHA256,
      expectedCurrentDocumentBytes: expectation.documentBytes,
      minimumEvidenceGeneration: 9,
      verificationUnixSeconds: time,
      expectedArtifactRole: expectation.fields.artifactRole
    )
  }

  static func closureAnchor(
    _ expectation: AnchoredRuntimeClosureExpectationDocument,
    time: UInt64
  ) -> RuntimeClosureExpectationTrustAnchor {
    RuntimeClosureExpectationTrustAnchor(
      expectedCurrentDocumentSHA256: expectation.documentSHA256,
      expectedCurrentDocumentBytes: expectation.documentBytes,
      minimumEvidenceGeneration: 9,
      verificationUnixSeconds: time,
      expectedArtifactRole: expectation.fields.artifactRole
    )
  }

  static func toolAnchor(
    _ fixture: Fixture,
    digest: String? = nil,
    bytes: UInt64? = nil,
    generation: UInt64? = nil,
    time: UInt64? = nil
  ) -> GitToolPolicyTrustAnchor {
    GitToolPolicyTrustAnchor(
      expectedCurrentPolicySHA256:
        digest ?? fixture.toolAnchor.expectedCurrentPolicySHA256,
      expectedCurrentPolicyBytes:
        bytes ?? fixture.toolAnchor.expectedCurrentPolicyBytes,
      minimumPolicyGeneration:
        generation ?? fixture.toolAnchor.minimumPolicyGeneration,
      verificationUnixSeconds:
        time ?? fixture.toolAnchor.verificationUnixSeconds
    )
  }

  static func denialAnchor(
    _ fixture: Fixture,
    digest: String? = nil,
    bytes: UInt64? = nil,
    generation: UInt64? = nil,
    time: UInt64? = nil
  ) -> GitRuntimePolicyDenialTrustAnchor {
    GitRuntimePolicyDenialTrustAnchor(
      expectedCurrentPolicySHA256:
        digest ?? fixture.denialAnchor.expectedCurrentPolicySHA256,
      expectedCurrentPolicyBytes:
        bytes ?? fixture.denialAnchor.expectedCurrentPolicyBytes,
      minimumPolicyGeneration:
        generation ?? fixture.denialAnchor.minimumPolicyGeneration,
      verificationUnixSeconds:
        time ?? fixture.denialAnchor.verificationUnixSeconds
    )
  }

  static func executableAnchor(
    _ original: ExecutableIdentityExpectationTrustAnchor,
    digest: String? = nil,
    bytes: UInt64? = nil,
    generation: UInt64? = nil,
    time: UInt64? = nil,
    role: ExecutableIdentityArtifactRole? = nil
  ) -> ExecutableIdentityExpectationTrustAnchor {
    ExecutableIdentityExpectationTrustAnchor(
      expectedCurrentDocumentSHA256:
        digest ?? original.expectedCurrentDocumentSHA256,
      expectedCurrentDocumentBytes:
        bytes ?? original.expectedCurrentDocumentBytes,
      minimumEvidenceGeneration:
        generation ?? original.minimumEvidenceGeneration,
      verificationUnixSeconds:
        time ?? original.verificationUnixSeconds,
      expectedArtifactRole: role ?? original.expectedArtifactRole
    )
  }

  static func closureAnchor(
    _ original: RuntimeClosureExpectationTrustAnchor,
    digest: String? = nil,
    bytes: UInt64? = nil,
    generation: UInt64? = nil,
    time: UInt64? = nil,
    role: RuntimeClosureExpectationArtifactRole? = nil
  ) -> RuntimeClosureExpectationTrustAnchor {
    RuntimeClosureExpectationTrustAnchor(
      expectedCurrentDocumentSHA256:
        digest ?? original.expectedCurrentDocumentSHA256,
      expectedCurrentDocumentBytes:
        bytes ?? original.expectedCurrentDocumentBytes,
      minimumEvidenceGeneration:
        generation ?? original.minimumEvidenceGeneration,
      verificationUnixSeconds:
        time ?? original.verificationUnixSeconds,
      expectedArtifactRole: role ?? original.expectedArtifactRole
    )
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

  static func installName(
    _ value: String
  ) -> SyntheticRuntimeClosureInstallName {
    let bytes = Data(value.utf8)
    return SyntheticRuntimeClosureInstallName(
      bytes: UInt64(bytes.count),
      base64URL: base64URL(bytes)
    )
  }

  static func fixedHex(seed: UInt8, count: Int) -> String {
    (0..<count).map {
      String(format: "%02x", seed &+ UInt8($0))
    }.joined()
  }

  static func hex(_ bytes: Data) -> String {
    bytes.map { String(format: "%02x", $0) }.joined()
  }

  static func hex8(_ value: UInt32) -> String {
    String(format: "%08x", value)
  }

  static func hexByte(_ value: UInt8) -> String {
    String(repeating: String(format: "%02x", value), count: 32)
  }

  static func fourDigit(_ value: Int) -> String {
    String(format: "%04d", value)
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

  static func appendUInt32LE(_ data: inout Data, _ value: UInt32) {
    data.append(UInt8(truncatingIfNeeded: value))
    data.append(UInt8(truncatingIfNeeded: value >> 8))
    data.append(UInt8(truncatingIfNeeded: value >> 16))
    data.append(UInt8(truncatingIfNeeded: value >> 24))
  }

  static func appendUInt32BE(_ data: inout Data, _ value: UInt32) {
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
