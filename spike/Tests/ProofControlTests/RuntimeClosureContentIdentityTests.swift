import CryptoKit
import Foundation
import XCTest

@testable import ProofControl

final class RuntimeClosureContentIdentityTests: XCTestCase {
  func testCanonicalGitAndSelfGuardContentIdentityIsCompactAndInert()
    throws
  {
    let git = try Self.fixture(role: .git)
    let selfGuard = try Self.fixture(role: .selfGuard)

    let gitResult = try Self.compare(git)
    let repeatedGitResult = try Self.compare(git)
    let selfGuardResult = try Self.compare(selfGuard)

    XCTAssertEqual(gitResult, repeatedGitResult)
    XCTAssertNotEqual(
      gitResult.contentEvidenceID,
      selfGuardResult.contentEvidenceID
    )
    try Self.assertResult(gitResult, fixture: git)
    try Self.assertResult(selfGuardResult, fixture: selfGuard)
    XCTAssertEqual(Self.effects, .zero)
  }

  func testGraphSectionComparisonUsesTheRetainedDataIndexSpace() throws {
    let fixture = try Self.fixture(role: .git)
    var backing = Data([0x00])
    backing.append(fixture.expectationFile.bytes)
    backing.append(0x00)
    let slice = backing[1..<(1 + fixture.expectationFile.bytes.count)]
    XCTAssertEqual(slice.startIndex, 1)

    let slicedFile = Self.admittedFile(slice)
    let slicedAnchor = Self.anchor(for: slicedFile, role: .git)
    let slicedExpectation = try RuntimeClosureExpectationVerifier.anchor(
      expectationFile: slicedFile,
      trustAnchor: slicedAnchor
    )
    let result = try RuntimeClosureContentIdentityVerifier.compare(
      anchoredExpectation: slicedExpectation,
      currentExpectationAnchor: slicedAnchor,
      rootExecutableContentEvidence: fixture.root,
      dynamicLoaderContentEvidence: fixture.loader,
      members: fixture.members,
      edges: fixture.edges,
      collection: fixture.collection,
      rootInventory: fixture.rootInventory,
      memberInventories: fixture.memberInventories,
      graphComparison: fixture.graph
    )

    XCTAssertEqual(result, try Self.compare(fixture))
    XCTAssertEqual(Self.effects, .zero)
  }

  func testCurrentAnchorAndRoleFailuresPreserveFirstTypedRefusal()
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
      expected: .expectationReanchor(.documentDigestMismatch)
    )

    let badByteCountAnchor = Self.anchor(
      for: git.expectationFile,
      role: .git,
      bytes: UInt64(git.expectationFile.bytes.count + 1)
    )
    Self.assertFailure(
      fixture: git,
      currentAnchor: badByteCountAnchor,
      expected: .expectationReanchor(
        .documentByteCountMismatch(
          expected: UInt64(git.expectationFile.bytes.count + 1),
          actual: UInt64(git.expectationFile.bytes.count)
        )
      )
    )

    let rollbackAnchor = Self.anchor(
      for: git.expectationFile,
      role: .git,
      minimumGeneration: 10
    )
    Self.assertFailure(
      fixture: git,
      currentAnchor: rollbackAnchor,
      expected: .expectationReanchor(
        .generationRollback(minimum: 10, actual: 9)
      )
    )

    let notYetValidAnchor = Self.anchor(
      for: git.expectationFile,
      role: .git,
      verificationUnixSeconds: 1_899_999_999
    )
    Self.assertFailure(
      fixture: git,
      currentAnchor: notYetValidAnchor,
      expected: .expectationReanchor(.documentNotYetValid)
    )

    let expiredAnchor = Self.anchor(
      for: git.expectationFile,
      role: .git,
      verificationUnixSeconds: 2_100_000_001
    )
    Self.assertFailure(
      fixture: git,
      currentAnchor: expiredAnchor,
      expected: .expectationReanchor(.documentExpired)
    )

    let foreignRoleAnchor = Self.anchor(
      for: git.expectationFile,
      role: .selfGuard
    )
    Self.assertFailure(
      fixture: git,
      currentAnchor: foreignRoleAnchor,
      expected: .expectationReanchor(
        .artifactRoleMismatch(expected: .selfGuard, actual: .git)
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
      expected: .expectationEvidenceMismatch
    )

    Self.assertFailure(
      fixture: git,
      root: selfGuard.root,
      expected: .rootRole(expected: .git, actual: .selfGuard)
    )
    XCTAssertEqual(Self.effects, .zero)
  }

  func testManifestAndGraphJoinsRejectDriftBeforeIdentity() throws {
    let git = try Self.fixture(role: .git)
    let selfGuard = try Self.fixture(role: .selfGuard)

    let foreignRootExpectation = try Self.expectation(
      for: git,
      rootID: String(repeating: "e", count: 64)
    )
    Self.assertFailure(
      fixture: git,
      anchoredExpectation: foreignRootExpectation,
      currentAnchor: foreignRootExpectation.trustAnchor,
      expected: .rootManifestIDMismatch
    )

    let foreignLoaderExpectation = try Self.expectation(
      for: git,
      loaderID: String(repeating: "d", count: 64)
    )
    Self.assertFailure(
      fixture: git,
      anchoredExpectation: foreignLoaderExpectation,
      currentAnchor: foreignLoaderExpectation.trustAnchor,
      expected: .dynamicLoaderManifestIDMismatch
    )

    Self.assertFailure(
      fixture: git,
      memberInventories: [],
      expected: .graph(.memberInventoryCountMismatch)
    )
    Self.assertFailure(
      fixture: git,
      graphComparison: selfGuard.graph,
      expected: .graphComparisonMismatch
    )
    XCTAssertEqual(Self.effects, .zero)
  }

  func testEachReviewedPreimageFieldChangesTheContentID() throws {
    let fixture = try Self.fixture(role: .git)
    let baseline = Self.expectedPreimage(fixture)
    let baselineID = Self.sha256Hex(baseline)
    let baselineLines = String(decoding: baseline, as: UTF8.self)
      .split(separator: "\n", omittingEmptySubsequences: false)
      .map(String.init)
    let replacements = [
      "artifact_role=self-guard",
      "manifest_sha256=\(String(repeating: "f", count: 64))",
      "manifest_bytes=\(fixture.expectation.documentBytes + 1)",
      "root_executable_content_evidence_id="
        + String(repeating: "e", count: 64),
      "dynamic_loader_content_evidence_id="
        + String(repeating: "d", count: 64),
      "shared_cache_set_id=" + String(repeating: "c", count: 64),
    ]

    for (payloadIndex, replacement) in replacements.enumerated() {
      var lines = baselineLines
      lines[payloadIndex + 1] = replacement
      let changed = Data(lines.joined(separator: "\n").utf8)
      XCTAssertNotEqual(Self.sha256Hex(changed), baselineID)
    }
    XCTAssertEqual(Self.effects, .zero)
  }

  func testFileMembersRemainAStageBRefusal() throws {
    let fixture = try Self.fixture(role: .git)
    let fileEvidence = try ExecutableContentIdentityVerifier.derive(
      artifactRole: .fileImage,
      comparison: fixture.root.comparison
    )
    let member =
      try SyntheticRuntimeClosureRecordSchemaVerifier
      .member(
        index: 0,
        source: .file(fileEvidence),
        installName: Self.installName(Self.primaryInstallName)
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
    let collection =
      try SyntheticRuntimeClosureRecordCollectionVerifier.derive(
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

  func testSealedD2FileImageMembersRemainAStageBRefusal() throws {
    let fixture = try Self.fixture(role: .git)
    let snapshot = try FMAFileImageFixture.snapshot(
      identityName: Data(Self.primaryInstallName.utf8)
    )
    let member =
      try SyntheticRuntimeClosureRecordSchemaVerifier
      .member(
        index: 0,
        source: .fileImage(snapshot),
        installName: snapshot.installName
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
    let collection =
      try SyntheticRuntimeClosureRecordCollectionVerifier.derive(
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

  func testCountCacheAndCanonicalSectionRefusalPrecedence() throws {
    let expected = try Self.fixture(role: .git)
    let extraMember = try Self.fixture(
      role: .git,
      memberSpecs: [
        MemberSpec(
          name: Self.primaryInstallName,
          commands: [
            Self.dylibCommand(
              name: Self.secondaryInstallName
            )
          ],
          uuidSeed: 0x31
        ),
        MemberSpec(
          name: Self.secondaryInstallName,
          uuidSeed: 0x32
        ),
      ]
    )
    Self.assertFailure(
      fixture: extraMember,
      anchoredExpectation: expected.expectation,
      currentAnchor: expected.anchor,
      expected: .graphExpectation(.memberCount)
    )

    let foreignCache = try Self.fixture(
      role: .git,
      cacheSeed: "a"
    )
    Self.assertFailure(
      fixture: foreignCache,
      anchoredExpectation: expected.expectation,
      currentAnchor: expected.anchor,
      expected: .memberCacheSetMismatch(index: 0)
    )

    let changedMember = try Self.fixture(
      role: .git,
      imageUUIDSeed: 0x41
    )
    Self.assertFailure(
      fixture: changedMember,
      anchoredExpectation: expected.expectation,
      currentAnchor: expected.anchor,
      expected: .graphExpectation(.canonicalSectionBytes)
    )
    XCTAssertEqual(Self.effects, .zero)
  }

  func testCallerMutationCannotChangeSealedScalarResult() throws {
    var fixture = try Self.fixture(role: .git)
    var callerBytes = fixture.expectationFile.bytes
    let result = try Self.compare(fixture)
    let retained = result

    callerBytes[callerBytes.startIndex] = 0x58
    fixture.members.removeAll()
    fixture.edges.removeAll()
    fixture.memberInventories.removeAll()

    XCTAssertEqual(result, retained)
    XCTAssertEqual(result.memberCount, 1)
    XCTAssertEqual(result.edgeCount, 1)
    XCTAssertEqual(Self.effects, .zero)
  }
}

extension RuntimeClosureContentIdentityTests {
  fileprivate struct Effects: Equatable {
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

  fileprivate struct MemberSpec {
    let name: String
    let commands: [Data]
    let uuidSeed: UInt8

    init(
      name: String,
      commands: [Data] = [],
      uuidSeed: UInt8 = 0x30
    ) {
      self.name = name
      self.commands = commands
      self.uuidSeed = uuidSeed
    }
  }

  fileprivate struct ImageInput {
    let name: String
    let snapshot: SyntheticSharedCacheImageLoadCommandSnapshot

    var contentEvidenceID: String {
      snapshot.imageEvidence.contentEvidenceID.sha256
    }
  }

  fileprivate struct EdgeInput {
    let parent: SyntheticRuntimeClosureEdgeParent
    let parentID: String
    let ordinal: UInt64
    let kind: SyntheticRuntimeClosureEdgeKind
    let resolved: SyntheticRuntimeClosureMemberRecordComparison
  }

  fileprivate struct Fixture {
    let role: RuntimeClosureExpectationArtifactRole
    let root: ExecutableContentIdentityEvidence
    let loader: DynamicLoaderContentIdentityEvidence
    let cacheSet: SyntheticSharedCacheSetIdentityEvidence
    var members: [SyntheticRuntimeClosureMemberRecordComparison]
    var edges: [SyntheticRuntimeClosureEdgeRecordComparison]
    let collection: SyntheticRuntimeClosureRecordCollectionComparison
    let rootInventory: SyntheticAcceptedDependencyCommandInventoryComparison
    var memberInventories: [SyntheticAcceptedDependencyCommandInventoryComparison]
    let graph: SyntheticRuntimeClosureGraphComparison
    let expectationFile: AdmittedFile
    let anchor: RuntimeClosureExpectationTrustAnchor
    let expectation: AnchoredRuntimeClosureExpectationDocument
  }

  fileprivate static let effects = Effects()
  fileprivate static let primaryInstallName = "/usr/lib/libFixture.dylib"
  fileprivate static let secondaryInstallName = "/usr/lib/libSecondary.dylib"

  fileprivate static func fixture(
    role: RuntimeClosureExpectationArtifactRole,
    cacheSeed: Character = "1",
    imageUUIDSeed: UInt8 = 0x30,
    memberSpecs: [MemberSpec]? = nil
  ) throws -> Fixture {
    let specs =
      memberSpecs ?? [
        MemberSpec(
          name: primaryInstallName,
          uuidSeed: imageUUIDSeed
        )
      ]
    let cacheSet = try SyntheticSharedCacheSetIdentityVerifier.derive(
      records: [
        SyntheticSharedCacheFileRecord(
          suffixBytes: 0,
          suffixBase64URL: "",
          fileSHA256: String(repeating: cacheSeed, count: 64),
          fileBytes: 4_096,
          headerUUID: String(repeating: cacheSeed, count: 32)
        )
      ]
    )
    let inputs = try specs.map { spec -> ImageInput in
      let commandBytes = commandRegion(spec.commands)
      let name = installName(spec.name)
      let image =
        try SyntheticSharedCacheImageContentIdentityVerifier
        .derive(
          cacheSetEvidence: cacheSet,
          facts: SyntheticSharedCacheImageContentFacts(
            installNameBytes: name.bytes,
            installNameBase64URL: name.base64URL,
            machOUUID: fixedHex(
              seed: spec.uuidSeed,
              count: 16
            ),
            primaryCodeDirectory: .absent,
            loadCommandsSHA256: sha256Hex(commandBytes)
          )
        )
      return try ImageInput(
        name: spec.name,
        snapshot:
          SyntheticSharedCacheImageLoadCommandSnapshotVerifier
          .derive(
            imageEvidence: image,
            loadCommandBytes: commandBytes
          )
      )
    }.sorted {
      $0.contentEvidenceID.utf8.lexicographicallyPrecedes(
        $1.contentEvidenceID.utf8
      )
    }
    let members = try inputs.enumerated().map { index, input in
      try SyntheticRuntimeClosureRecordSchemaVerifier.member(
        index: index,
        source: .sharedCache(input.snapshot.imageEvidence),
        installName: installName(input.name)
      )
    }
    let snapshotsByID = Dictionary(
      uniqueKeysWithValues: inputs.map {
        ($0.contentEvidenceID, $0.snapshot)
      }
    )
    let memberInventories = try members.map {
      try SyntheticAcceptedDependencyCommandInventoryVerifier
        .sharedCacheMember(snapshotsByID[$0.contentEvidenceID]!)
    }

    let rootTarget = specs[0].name
    let root = try rootEvidence(
      role: executableRole(role),
      loadCommands: [dylibCommand(name: rootTarget)]
    )
    let rootInventory =
      try SyntheticAcceptedDependencyCommandInventoryVerifier
      .root(root)
    let membersByName = Dictionary(
      uniqueKeysWithValues: members.map {
        (String(decoding: $0.decodedInstallName, as: UTF8.self), $0)
      }
    )
    var edgeInputs: [EdgeInput] = []
    for entry in rootInventory.entries {
      let name = String(
        decoding: entry.decodedInstallName,
        as: UTF8.self
      )
      edgeInputs.append(
        EdgeInput(
          parent: .root(root),
          parentID: root.contentEvidenceID.sha256,
          ordinal: entry.loadCommandOrdinal,
          kind: entry.kind,
          resolved: try XCTUnwrap(membersByName[name])
        )
      )
    }
    for (member, inventory) in zip(members, memberInventories) {
      for entry in inventory.entries {
        let name = String(
          decoding: entry.decodedInstallName,
          as: UTF8.self
        )
        edgeInputs.append(
          EdgeInput(
            parent: .member(member),
            parentID: member.contentEvidenceID,
            ordinal: entry.loadCommandOrdinal,
            kind: entry.kind,
            resolved: try XCTUnwrap(membersByName[name])
          )
        )
      }
    }
    let edges = try edgeInputs.sorted(by: edgeLess)
      .enumerated().map { index, input in
        try SyntheticRuntimeClosureRecordSchemaVerifier.edge(
          index: index,
          parent: input.parent,
          loadCommandOrdinal: input.ordinal,
          kind: input.kind,
          installName: input.resolved.installName,
          resolved: input.resolved
        )
      }
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
    let anchor = anchor(for: expectationFile, role: role)
    let expectation = try RuntimeClosureExpectationVerifier.anchor(
      expectationFile: expectationFile,
      trustAnchor: anchor
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
      expectation: expectation
    )
  }

  fileprivate static func compare(
    _ fixture: Fixture
  ) throws -> RuntimeClosureContentExpectationComparison {
    try RuntimeClosureContentIdentityVerifier.compare(
      anchoredExpectation: fixture.expectation,
      currentExpectationAnchor: fixture.anchor,
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

  fileprivate static func expectation(
    for fixture: Fixture,
    rootID: String? = nil,
    loaderID: String? = nil
  ) throws -> AnchoredRuntimeClosureExpectationDocument {
    let canonicalBytes = renderExpectation(
      role: fixture.role,
      rootID: fixture.root.contentEvidenceID.sha256,
      loaderID: fixture.loader.contentEvidenceID.sha256,
      cacheSet: fixture.cacheSet,
      members: fixture.members,
      edges: fixture.edges
    )
    var text = String(decoding: canonicalBytes, as: UTF8.self)
    if let rootID {
      text = text.replacingOccurrences(
        of: fixture.root.contentEvidenceID.sha256,
        with: rootID
      )
    }
    if let loaderID {
      text = text.replacingOccurrences(
        of: fixture.loader.contentEvidenceID.sha256,
        with: loaderID
      )
    }
    let bytes = Data(text.utf8)
    let file = admittedFile(bytes)
    let trustAnchor = anchor(for: file, role: fixture.role)
    return try RuntimeClosureExpectationVerifier.anchor(
      expectationFile: file,
      trustAnchor: trustAnchor
    )
  }

  fileprivate static func assertFailure(
    fixture: Fixture,
    anchoredExpectation:
      AnchoredRuntimeClosureExpectationDocument? = nil,
    currentAnchor: RuntimeClosureExpectationTrustAnchor? = nil,
    root: ExecutableContentIdentityEvidence? = nil,
    members: [SyntheticRuntimeClosureMemberRecordComparison]? = nil,
    edges: [SyntheticRuntimeClosureEdgeRecordComparison]? = nil,
    collection: SyntheticRuntimeClosureRecordCollectionComparison? = nil,
    memberInventories: [SyntheticAcceptedDependencyCommandInventoryComparison]? = nil,
    graphComparison: SyntheticRuntimeClosureGraphComparison? = nil,
    expected: RuntimeClosureContentIdentityFailure,
    file: StaticString = #filePath,
    line: UInt = #line
  ) {
    XCTAssertThrowsError(
      try RuntimeClosureContentIdentityVerifier.compare(
        anchoredExpectation:
          anchoredExpectation ?? fixture.expectation,
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
        $0 as? RuntimeClosureContentIdentityFailure,
        expected,
        file: file,
        line: line
      )
    }
  }

  fileprivate static func assertResult(
    _ result: RuntimeClosureContentExpectationComparison,
    fixture: Fixture,
    file: StaticString = #filePath,
    line: UInt = #line
  ) throws {
    let expectedPreimage = expectedPreimage(fixture)
    XCTAssertEqual(
      result.contentEvidenceID.sha256,
      sha256Hex(expectedPreimage),
      file: file,
      line: line
    )
    XCTAssertEqual(
      result.contentEvidenceID.sha256,
      fixture.role == .git
        ? "44e0279168fbb49377aaa669273f9fde" + "9cf20f4b1e8488fb77bdc9936307b7ed"
        : "f67e3907da57ac787d5b1e3681f4970" + "81b59d641ee8c4cbdab37570d91e88974",
      file: file,
      line: line
    )
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
    XCTAssertTrue(result.provesExpectationAnchorMatch, file: file, line: line)
    XCTAssertTrue(result.provesManifestContentMatch, file: file, line: line)
    XCTAssertTrue(result.provesDeclaredStaticGraphMatch, file: file, line: line)
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

    let labels = Set(
      Mirror(reflecting: result).children.compactMap(\.label)
    )
    for forbidden in [
      "expectationFile", "identityPreimage", "members", "edges",
      "collection", "graphComparison", "inventory", "anchor",
      "path", "fileDescriptor", "argv", "process", "sandbox",
      "nonce", "receipt", "capability",
    ] {
      XCTAssertFalse(labels.contains(forbidden), file: file, line: line)
    }
  }

  fileprivate static func executableRole(
    _ role: RuntimeClosureExpectationArtifactRole
  ) -> ExecutableContentArtifactRole {
    switch role {
    case .git: .git
    case .selfGuard: .selfGuard
    }
  }

  fileprivate static func edgeLess(_ lhs: EdgeInput, _ rhs: EdgeInput) -> Bool {
    if lhs.parentID != rhs.parentID {
      return lhs.parentID.utf8.lexicographicallyPrecedes(
        rhs.parentID.utf8
      )
    }
    return lhs.ordinal < rhs.ordinal
  }

  fileprivate static func renderExpectation(
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
        "\(prefix)primary_code_directory_blob_sha256=\(member.primaryCodeDirectoryBlobSHA256)")
      lines.append("\(prefix)load_commands_sha256=\(member.loadCommandsSHA256)")
    }
    lines.append("edge_count=\(edges.count)")
    for (index, edge) in edges.enumerated() {
      let prefix = "edge_\(fourDigit(index))_"
      lines.append("\(prefix)parent_content_evidence_id=\(edge.parentContentEvidenceID)")
      lines.append("\(prefix)load_command_ordinal=\(edge.loadCommandOrdinal)")
      lines.append("\(prefix)kind=\(edge.kind.rawValue)")
      lines.append("\(prefix)install_name_bytes=\(edge.installName.bytes)")
      lines.append("\(prefix)install_name_base64url=\(edge.installName.base64URL)")
      lines.append("\(prefix)resolved_content_evidence_id=\(edge.resolvedContentEvidenceID)")
    }
    lines.append("runtime_resolution_outcome=unproved-static-comparison-only")
    lines.append("runtime_authority=none")
    return Data((lines.joined(separator: "\n") + "\n").utf8)
  }

  fileprivate static func admittedFile(_ bytes: Data) -> AdmittedFile {
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

  fileprivate static func anchor(
    for file: AdmittedFile,
    role: RuntimeClosureExpectationArtifactRole,
    sha256: String? = nil,
    bytes: UInt64? = nil,
    minimumGeneration: UInt64 = 9,
    verificationUnixSeconds: UInt64 = 2_000_000_000
  ) -> RuntimeClosureExpectationTrustAnchor {
    RuntimeClosureExpectationTrustAnchor(
      expectedCurrentDocumentSHA256: sha256 ?? file.sha256,
      expectedCurrentDocumentBytes: bytes ?? UInt64(file.bytes.count),
      minimumEvidenceGeneration: minimumGeneration,
      verificationUnixSeconds: verificationUnixSeconds,
      expectedArtifactRole: role
    )
  }

  fileprivate static func expectedPreimage(_ fixture: Fixture) -> Data {
    Data(
      ([
        RuntimeClosureContentIdentityVerifier.identityDomain,
        "artifact_role=\(fixture.role.rawValue)",
        "manifest_sha256=\(fixture.expectation.documentSHA256)",
        "manifest_bytes=\(fixture.expectation.documentBytes)",
        "root_executable_content_evidence_id=" + fixture.root.contentEvidenceID.sha256,
        "dynamic_loader_content_evidence_id=" + fixture.loader.contentEvidenceID.sha256,
        "shared_cache_set_id=" + fixture.cacheSet.sharedCacheSetID.sha256,
      ].joined(separator: "\n") + "\n").utf8
    )
  }

  fileprivate static func rootEvidence(
    role: ExecutableContentArtifactRole,
    loadCommands: [Data]
  ) throws -> ExecutableContentIdentityEvidence {
    let signature = superBlob(
      entries: [
        (
          0,
          codeDirectory(
            signingIdentifier:
              Data("com.example.c2.\(role.rawValue)".utf8)
          )
        )
      ]
    )
    let comparison = try SyntheticMachOIdentityParser.parse(
      executeMachO(
        loadCommands: loadCommands,
        signatureRegion: signature
      )
    )
    return try ExecutableContentIdentityVerifier.derive(
      artifactRole: role,
      comparison: comparison
    )
  }

  fileprivate static func dynamicLoaderEvidence()
    throws -> DynamicLoaderContentIdentityEvidence
  {
    let signature = superBlob(
      entries: [
        (
          0,
          codeDirectory(
            signingIdentifier: Data("com.example.c2.dyld".utf8)
          )
        )
      ]
    )
    let comparison =
      try SyntheticDynamicLoaderMachOIdentityParser.parse(
        dynamicLoaderMachO(signatureRegion: signature)
      )
    return try DynamicLoaderContentIdentityVerifier.derive(
      comparison: comparison
    )
  }

  fileprivate static func executeMachO(
    loadCommands extraCommands: [Data],
    signatureRegion: Data
  ) -> Data {
    var commands = uuidCommand(seed: 0x10)
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

  fileprivate static func dynamicLoaderMachO(signatureRegion: Data) -> Data {
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

  fileprivate static func machOHeader(
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

  fileprivate static func uuidCommand(seed: UInt8) -> Data {
    var result = Data()
    appendUInt32LE(&result, 0x1b)
    appendUInt32LE(&result, 24)
    result.append(
      contentsOf: (0..<16).map { seed &+ UInt8($0) }
    )
    return result
  }

  fileprivate static func dynamicLoaderIdentityCommand() -> Data {
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

  fileprivate static func dylibCommand(name: String) -> Data {
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

  fileprivate static func commandRegion(_ commands: [Data]) -> Data {
    commands.reduce(into: Data()) { $0.append($1) }
  }

  fileprivate static func installName(
    _ value: String
  ) -> SyntheticRuntimeClosureInstallName {
    let bytes = Data(value.utf8)
    return SyntheticRuntimeClosureInstallName(
      bytes: UInt64(bytes.count),
      base64URL: base64URL(bytes)
    )
  }

  fileprivate static func superBlob(entries: [(UInt32, Data)]) -> Data {
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

  fileprivate static func codeDirectory(signingIdentifier: Data) -> Data {
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

  fileprivate static func fixedHex(seed: UInt8, count: Int) -> String {
    (0..<count).map {
      String(format: "%02x", seed &+ UInt8($0))
    }.joined()
  }

  fileprivate static func fourDigit(_ value: Int) -> String {
    String(format: "%04d", value)
  }

  fileprivate static func base64URL(_ data: Data) -> String {
    data.base64EncodedString()
      .replacingOccurrences(of: "+", with: "-")
      .replacingOccurrences(of: "/", with: "_")
      .replacingOccurrences(of: "=", with: "")
  }

  fileprivate static func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data)
      .map { String(format: "%02x", $0) }
      .joined()
  }

  fileprivate static func appendUInt32LE(_ data: inout Data, _ value: UInt32) {
    data.append(UInt8(truncatingIfNeeded: value))
    data.append(UInt8(truncatingIfNeeded: value >> 8))
    data.append(UInt8(truncatingIfNeeded: value >> 16))
    data.append(UInt8(truncatingIfNeeded: value >> 24))
  }

  fileprivate static func appendUInt32BE(_ data: inout Data, _ value: UInt32) {
    data.append(UInt8(truncatingIfNeeded: value >> 24))
    data.append(UInt8(truncatingIfNeeded: value >> 16))
    data.append(UInt8(truncatingIfNeeded: value >> 8))
    data.append(UInt8(truncatingIfNeeded: value))
  }

  fileprivate static func writeUInt32LE(
    _ data: inout Data,
    at offset: Int,
    value: UInt32
  ) {
    data[offset] = UInt8(truncatingIfNeeded: value)
    data[offset + 1] = UInt8(truncatingIfNeeded: value >> 8)
    data[offset + 2] = UInt8(truncatingIfNeeded: value >> 16)
    data[offset + 3] = UInt8(truncatingIfNeeded: value >> 24)
  }

  fileprivate static func writeUInt32BE(
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
