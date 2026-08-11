import CryptoKit
import Foundation
import XCTest

@testable import ProofControl

final class FileImageExecutionIdentityComparisonTests: XCTestCase {
    func testCanonicalP3DerivesExactCompactNoGoEvidence() throws {
        _ = FileImageExecutionIdentityAnchorContextID.self
        _ = FileImageExecutionIdentityComparisonFailure.self
        _ = FileImageExecutionIdentityComparisonEvidence.self
        _ = FileImageExecutionIdentityComparisonVerifier.self

        let fixture = try Self.p3Fixture()
        let result = try Self.p3Compare(fixture)
        let repeated = try Self.p3Compare(fixture)
        let context = Self.p3ContextPreimage(fixture)
        let final = Self.p3FinalPreimage(
            fixture,
            contextID: Self.sha256Hex(context)
        )
        let expectedContext = Data(
            ("""
            fast-mlx-proof-control-file-image-execution-identity-anchor-context-id-v1
            git_tool_policy_v2_expected_sha256=4a0dacc1258d6d07ee1b31cbb455f093b088fcd8e847fe32e58d084ccd67a4fa
            git_tool_policy_v2_expected_bytes=1908
            git_tool_policy_v2_minimum_generation=7
            git_tool_policy_v2_verification_unix_seconds=2000000000
            runtime_denial_v2_expected_sha256=f3c67dfdda5dcde5d7fd57226f375457804db97a98a6932d39d98520c64bbf2f
            runtime_denial_v2_expected_bytes=2449
            runtime_denial_v2_minimum_generation=7
            runtime_denial_v2_verification_unix_seconds=2000000000
            git_executable_expectation_sha256=cb25bbfe9663ff8c0ca73eed5c1cdc9b2046a451f5d353addfa947866fe7a2ee
            git_executable_expectation_bytes=1449
            git_executable_minimum_generation=9
            git_executable_verification_unix_seconds=2000000000
            self_guard_executable_expectation_sha256=d240b4bdef208dc117a9a23f70059c10a030d179d6c8f3454200cbcba7c3af3e
            self_guard_executable_expectation_bytes=1467
            self_guard_executable_minimum_generation=9
            self_guard_executable_verification_unix_seconds=2000000000
            git_closure_expectation_sha256=8a4f21652610ece8c942c75ee7dec96db11d2ab7fe1ac20d86d227f044540da2
            git_closure_expectation_bytes=1983
            git_closure_minimum_generation=9
            git_closure_verification_unix_seconds=2000000000
            self_guard_closure_expectation_sha256=7e414e68fe5440e2a18a47769ab702c6e02c112404240837e9d36585a0170de2
            self_guard_closure_expectation_bytes=1990
            self_guard_closure_minimum_generation=9
            self_guard_closure_verification_unix_seconds=2000000000
            """ + "\n").utf8
        )
        let expectedFinal = Data(
            ("""
            fast-mlx-proof-control-file-image-execution-identity-comparison-id-v1
            run_claim_id=3027fcf02b0f6601c355c20f2331788e9a0efb033ac8d0cd0d13b43ab58b9385
            anchor_context_id=d2d1bf4e5a80896ef0b0ccda60b959b5212e8a1bd97286276df72a7295c76ac6
            git_tool_policy_v2_sha256=4a0dacc1258d6d07ee1b31cbb455f093b088fcd8e847fe32e58d084ccd67a4fa
            runtime_denial_policy_v2_sha256=f3c67dfdda5dcde5d7fd57226f375457804db97a98a6932d39d98520c64bbf2f
            git_executable_expectation_sha256=cb25bbfe9663ff8c0ca73eed5c1cdc9b2046a451f5d353addfa947866fe7a2ee
            git_executable_content_evidence_id=faf601f4b03581b972bbd683ae578d0468d836a722839bf44cca9a8618849885
            self_guard_executable_expectation_sha256=d240b4bdef208dc117a9a23f70059c10a030d179d6c8f3454200cbcba7c3af3e
            self_guard_executable_content_evidence_id=aa51f0b2ee0eb48dc878f64720d888d268e3d96a632e1eb08d5325ee5541393c
            git_closure_manifest_sha256=8a4f21652610ece8c942c75ee7dec96db11d2ab7fe1ac20d86d227f044540da2
            git_file_image_runtime_closure_content_evidence_id=0f08e67415675fb5ce9738107e1886379a4f6b48047b5f49a433e81b62a85e4e
            self_guard_closure_manifest_sha256=7e414e68fe5440e2a18a47769ab702c6e02c112404240837e9d36585a0170de2
            self_guard_file_image_runtime_closure_content_evidence_id=f0717cc182bf4728f91e6f8fe1b4c04c8dddac31f9668d4829ac427a224cd20a
            runtime_decision=no-go
            """ + "\n").utf8
        )

        XCTAssertEqual(result, repeated)
        XCTAssertEqual(context, expectedContext)
        XCTAssertEqual(final, expectedFinal)
        XCTAssertEqual(context.count, 1_472)
        XCTAssertEqual(final.count, 1_286)
        XCTAssertEqual(
            result.anchorContextID.sha256,
            "d2d1bf4e5a80896ef0b0ccda60b959b5212e8a1bd97286276df72a7295c76ac6"
        )
        XCTAssertEqual(
            result.comparisonSHA256,
            "4ffe34267841e0d117a54803612342fe424c104f4cc1c7a30a2858d91d02e7f4"
        )
        XCTAssertTrue(result.provesSignedClaimToolPolicyMatch)
        XCTAssertTrue(result.provesRuntimeDenialMatch)
        XCTAssertTrue(result.provesExecutableIdentityMatch)
        XCTAssertTrue(result.provesFileImageRuntimeClosureIdentityMatch)
        XCTAssertTrue(result.provesSingleCurrentAnchorContext)
        XCTAssertEqual(result.runtimeDecision, .noGo)
        Self.assertP3NoAuthority(result)
        let labels = Set(Mirror(reflecting: result).children.compactMap(\.label))
        XCTAssertEqual(labels, Self.p3ResultScalarAllowlist)
        XCTAssertEqual(Self.effects, .zero)
    }

    func testP3AnchorScalarsRolesAndCommonSecondWinFirst() throws {
        let fixture = try Self.p3Fixture()
        Self.assertP3Failure(
            fixture,
            toolAnchor: GitToolPolicyV2TrustAnchor(
                expectedCurrentPolicySHA256:
                    fixture.toolAnchor.expectedCurrentPolicySHA256.uppercased(),
                expectedCurrentPolicyBytes:
                    fixture.toolAnchor.expectedCurrentPolicyBytes,
                minimumPolicyGeneration: 7,
                verificationUnixSeconds: Self.p3CommonTime
            ),
            expected: .invalidAnchorScalar(
                slot: .gitToolPolicy,
                field: .digest
            )
        )
        Self.assertP3Failure(
            fixture,
            denialAnchor: GitRuntimePolicyDenialV2TrustAnchor(
                expectedCurrentPolicySHA256:
                    fixture.denialAnchor.expectedCurrentPolicySHA256,
                expectedCurrentPolicyBytes: 0,
                minimumPolicyGeneration: 7,
                verificationUnixSeconds: Self.p3CommonTime
            ),
            expected: .invalidAnchorScalar(
                slot: .runtimeDenial,
                field: .byteCount
            )
        )
        let mixedTime = GitRuntimePolicyDenialV2TrustAnchor(
            expectedCurrentPolicySHA256:
                fixture.denialAnchor.expectedCurrentPolicySHA256,
            expectedCurrentPolicyBytes:
                fixture.denialAnchor.expectedCurrentPolicyBytes,
            minimumPolicyGeneration: 7,
            verificationUnixSeconds: Self.p3CommonTime + 1
        )
        Self.assertP3Failure(
            fixture,
            denialAnchor: mixedTime,
            expected: .anchorEvaluationTimeMismatch(.runtimeDenial)
        )
        XCTAssertEqual(Self.effects, .zero)
    }

    func testP3PredecessorRolesAndTypedReanchorFailuresPrecedeJoins()
        throws
    {
        let fixture = try Self.p3Fixture()
        Self.assertP3Failure(
            fixture,
            toolAnchor: GitToolPolicyV2TrustAnchor(
                expectedCurrentPolicySHA256: String(repeating: "f", count: 64),
                expectedCurrentPolicyBytes:
                    fixture.toolAnchor.expectedCurrentPolicyBytes,
                minimumPolicyGeneration: 7,
                verificationUnixSeconds: Self.p3CommonTime
            ),
            expected: .gitToolPolicyReanchor(.policyDigestMismatch)
        )
        Self.assertP3Failure(
            fixture,
            gitExecutable: fixture.selfGuardExecutable,
            expected: .roleSubstitution(slot: .gitExecutableEvidence)
        )
        Self.assertP3Failure(
            fixture,
            gitClosureReference: fixture.selfGuardClosure.reference,
            expected: .roleSubstitution(slot: .gitClosureReference)
        )
        Self.assertP3Failure(
            fixture,
            gitClosureComparison: fixture.selfGuardClosureComparison,
            expected: .roleSubstitution(slot: .gitClosureComparison)
        )
        XCTAssertEqual(Self.effects, .zero)
    }

    func testP3SingletonCacheMismatchRefusesBeforeIdentity() throws {
        let fixture = try Self.p3Fixture(selfGuardCacheSeed: "2")
        Self.assertP3Failure(
            fixture,
            expected: .sharedEnvironmentMismatch(.sharedCacheSetID)
        )
        XCTAssertEqual(Self.effects, .zero)
    }

    func testP3CallerMutationCannotChangeSealedEvidence() throws {
        let fixture = try Self.p3Fixture()
        let before = try Self.p3Compare(fixture)
        let retained = [
            fixture.toolReference.policyDocument.policyFile.bytes,
            fixture.denial.policyFile.bytes,
            fixture.gitExecutable.expectation.expectationFile.bytes,
            fixture.selfGuardExecutable.expectation.expectationFile.bytes,
            fixture.gitClosure.expectation.expectationFile.bytes,
            fixture.selfGuardClosure.expectation.expectationFile.bytes,
        ]
        var callerOwned = retained
        for index in callerOwned.indices {
            callerOwned[index][callerOwned[index].startIndex] ^= 0xff
            XCTAssertNotEqual(callerOwned[index], retained[index])
        }
        XCTAssertEqual(
            fixture.toolReference.policyDocument.policyFile.bytes,
            retained[0]
        )
        XCTAssertEqual(fixture.denial.policyFile.bytes, retained[1])
        XCTAssertEqual(
            fixture.gitExecutable.expectation.expectationFile.bytes,
            retained[2]
        )
        XCTAssertEqual(
            fixture.selfGuardExecutable.expectation.expectationFile.bytes,
            retained[3]
        )
        XCTAssertEqual(
            fixture.gitClosure.expectation.expectationFile.bytes,
            retained[4]
        )
        XCTAssertEqual(
            fixture.selfGuardClosure.expectation.expectationFile.bytes,
            retained[5]
        )
        XCTAssertEqual(try Self.p3Compare(fixture), before)
        XCTAssertEqual(Self.effects, .zero)
    }

    func testP3EveryAnchorRejectsEveryScalarBeforeEvidence() throws {
        let fixture = try Self.p3Fixture()
        Self.assertP3ToolAnchorScalarFailures(fixture)
        Self.assertP3DenialAnchorScalarFailures(fixture)
        Self.assertP3ExecutableAnchorScalarFailures(
            fixture,
            original: fixture.gitExecutableAnchor,
            slot: .gitExecutableExpectation,
            git: true
        )
        Self.assertP3ExecutableAnchorScalarFailures(
            fixture,
            original: fixture.selfGuardExecutableAnchor,
            slot: .selfGuardExecutableExpectation,
            git: false
        )
        Self.assertP3ClosureAnchorScalarFailures(
            fixture,
            original: fixture.gitClosure.anchor,
            slot: .gitClosureExpectation,
            git: true
        )
        Self.assertP3ClosureAnchorScalarFailures(
            fixture,
            original: fixture.selfGuardClosure.anchor,
            slot: .selfGuardClosureExpectation,
            git: false
        )
        XCTAssertEqual(Self.effects, .zero)
    }

    func testP3EveryNonToolAnchorRejectsMixedEvaluationSecond()
        throws
    {
        let fixture = try Self.p3Fixture()
        let changed = Self.p3CommonTime + 1
        Self.assertP3Failure(
            fixture,
            denialAnchor: Self.p3DenialAnchor(fixture, time: changed),
            expected: .anchorEvaluationTimeMismatch(.runtimeDenial)
        )
        Self.assertP3Failure(
            fixture,
            gitExecutableAnchor: Self.p3ExecutableAnchor(
                fixture.gitExecutableAnchor,
                time: changed
            ),
            expected:
                .anchorEvaluationTimeMismatch(.gitExecutableExpectation)
        )
        Self.assertP3Failure(
            fixture,
            selfGuardExecutableAnchor: Self.p3ExecutableAnchor(
                fixture.selfGuardExecutableAnchor,
                time: changed
            ),
            expected: .anchorEvaluationTimeMismatch(
                .selfGuardExecutableExpectation
            )
        )
        Self.assertP3Failure(
            fixture,
            gitClosureAnchor: Self.p3ClosureAnchor(
                fixture.gitClosure.anchor,
                time: changed
            ),
            expected: .anchorEvaluationTimeMismatch(.gitClosureExpectation)
        )
        Self.assertP3Failure(
            fixture,
            selfGuardClosureAnchor: Self.p3ClosureAnchor(
                fixture.selfGuardClosure.anchor,
                time: changed
            ),
            expected: .anchorEvaluationTimeMismatch(
                .selfGuardClosureExpectation
            )
        )
        XCTAssertEqual(Self.effects, .zero)
    }

    func testP3EveryClosedRoleSlotRejectsSubstitutionInStageOrder()
        throws
    {
        let fixture = try Self.p3Fixture()
        Self.assertP3Failure(
            fixture,
            gitExecutableAnchor: Self.p3ExecutableAnchor(
                fixture.gitExecutableAnchor,
                role: .selfGuard
            ),
            expected: .roleSubstitution(slot: .gitExecutableAnchor)
        )
        Self.assertP3Failure(
            fixture,
            selfGuardExecutableAnchor: Self.p3ExecutableAnchor(
                fixture.selfGuardExecutableAnchor,
                role: .git
            ),
            expected: .roleSubstitution(
                slot: .selfGuardExecutableAnchor
            )
        )
        Self.assertP3Failure(
            fixture,
            gitClosureAnchor: Self.p3ClosureAnchor(
                fixture.gitClosure.anchor,
                role: .selfGuard
            ),
            expected: .roleSubstitution(slot: .gitClosureAnchor)
        )
        Self.assertP3Failure(
            fixture,
            selfGuardClosureAnchor: Self.p3ClosureAnchor(
                fixture.selfGuardClosure.anchor,
                role: .git
            ),
            expected: .roleSubstitution(slot: .selfGuardClosureAnchor)
        )
        Self.assertP3Failure(
            fixture,
            gitExecutable: fixture.selfGuardExecutable,
            expected: .roleSubstitution(slot: .gitExecutableEvidence)
        )
        Self.assertP3Failure(
            fixture,
            selfGuardExecutable: fixture.gitExecutable,
            expected: .roleSubstitution(
                slot: .selfGuardExecutableEvidence
            )
        )
        Self.assertP3Failure(
            fixture,
            gitClosureExpectation: fixture.selfGuardClosure.expectation,
            expected: .roleSubstitution(slot: .gitClosureEvidence)
        )
        Self.assertP3Failure(
            fixture,
            selfGuardClosureExpectation: fixture.gitClosure.expectation,
            expected: .roleSubstitution(slot: .selfGuardClosureEvidence)
        )
        Self.assertP3Failure(
            fixture,
            gitClosureReference: fixture.selfGuardClosure.reference,
            expected: .roleSubstitution(slot: .gitClosureReference)
        )
        Self.assertP3Failure(
            fixture,
            selfGuardClosureReference: fixture.gitClosure.reference,
            expected: .roleSubstitution(slot: .selfGuardClosureReference)
        )
        Self.assertP3Failure(
            fixture,
            gitClosureComparison: fixture.selfGuardClosureComparison,
            expected: .roleSubstitution(slot: .gitClosureComparison)
        )
        Self.assertP3Failure(
            fixture,
            selfGuardClosureComparison: fixture.gitClosureComparison,
            expected: .roleSubstitution(
                slot: .selfGuardClosureComparison
            )
        )
        Self.assertP3Failure(
            fixture,
            gitExecutable: fixture.selfGuardExecutable,
            selfGuardExecutable: fixture.gitExecutable,
            expected: .roleSubstitution(slot: .gitExecutableEvidence)
        )
        XCTAssertEqual(Self.effects, .zero)
    }

    func testP3TypedReanchorsAndReferenceEqualityPrecedeComparisons()
        throws
    {
        let fixture = try Self.p3Fixture()
        let otherDigest = String(repeating: "f", count: 64)
        Self.assertP3Failure(
            fixture,
            toolAnchor: Self.p3ToolAnchor(
                fixture,
                digest: otherDigest
            ),
            expected: .gitToolPolicyReanchor(.policyDigestMismatch)
        )
        Self.assertP3Failure(
            fixture,
            denialAnchor: Self.p3DenialAnchor(
                fixture,
                digest: otherDigest
            ),
            expected: .runtimeDenialReanchor(.policyDigestMismatch)
        )
        Self.assertP3Failure(
            try Self.p3Fixture(toolRuntimePolicySHA256: otherDigest),
            expected: .runtimeDenialReference(
                .runtimePolicyDigestMismatch
            )
        )
        Self.assertP3Failure(
            try Self.p3Fixture(
                denialLimits: Self.p3Limits(
                    maxOpenFiles:
                        GitToolPolicyVerifier.phase1ResourceCeilings
                        .maxOpenFiles - 1
                )
            ),
            expected: .runtimeDenialReference(
                .toolPolicyResourceMismatch(.maxOpenFiles)
            )
        )
        Self.assertP3Failure(
            fixture,
            gitExecutableAnchor: Self.p3ExecutableAnchor(
                fixture.gitExecutableAnchor,
                digest: otherDigest
            ),
            expected: .executableExpectation(
                role: .git,
                failure: .documentDigestMismatch
            )
        )
        Self.assertP3Failure(
            fixture,
            selfGuardExecutableAnchor: Self.p3ExecutableAnchor(
                fixture.selfGuardExecutableAnchor,
                digest: otherDigest
            ),
            expected: .executableExpectation(
                role: .selfGuard,
                failure: .documentDigestMismatch
            )
        )
        Self.assertP3Failure(
            fixture,
            gitClosureAnchor: Self.p3ClosureAnchor(
                fixture.gitClosure.anchor,
                digest: otherDigest
            ),
            expected: .closureExpectation(
                role: .git,
                failure: .documentDigestMismatch
            )
        )
        Self.assertP3Failure(
            fixture,
            selfGuardClosureAnchor: Self.p3ClosureAnchor(
                fixture.selfGuardClosure.anchor,
                digest: otherDigest
            ),
            expected: .closureExpectation(
                role: .selfGuard,
                failure: .documentDigestMismatch
            )
        )

        let otherGit = try Self.fixture(role: .git, cacheSeed: "2")
        let otherSelfGuard = try Self.fixture(
            role: .selfGuard,
            cacheSeed: "2"
        )
        Self.assertP3Failure(
            fixture,
            gitClosureReference: otherGit.reference,
            expected: .closureReferenceMismatch(role: .git)
        )
        Self.assertP3Failure(
            fixture,
            selfGuardClosureReference: otherSelfGuard.reference,
            expected: .closureReferenceMismatch(role: .selfGuard)
        )
        Self.assertP3Failure(
            fixture,
            gitClosureComparison: try Self.compare(otherGit),
            expected: .closureContinuity(
                role: .git,
                field: .manifestDigest
            )
        )
        Self.assertP3Failure(
            fixture,
            selfGuardClosureComparison: try Self.compare(otherSelfGuard),
            expected: .closureContinuity(
                role: .selfGuard,
                field: .manifestDigest
            )
        )
        XCTAssertEqual(Self.effects, .zero)
    }

    func testP3DirectionalPolicyAndEnvironmentJoinsRefuseInOrder()
        throws
    {
        let changedHex = String(repeating: "e", count: 64)
        let changedUUID = String(repeating: "e", count: 32)

        Self.assertP3Failure(
            try Self.p3Fixture(toolExecutableSHA256: changedHex),
            expected: .executablePolicyContinuity(
                role: .git,
                field: .fileDigest
            )
        )
        Self.assertP3Failure(
            try Self.p3Fixture(toolExecutableBytes: 1),
            expected: .executablePolicyContinuity(
                role: .git,
                field: .fileBytes
            )
        )
        Self.assertP3Failure(
            try Self.p3Fixture(toolExecutableMachOUUID: changedUUID),
            expected: .executablePolicyContinuity(
                role: .git,
                field: .machOUUID
            )
        )
        Self.assertP3Failure(
            try Self.p3Fixture(
                toolExecutableCodeDirectorySHA256: changedHex
            ),
            expected: .executablePolicyContinuity(
                role: .git,
                field: .primaryCodeDirectoryDigest
            )
        )
        Self.assertP3Failure(
            try Self.p3Fixture(denialSelfGuardSHA256: changedHex),
            expected: .executablePolicyContinuity(
                role: .selfGuard,
                field: .fileDigest
            )
        )
        Self.assertP3Failure(
            try Self.p3Fixture(denialSelfGuardBytes: 1),
            expected: .executablePolicyContinuity(
                role: .selfGuard,
                field: .fileBytes
            )
        )
        Self.assertP3Failure(
            try Self.p3Fixture(denialSelfGuardMachOUUID: changedUUID),
            expected: .executablePolicyContinuity(
                role: .selfGuard,
                field: .machOUUID
            )
        )
        Self.assertP3Failure(
            try Self.p3Fixture(
                denialSelfGuardCodeDirectorySHA256: changedHex
            ),
            expected: .executablePolicyContinuity(
                role: .selfGuard,
                field: .primaryCodeDirectoryDigest
            )
        )
        Self.assertP3Failure(
            try Self.p3Fixture(
                toolGitClosureManifestSHA256: changedHex
            ),
            expected: .closureContinuity(
                role: .git,
                field: .manifestDigest
            )
        )
        Self.assertP3Failure(
            try Self.p3Fixture(toolGitClosureManifestBytes: 1),
            expected: .closureContinuity(
                role: .git,
                field: .manifestBytes
            )
        )
        Self.assertP3Failure(
            try Self.p3Fixture(
                toolGitClosureContentEvidenceID: changedHex
            ),
            expected: .closureContinuity(
                role: .git,
                field: .toolContentEvidenceID
            )
        )
        Self.assertP3Failure(
            try Self.p3Fixture(
                denialGitClosureContentEvidenceID: changedHex
            ),
            expected: .closureContinuity(
                role: .git,
                field: .denialContentEvidenceID
            )
        )
        Self.assertP3Failure(
            try Self.p3Fixture(
                denialSelfGuardClosureContentEvidenceID: changedHex
            ),
            expected: .closureContinuity(
                role: .selfGuard,
                field: .denialContentEvidenceID
            )
        )
        Self.assertP3Failure(
            try Self.p3Fixture(selfGuardLoaderUUIDSeed: 0x21),
            expected: .sharedEnvironmentMismatch(.dynamicLoaderContentID)
        )
        Self.assertP3Failure(
            try Self.p3Fixture(selfGuardCacheSeed: "2"),
            expected: .sharedEnvironmentMismatch(.sharedCacheSetID)
        )
        XCTAssertEqual(Self.effects, .zero)
    }

    func testP3PlatformMismatchBranchesAreSealedByStageC1Terminals()
        throws
    {
        let fixture = try Self.p3Fixture()
        XCTAssertEqual(
            fixture.gitClosure.expectation.fields.platformArchitecture,
            "arm64"
        )
        XCTAssertEqual(
            fixture.gitClosure.expectation.fields.platformHardwareModel,
            "Mac15,14"
        )
        XCTAssertEqual(
            fixture.gitClosure.expectation.fields.platformOSVersion,
            "26.5.2"
        )
        XCTAssertEqual(
            fixture.gitClosure.expectation.fields.platformOSBuild,
            "25F84"
        )

        let substitutions = [
            (
                "platform_architecture=arm64",
                "platform_architecture=x86_64"
            ),
            (
                "platform_hardware_model=Mac15,14",
                "platform_hardware_model=Mac15,15"
            ),
            (
                "platform_os_version=26.5.2",
                "platform_os_version=26.5.3"
            ),
            (
                "platform_os_build=25F84",
                "platform_os_build=25F85"
            ),
        ]
        let canonical = String(
            decoding: fixture.gitClosure.expectation.expectationFile.bytes,
            as: UTF8.self
        )
        for (expected, replacement) in substitutions {
            let changed = Data(
                canonical.replacingOccurrences(
                    of: expected,
                    with: replacement
                ).utf8
            )
            let file = Self.admittedFile(changed)
            let anchor = RuntimeClosureExpectationTrustAnchor(
                expectedCurrentDocumentSHA256: file.sha256,
                expectedCurrentDocumentBytes: UInt64(file.bytes.count),
                minimumEvidenceGeneration: 9,
                verificationUnixSeconds: Self.p3CommonTime,
                expectedArtifactRole: .git
            )
            XCTAssertThrowsError(
                try RuntimeClosureExpectationVerifier.anchor(
                    expectationFile: file,
                    trustAnchor: anchor
                )
            ) { error in
                XCTAssertEqual(
                    error as? RuntimeClosureExpectationError,
                    .nonCanonicalDocument
                )
            }
        }
        XCTAssertEqual(Self.effects, .zero)
    }

    func testP3EveryContextAndFinalPayloadMutationChangesItsID()
        throws
    {
        let fixture = try Self.p3Fixture()
        let context = Self.p3ContextPreimage(fixture)
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
                Self.sha256Hex(
                    Data(changed.joined(separator: "\n").utf8)
                ),
                contextID
            )
        }

        let final = Self.p3FinalPreimage(fixture, contextID: contextID)
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
                Self.sha256Hex(
                    Data(changed.joined(separator: "\n").utf8)
                ),
                finalID
            )
        }
        var changedTerminal = finalLines
        changedTerminal[13] = "runtime_decision=go"
        XCTAssertNotEqual(
            Self.sha256Hex(
                Data(changedTerminal.joined(separator: "\n").utf8)
            ),
            finalID
        )
        XCTAssertEqual(Self.effects, .zero)
    }

    func testReviewedD2C2TypesExist() {
        _ = FileImageRuntimeClosureContentEvidenceID.self
        _ = FileImageRuntimeClosureContentIdentityFailure.self
        _ = FileImageRuntimeClosureContentExpectationComparison.self
        _ = FileImageRuntimeClosureContentIdentityVerifier.self
    }

    func testFrozenCompleteGitAndSelfGuardPredecessorFixtures() throws {
        let git = try Self.fixture(role: .git)
        let selfGuard = try Self.fixture(role: .selfGuard)

        let gitPreimage = Self.expectedPreimage(git)
        let selfGuardPreimage = Self.expectedPreimage(selfGuard)
        XCTAssertEqual(gitPreimage.count, 504)
        XCTAssertEqual(
            Self.sha256Hex(gitPreimage),
            "0f08e67415675fb5ce9738107e1886379" +
                "a4f6b48047b5f49a433e81b62a85e4e"
        )
        XCTAssertEqual(selfGuardPreimage.count, 511)
        XCTAssertEqual(
            Self.sha256Hex(selfGuardPreimage),
            "f0717cc182bf4728f91e6f8fe1b4c04c" +
                "8dddac31f9668d4829ac427a224cd20a"
        )

        XCTAssertEqual(git.reference.declaredFileImageMemberCount, 1)
        XCTAssertEqual(selfGuard.reference.declaredFileImageMemberCount, 1)
        XCTAssertEqual(git.graph.fileImageMemberCount, 1)
        XCTAssertEqual(selfGuard.graph.fileImageMemberCount, 1)
    }

    func testCanonicalRolesProduceExactCompactInertResults() throws {
        let git = try Self.fixture(role: .git)
        let selfGuard = try Self.fixture(role: .selfGuard)

        let gitResult = try Self.compare(git)
        let repeatedGitResult = try Self.compare(git)
        let selfGuardResult = try Self.compare(selfGuard)

        XCTAssertEqual(gitResult, repeatedGitResult)
        XCTAssertEqual(
            gitResult.contentEvidenceID.sha256,
            "0f08e67415675fb5ce9738107e1886379" +
                "a4f6b48047b5f49a433e81b62a85e4e"
        )
        XCTAssertEqual(
            selfGuardResult.contentEvidenceID.sha256,
            "f0717cc182bf4728f91e6f8fe1b4c04c" +
                "8dddac31f9668d4829ac427a224cd20a"
        )
        XCTAssertEqual(
            gitResult.contentEvidenceID.sha256,
            Self.sha256Hex(Self.expectedPreimage(git))
        )
        XCTAssertEqual(
            selfGuardResult.contentEvidenceID.sha256,
            Self.sha256Hex(Self.expectedPreimage(selfGuard))
        )
        try Self.assertResult(gitResult, fixture: git)
        try Self.assertResult(selfGuardResult, fixture: selfGuard)
        XCTAssertEqual(Self.effects, .zero)
    }

    func testIndependentRoleVectorsFreezeExactIdentityBytes() {
        let git = Self.independentPreimage(role: .git)
        let selfGuard = Self.independentPreimage(role: .selfGuard)

        XCTAssertEqual(git.count, 503)
        XCTAssertEqual(
            Self.sha256Hex(git),
            "534995290724819a5f09eca4c81f6bef" +
                "c3b90679805b6bc10f20dae7d3fc8ea3"
        )
        XCTAssertEqual(selfGuard.count, 510)
        XCTAssertEqual(
            Self.sha256Hex(selfGuard),
            "ab6f1b97450fd79ee978a885af195311" +
                "af00dc9f6fe679935e2857c5f4b3da22"
        )
        XCTAssertEqual(Self.effects, .zero)
    }

    func testEveryReviewedIdentityFieldDriftChangesTheDigest() {
        let baseline = Self.independentPreimage(role: .git)
        let baselineID = Self.sha256Hex(baseline)
        let baselineLines = String(decoding: baseline, as: UTF8.self)
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        let replacements = [
            "artifact_role=self-guard",
            "manifest_sha256=" + String(repeating: "f", count: 64),
            "manifest_bytes=124",
            "root_executable_content_evidence_id=" +
                String(repeating: "e", count: 64),
            "dynamic_loader_content_evidence_id=" +
                String(repeating: "d", count: 64),
            "shared_cache_set_id=" + String(repeating: "c", count: 64),
            "file_image_member_count=2",
        ]

        for (index, replacement) in replacements.enumerated() {
            var lines = baselineLines
            lines[index + 1] = replacement
            let changed = Data(lines.joined(separator: "\n").utf8)
            XCTAssertNotEqual(Self.sha256Hex(changed), baselineID)
        }
        XCTAssertEqual(Self.effects, .zero)
    }

    func testMixedAndAllFileImageGraphsJoinPositiveCounts() throws {
        let mixed = try Self.fixture(
            role: .git,
            rootTargets: ["/usr/lib/libD2MixedFile.dylib"],
            specs: [
                .fileImage(
                    "/usr/lib/libD2MixedFile.dylib",
                    targets: ["/usr/lib/libD2MixedCache.dylib"]
                ),
                .sharedCache("/usr/lib/libD2MixedCache.dylib"),
            ]
        )
        let representative = try Self.allFileImageChain(count: 4)
        let upperBound = try Self.allFileImageChain(count: 256)

        let mixedResult = try Self.compare(mixed)
        let representativeResult = try Self.compare(representative)
        let upperBoundResult = try Self.compare(upperBound)

        XCTAssertEqual(mixedResult.memberCount, 2)
        XCTAssertEqual(mixedResult.fileImageMemberCount, 1)
        XCTAssertEqual(representativeResult.memberCount, 4)
        XCTAssertEqual(representativeResult.fileImageMemberCount, 4)
        XCTAssertEqual(upperBoundResult.memberCount, 256)
        XCTAssertEqual(upperBoundResult.fileImageMemberCount, 256)
        XCTAssertEqual(Self.effects, .zero)
    }

    func testReferenceRootLoaderAndGraphFailuresKeepReviewedOrder()
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
            expected: .expectationReference(
                .expectationReanchor(.documentDigestMismatch)
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
            expected: .expectationReference(
                .expectationEvidenceMismatch
            )
        )
        Self.assertFailure(
            fixture: git,
            root: selfGuard.root,
            expected: .rootRole(expected: .git, actual: .selfGuard)
        )

        let foreignRootReference = try Self.reference(
            for: git,
            rootID: String(repeating: "e", count: 64)
        )
        Self.assertFailure(
            fixture: git,
            expectationReference: foreignRootReference,
            currentAnchor:
                foreignRootReference.anchoredExpectation.trustAnchor,
            expected: .rootManifestIDMismatch
        )
        let foreignLoaderReference = try Self.reference(
            for: git,
            loaderID: String(repeating: "d", count: 64)
        )
        Self.assertFailure(
            fixture: git,
            expectationReference: foreignLoaderReference,
            currentAnchor:
                foreignLoaderReference.anchoredExpectation.trustAnchor,
            expected: .dynamicLoaderManifestIDMismatch
        )
        Self.assertFailure(
            fixture: git,
            memberInventories: [],
            expected: .graph(.memberInventoryCountMismatch)
        )
        let foreignGraph = try Self.fixture(
            role: .git,
            rootTargets: ["/usr/lib/libD2C2Foreign.dylib"],
            specs: [.fileImage("/usr/lib/libD2C2Foreign.dylib")]
        )
        Self.assertFailure(
            fixture: git,
            graphComparison: foreignGraph.graph,
            expected: .graphComparisonMismatch
        )
        XCTAssertEqual(Self.effects, .zero)
    }

    func testCountAndRowFailuresPrecedeSectionAndIdentity()
        throws
    {
        let twoFiles = try Self.fixture(
            role: .git,
            rootTargets: ["/usr/lib/libD2CountA.dylib"],
            specs: [
                .fileImage(
                    "/usr/lib/libD2CountA.dylib",
                    targets: ["/usr/lib/libD2CountB.dylib"]
                ),
                .fileImage("/usr/lib/libD2CountB.dylib"),
            ]
        )
        let rootEdge = try XCTUnwrap(
            twoFiles.edges.first {
                $0.parentContentEvidenceID ==
                    twoFiles.root.contentEvidenceID.sha256
            }
        )
        let rootMember = try XCTUnwrap(
            twoFiles.members.first {
                $0.contentEvidenceID ==
                    rootEdge.resolvedContentEvidenceID
            }
        )
        let oneMemberReference = try Self.reference(
            for: twoFiles,
            memberFields: Self.memberFields([rootMember]),
            edgeFields: Self.edgeFields([rootEdge])
        )
        Self.assertFailure(
            fixture: twoFiles,
            expectationReference: oneMemberReference,
            currentAnchor:
                oneMemberReference.anchoredExpectation.trustAnchor,
            expected: .graphExpectation(.memberCount)
        )
        let oneEdgeReference = try Self.reference(
            for: twoFiles,
            memberFields: Self.memberFields(twoFiles.members),
            edgeFields: Self.edgeFields(Array(twoFiles.edges.dropLast()))
        )
        Self.assertFailure(
            fixture: twoFiles,
            expectationReference: oneEdgeReference,
            currentAnchor:
                oneEdgeReference.anchoredExpectation.trustAnchor,
            expected: .graphExpectation(.edgeCount)
        )

        let mixed = try Self.fixture(
            role: .git,
            rootTargets: ["/usr/lib/libD2CountFile.dylib"],
            specs: [
                .fileImage(
                    "/usr/lib/libD2CountFile.dylib",
                    targets: ["/usr/lib/libD2CountCache.dylib"]
                ),
                .sharedCache("/usr/lib/libD2CountCache.dylib"),
            ]
        )
        var bothFileFields = Self.memberFields(mixed.members)
        let sharedIndex = try XCTUnwrap(
            mixed.members.firstIndex { $0.storage == .sharedCache }
        )
        bothFileFields[sharedIndex] = Self.memberField(
            bothFileFields[sharedIndex],
            storage: .file,
            primaryCodeDirectoryBlobSHA256:
                String(repeating: "f", count: 64)
        )
        let declaredCountReference = try Self.reference(
            for: mixed,
            memberFields: bothFileFields
        )
        Self.assertFailure(
            fixture: mixed,
            expectationReference: declaredCountReference,
            currentAnchor:
                declaredCountReference.anchoredExpectation.trustAnchor,
            expected: .graphExpectation(.declaredFileImageMemberCount)
        )

        let canonical = try Self.fixture(role: .git)
        var changedUUID = Self.memberFields(canonical.members)
        changedUUID[0] = Self.memberField(
            changedUUID[0],
            machOUUID: String(repeating: "f", count: 32)
        )
        let rowReference = try Self.reference(
            for: canonical,
            memberFields: changedUUID
        )
        Self.assertFailure(
            fixture: canonical,
            expectationReference: rowReference,
            currentAnchor: rowReference.anchoredExpectation.trustAnchor,
            expected: .memberExpectation(
                index: 0,
                field: .machOUUID
            )
        )

        let canonicalEdge = try XCTUnwrap(canonical.edges.first)
        let longerOrdinal = RuntimeClosureExpectationEdgeFields(
            parentContentEvidenceID:
                canonicalEdge.parentContentEvidenceID,
            loadCommandOrdinal: 10,
            kind: canonicalEdge.kind,
            installName: canonicalEdge.installName,
            decodedInstallName: canonicalEdge.decodedInstallName,
            resolvedContentEvidenceID:
                canonicalEdge.resolvedContentEvidenceID
        )
        let longerSectionReference = try Self.reference(
            for: canonical,
            edgeFields: [longerOrdinal]
        )
        Self.assertFailure(
            fixture: canonical,
            expectationReference: longerSectionReference,
            currentAnchor:
                longerSectionReference.anchoredExpectation.trustAnchor,
            expected: .graphExpectation(.canonicalSectionRange)
        )

        let changedSingleDigitOrdinal: UInt64 =
            canonicalEdge.loadCommandOrdinal == 1 ? 2 : 1
        let sameLengthOrdinal = RuntimeClosureExpectationEdgeFields(
            parentContentEvidenceID:
                canonicalEdge.parentContentEvidenceID,
            loadCommandOrdinal: changedSingleDigitOrdinal,
            kind: canonicalEdge.kind,
            installName: canonicalEdge.installName,
            decodedInstallName: canonicalEdge.decodedInstallName,
            resolvedContentEvidenceID:
                canonicalEdge.resolvedContentEvidenceID
        )
        let changedSectionReference = try Self.reference(
            for: canonical,
            edgeFields: [sameLengthOrdinal]
        )
        Self.assertFailure(
            fixture: canonical,
            expectationReference: changedSectionReference,
            currentAnchor:
                changedSectionReference.anchoredExpectation.trustAnchor,
            expected: .graphExpectation(.canonicalSectionBytes)
        )
        XCTAssertEqual(Self.effects, .zero)
    }

    func testEveryConstructibleExpectationMemberFieldDriftsIndependently()
        throws
    {
        let fixture = try Self.fixture(role: .git)
        let member = try XCTUnwrap(
            Self.memberFields(fixture.members).first
        )
        let edge = try XCTUnwrap(
            Self.edgeFields(fixture.edges).first
        )

        let foreignContentEvidenceID = String(repeating: "e", count: 64)
        try Self.assertMemberExpectationFailure(
            fixture: fixture,
            member: Self.memberField(
                member,
                contentEvidenceID: foreignContentEvidenceID
            ),
            edge: Self.edgeField(
                edge,
                resolvedContentEvidenceID: foreignContentEvidenceID
            ),
            field: .contentEvidenceID
        )

        let longerName = FMAFileImageFixture.installName(
            "/usr/lib/libD2C2FixtureLong.dylib"
        )
        try Self.assertMemberExpectationFailure(
            fixture: fixture,
            member: Self.memberField(
                member,
                installName: longerName,
                decodedInstallName:
                    Data("/usr/lib/libD2C2FixtureLong.dylib".utf8)
            ),
            edge: Self.edgeField(
                edge,
                installName: longerName,
                decodedInstallName:
                    Data("/usr/lib/libD2C2FixtureLong.dylib".utf8)
            ),
            field: .installNameBytes
        )

        let sameLengthNameString =
            "/usr/lib/libD2C2Fixture.dylia"
        XCTAssertEqual(
            sameLengthNameString.utf8.count,
            Int(member.installName.bytes)
        )
        let sameLengthName = FMAFileImageFixture.installName(
            sameLengthNameString
        )
        try Self.assertMemberExpectationFailure(
            fixture: fixture,
            member: Self.memberField(
                member,
                installName: sameLengthName,
                decodedInstallName: Data(sameLengthNameString.utf8)
            ),
            edge: Self.edgeField(
                edge,
                installName: sameLengthName,
                decodedInstallName: Data(sameLengthNameString.utf8)
            ),
            field: .installNameBase64URL
        )

        try Self.assertMemberExpectationFailure(
            fixture: fixture,
            member: Self.memberField(
                member,
                machOUUID: String(repeating: "f", count: 32)
            ),
            edge: edge,
            field: .machOUUID
        )
        try Self.assertMemberExpectationFailure(
            fixture: fixture,
            member: Self.memberField(
                member,
                primaryCodeDirectoryBlobSHA256:
                    String(repeating: "f", count: 64)
            ),
            edge: edge,
            field: .primaryCodeDirectoryBlobSHA256
        )
        try Self.assertMemberExpectationFailure(
            fixture: fixture,
            member: Self.memberField(
                member,
                loadCommandsSHA256: String(repeating: "e", count: 64)
            ),
            edge: edge,
            field: .loadCommandsSHA256
        )
        XCTAssertEqual(Self.effects, .zero)
    }

    func testGenericFileRefusesThroughFMBeforeLocalCountWork() throws {
        let fixture = try Self.fixture(role: .git)
        let fileEvidence = try ExecutableContentIdentityVerifier.derive(
            artifactRole: .fileImage,
            comparison: fixture.root.comparison
        )
        let member = try SyntheticRuntimeClosureRecordSchemaVerifier.member(
            index: 0,
            source: .file(fileEvidence),
            installName: FMAFileImageFixture.installName(Self.memberName)
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
        let collection = try
            SyntheticRuntimeClosureRecordCollectionVerifier.derive(
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

    func testAllSharedCacheGraphRefusesAtD2C1() {
        XCTAssertThrowsError(
            try Self.fixture(
                role: .git,
                rootTargets: [Self.memberName],
                specs: [.sharedCache(Self.memberName)]
            )
        ) {
            XCTAssertEqual(
                $0 as? FileImageRuntimeClosureExpectationFailure,
                .declaredFileImageMemberCount
            )
        }
        XCTAssertEqual(Self.effects, .zero)
    }

    func testSharedCacheEvidenceMustMatchTheAnchoredCacheSet() throws {
        let specs: [MemberSpec] = [
            .fileImage(
                "/usr/lib/libD2CacheFile.dylib",
                targets: ["/usr/lib/libD2CacheMember.dylib"]
            ),
            .sharedCache("/usr/lib/libD2CacheMember.dylib"),
        ]
        let actual = try Self.fixture(
            role: .git,
            rootTargets: ["/usr/lib/libD2CacheFile.dylib"],
            specs: specs,
            cacheSeed: "1"
        )
        let foreign = try Self.fixture(
            role: .git,
            rootTargets: ["/usr/lib/libD2CacheFile.dylib"],
            specs: specs,
            cacheSeed: "2"
        )
        XCTAssertEqual(
            actual.members.map(\.storage),
            foreign.members.map(\.storage)
        )
        let foreignReference = try Self.reference(
            for: actual,
            cacheSet: foreign.cacheSet,
            memberFields: Self.memberFields(foreign.members),
            edgeFields: Self.edgeFields(foreign.edges)
        )
        let sharedIndex = try XCTUnwrap(
            actual.members.firstIndex { $0.storage == .sharedCache }
        )

        Self.assertFailure(
            fixture: actual,
            expectationReference: foreignReference,
            currentAnchor:
                foreignReference.anchoredExpectation.trustAnchor,
            expected: .memberCacheSetMismatch(index: sharedIndex)
        )
        XCTAssertEqual(Self.effects, .zero)
    }

    func testNonzeroDataIndexAndCallerMutationPreserveCompactResult()
        throws
    {
        var fixture = try Self.fixture(role: .git)
        var backing = Data([0])
        backing.append(fixture.expectationFile.bytes)
        backing.append(0)
        let slice = backing[1..<(1 + fixture.expectationFile.bytes.count)]
        XCTAssertEqual(slice.startIndex, 1)
        let slicedFile = Self.admittedFile(slice)
        let slicedAnchor = Self.anchor(
            for: slicedFile,
            role: fixture.role
        )
        let slicedExpectation = try RuntimeClosureExpectationVerifier
            .anchor(
                expectationFile: slicedFile,
                trustAnchor: slicedAnchor
            )
        let slicedReference = try
            FileImageRuntimeClosureExpectationVerifier.reference(
                anchoredExpectation: slicedExpectation,
                currentExpectationAnchor: slicedAnchor
            )
        let result = try Self.compare(
            fixture,
            expectationReference: slicedReference,
            currentAnchor: slicedAnchor
        )
        let retained = result

        backing[slice.startIndex] = 0xff
        fixture.members.removeAll()
        fixture.edges.removeAll()
        fixture.memberInventories.removeAll()

        XCTAssertEqual(result, retained)
        XCTAssertEqual(result.memberCount, 1)
        XCTAssertEqual(result.fileImageMemberCount, 1)
        XCTAssertEqual(Self.effects, .zero)
    }
}

private extension FileImageExecutionIdentityComparisonTests {
    struct Effects: Equatable {
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

    struct MemberSpec {
        enum Kind {
            case sharedCache
            case fileImage
        }

        let name: String
        let kind: Kind
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

    struct EdgeInput {
        let parent: SyntheticRuntimeClosureEdgeParent
        let parentID: String
        let ordinal: UInt64
        let kind: SyntheticRuntimeClosureEdgeKind
        let resolved: SyntheticRuntimeClosureMemberRecordComparison
    }

    struct Fixture {
        let role: RuntimeClosureExpectationArtifactRole
        let root: ExecutableContentIdentityEvidence
        let loader: DynamicLoaderContentIdentityEvidence
        let cacheSet: SyntheticSharedCacheSetIdentityEvidence
        var members: [SyntheticRuntimeClosureMemberRecordComparison]
        var edges: [SyntheticRuntimeClosureEdgeRecordComparison]
        let collection: SyntheticRuntimeClosureRecordCollectionComparison
        let rootInventory:
            SyntheticAcceptedDependencyCommandInventoryComparison
        var memberInventories:
            [SyntheticAcceptedDependencyCommandInventoryComparison]
        let graph: SyntheticFileImageRuntimeClosureGraphComparison
        let expectationFile: AdmittedFile
        let anchor: RuntimeClosureExpectationTrustAnchor
        let expectation: AnchoredRuntimeClosureExpectationDocument
        let reference: FileImageRuntimeClosureExpectationReference
    }

    struct P3TestKey {
        let privateKey: Curve25519.Signing.PrivateKey  // gitleaks:allow
        let publicKeyBase64: String
        let keyID: String
    }

    struct P3Fixture {
        let toolReference: SignedClaimGitToolPolicyV2Reference
        let toolAnchor: GitToolPolicyV2TrustAnchor
        let denial: AnchoredGitRuntimePolicyDenialV2Document
        let denialAnchor: GitRuntimePolicyDenialV2TrustAnchor
        let gitExecutable: ExecutableContentExpectationComparison
        let gitExecutableAnchor: ExecutableIdentityExpectationTrustAnchor
        let selfGuardExecutable: ExecutableContentExpectationComparison
        let selfGuardExecutableAnchor:
            ExecutableIdentityExpectationTrustAnchor
        let gitClosure: Fixture
        let gitClosureComparison:
            FileImageRuntimeClosureContentExpectationComparison
        let selfGuardClosure: Fixture
        let selfGuardClosureComparison:
            FileImageRuntimeClosureContentExpectationComparison
    }

    static let effects = Effects()
    static let memberName = "/usr/lib/libD2C2Fixture.dylib"
    static let p3CommonTime: UInt64 = 2_000_000_000

    static func p3Fixture(
        selfGuardCacheSeed: Character = "1",
        selfGuardLoaderUUIDSeed: UInt8 = 0x20,
        toolExecutableSHA256: String? = nil,
        toolExecutableBytes: UInt64? = nil,
        toolExecutableMachOUUID: String? = nil,
        toolExecutableCodeDirectorySHA256: String? = nil,
        toolGitClosureManifestSHA256: String? = nil,
        toolGitClosureManifestBytes: UInt64? = nil,
        toolGitClosureContentEvidenceID: String? = nil,
        denialSelfGuardSHA256: String? = nil,
        denialSelfGuardBytes: UInt64? = nil,
        denialSelfGuardMachOUUID: String? = nil,
        denialSelfGuardCodeDirectorySHA256: String? = nil,
        denialSelfGuardClosureContentEvidenceID: String? = nil,
        denialGitClosureContentEvidenceID: String? = nil,
        denialLimits: GitToolPolicyResourceLimits? = nil,
        toolRuntimePolicySHA256: String? = nil
    ) throws -> P3Fixture {
        let git = try fixture(role: .git)
        let selfGuard = try fixture(
            role: .selfGuard,
            cacheSeed: selfGuardCacheSeed,
            loaderUUIDSeed: selfGuardLoaderUUIDSeed
        )
        let gitClosureComparison = try compare(git)
        let selfGuardClosureComparison = try compare(selfGuard)
        let (gitExecutable, gitExecutableAnchor) =
            try p3ExecutableComparison(
                evidence: git.root,
                role: .git
            )
        let (selfGuardExecutable, selfGuardExecutableAnchor) =
            try p3ExecutableComparison(
                evidence: selfGuard.root,
                role: .selfGuard
            )
        let gitCodeDirectory = try XCTUnwrap(
            gitExecutable.expectation.fields.codeDirectories.first
        )
        let selfGuardCodeDirectory = try XCTUnwrap(
            selfGuardExecutable.expectation.fields.codeDirectories.first
        )
        let limits = GitToolPolicyVerifier.phase1ResourceCeilings
        let denialFields = GitRuntimePolicyDenialV2Fields(
            policyGeneration: 7,
            validFromUnixSeconds: 1_900_000_000,
            validUntilUnixSeconds: 2_100_000_000,
            selfGuardSHA256:
                denialSelfGuardSHA256 ??
                selfGuardExecutable.expectation.fields.fileSHA256,
            selfGuardBytes:
                denialSelfGuardBytes ??
                selfGuardExecutable.expectation.fields.fileBytes,
            selfGuardMachOUUID:
                denialSelfGuardMachOUUID ??
                selfGuardExecutable.expectation.fields.machOUUID,
            selfGuardCodeDirectorySHA256:
                denialSelfGuardCodeDirectorySHA256 ??
                selfGuardCodeDirectory.blobSHA256,
            selfGuardFileImageRuntimeClosureContentEvidenceID:
                denialSelfGuardClosureContentEvidenceID ??
                selfGuardClosureComparison.contentEvidenceID.sha256,
            gitFileImageRuntimeClosureContentEvidenceID:
                denialGitClosureContentEvidenceID ??
                gitClosureComparison.contentEvidenceID.sha256,
            limits: denialLimits ?? limits,
            terminationGraceMilliseconds: 2_000,
            reapTimeoutMilliseconds: 5_000
        )
        let denialFile = admittedFile(
            try GitRuntimePolicyDenialV2Verifier.policyBytes(
                fields: denialFields
            )
        )
        let denialAnchor = GitRuntimePolicyDenialV2TrustAnchor(
            expectedCurrentPolicySHA256: denialFile.sha256,
            expectedCurrentPolicyBytes: UInt64(denialFile.bytes.count),
            minimumPolicyGeneration: 7,
            verificationUnixSeconds: p3CommonTime
        )
        let denial = try GitRuntimePolicyDenialV2Verifier.anchor(
            policyFile: denialFile,
            trustAnchor: denialAnchor
        )
        let toolFields = GitToolPolicyV2Fields(
            policyGeneration: 7,
            validFromUnixSeconds: 1_900_000_000,
            validUntilUnixSeconds: 2_100_000_000,
            executableSHA256:
                toolExecutableSHA256 ??
                gitExecutable.expectation.fields.fileSHA256,
            executableBytes:
                toolExecutableBytes ??
                gitExecutable.expectation.fields.fileBytes,
            executableMachOUUID:
                toolExecutableMachOUUID ??
                gitExecutable.expectation.fields.machOUUID,
            executableCodeDirectorySHA256:
                toolExecutableCodeDirectorySHA256 ??
                gitCodeDirectory.blobSHA256,
            gitRuntimeClosureManifestSHA256:
                toolGitClosureManifestSHA256 ??
                git.expectation.documentSHA256,
            gitRuntimeClosureManifestBytes:
                toolGitClosureManifestBytes ??
                git.expectation.documentBytes,
            gitFileImageRuntimeClosureContentEvidenceID:
                toolGitClosureContentEvidenceID ??
                gitClosureComparison.contentEvidenceID.sha256,
            runtimePolicySHA256:
                toolRuntimePolicySHA256 ?? denial.policySHA256,
            limits: limits
        )
        let toolFile = admittedFile(
            try GitToolPolicyV2Verifier.policyBytes(fields: toolFields)
        )
        let toolAnchor = GitToolPolicyV2TrustAnchor(
            expectedCurrentPolicySHA256: toolFile.sha256,
            expectedCurrentPolicyBytes: UInt64(toolFile.bytes.count),
            minimumPolicyGeneration: 7,
            verificationUnixSeconds: p3CommonTime
        )
        let tool = try GitToolPolicyV2Verifier.anchor(
            policyFile: toolFile,
            trustAnchor: toolAnchor
        )
        let claim = try p3SignedClaim(
            toolManifestSHA256: tool.policySHA256,
            toolManifestBytes: tool.policyBytes,
            runtimePolicySHA256: tool.fields.runtimePolicySHA256,
            signingDynamically:
                selfGuardCacheSeed != "1" ||
                selfGuardLoaderUUIDSeed != 0x20 ||
                toolExecutableSHA256 != nil ||
                toolExecutableBytes != nil ||
                toolExecutableMachOUUID != nil ||
                toolExecutableCodeDirectorySHA256 != nil ||
                toolGitClosureManifestSHA256 != nil ||
                toolGitClosureManifestBytes != nil ||
                toolGitClosureContentEvidenceID != nil ||
                denialSelfGuardSHA256 != nil ||
                denialSelfGuardBytes != nil ||
                denialSelfGuardMachOUUID != nil ||
                denialSelfGuardCodeDirectorySHA256 != nil ||
                denialSelfGuardClosureContentEvidenceID != nil ||
                denialGitClosureContentEvidenceID != nil ||
                denialLimits != nil ||
                toolRuntimePolicySHA256 != nil
        )
        let toolReference = try GitToolPolicyV2Verifier.reference(
            signedClaim: claim,
            policyDocument: tool
        )
        return P3Fixture(
            toolReference: toolReference,
            toolAnchor: toolAnchor,
            denial: denial,
            denialAnchor: denialAnchor,
            gitExecutable: gitExecutable,
            gitExecutableAnchor: gitExecutableAnchor,
            selfGuardExecutable: selfGuardExecutable,
            selfGuardExecutableAnchor: selfGuardExecutableAnchor,
            gitClosure: git,
            gitClosureComparison: gitClosureComparison,
            selfGuardClosure: selfGuard,
            selfGuardClosureComparison: selfGuardClosureComparison
        )
    }

    static func p3Compare(
        _ fixture: P3Fixture,
        toolReference: SignedClaimGitToolPolicyV2Reference? = nil,
        toolAnchor: GitToolPolicyV2TrustAnchor? = nil,
        denial: AnchoredGitRuntimePolicyDenialV2Document? = nil,
        denialAnchor: GitRuntimePolicyDenialV2TrustAnchor? = nil,
        gitExecutable: ExecutableContentExpectationComparison? = nil,
        gitExecutableAnchor: ExecutableIdentityExpectationTrustAnchor? = nil,
        selfGuardExecutable: ExecutableContentExpectationComparison? = nil,
        selfGuardExecutableAnchor:
            ExecutableIdentityExpectationTrustAnchor? = nil,
        gitClosureExpectation:
            AnchoredRuntimeClosureExpectationDocument? = nil,
        gitClosureAnchor: RuntimeClosureExpectationTrustAnchor? = nil,
        gitClosureReference:
            FileImageRuntimeClosureExpectationReference? = nil,
        gitClosureComparison:
            FileImageRuntimeClosureContentExpectationComparison? = nil,
        selfGuardClosureExpectation:
            AnchoredRuntimeClosureExpectationDocument? = nil,
        selfGuardClosureAnchor:
            RuntimeClosureExpectationTrustAnchor? = nil,
        selfGuardClosureReference:
            FileImageRuntimeClosureExpectationReference? = nil,
        selfGuardClosureComparison:
            FileImageRuntimeClosureContentExpectationComparison? = nil
    ) throws -> FileImageExecutionIdentityComparisonEvidence {
        try FileImageExecutionIdentityComparisonVerifier.compare(
            signedClaimToolPolicyReference:
                toolReference ?? fixture.toolReference,
            currentGitToolPolicyAnchor:
                toolAnchor ?? fixture.toolAnchor,
            runtimeDenialDocument: denial ?? fixture.denial,
            currentRuntimeDenialAnchor:
                denialAnchor ?? fixture.denialAnchor,
            gitExecutableComparison:
                gitExecutable ?? fixture.gitExecutable,
            currentGitExecutableAnchor:
                gitExecutableAnchor ?? fixture.gitExecutableAnchor,
            selfGuardExecutableComparison:
                selfGuardExecutable ?? fixture.selfGuardExecutable,
            currentSelfGuardExecutableAnchor:
                selfGuardExecutableAnchor ??
                fixture.selfGuardExecutableAnchor,
            gitClosureExpectation:
                gitClosureExpectation ?? fixture.gitClosure.expectation,
            currentGitClosureAnchor:
                gitClosureAnchor ?? fixture.gitClosure.anchor,
            gitClosureReference:
                gitClosureReference ?? fixture.gitClosure.reference,
            gitClosureComparison:
                gitClosureComparison ?? fixture.gitClosureComparison,
            selfGuardClosureExpectation:
                selfGuardClosureExpectation ??
                fixture.selfGuardClosure.expectation,
            currentSelfGuardClosureAnchor:
                selfGuardClosureAnchor ?? fixture.selfGuardClosure.anchor,
            selfGuardClosureReference:
                selfGuardClosureReference ??
                fixture.selfGuardClosure.reference,
            selfGuardClosureComparison:
                selfGuardClosureComparison ??
                fixture.selfGuardClosureComparison
        )
    }

    static func assertP3Failure(
        _ fixture: P3Fixture,
        toolReference: SignedClaimGitToolPolicyV2Reference? = nil,
        toolAnchor: GitToolPolicyV2TrustAnchor? = nil,
        denial: AnchoredGitRuntimePolicyDenialV2Document? = nil,
        denialAnchor: GitRuntimePolicyDenialV2TrustAnchor? = nil,
        gitExecutable: ExecutableContentExpectationComparison? = nil,
        gitExecutableAnchor: ExecutableIdentityExpectationTrustAnchor? = nil,
        selfGuardExecutable: ExecutableContentExpectationComparison? = nil,
        selfGuardExecutableAnchor:
            ExecutableIdentityExpectationTrustAnchor? = nil,
        gitClosureExpectation:
            AnchoredRuntimeClosureExpectationDocument? = nil,
        gitClosureAnchor: RuntimeClosureExpectationTrustAnchor? = nil,
        gitClosureReference:
            FileImageRuntimeClosureExpectationReference? = nil,
        gitClosureComparison:
            FileImageRuntimeClosureContentExpectationComparison? = nil,
        selfGuardClosureExpectation:
            AnchoredRuntimeClosureExpectationDocument? = nil,
        selfGuardClosureAnchor:
            RuntimeClosureExpectationTrustAnchor? = nil,
        selfGuardClosureReference:
            FileImageRuntimeClosureExpectationReference? = nil,
        selfGuardClosureComparison:
            FileImageRuntimeClosureContentExpectationComparison? = nil,
        expected: FileImageExecutionIdentityComparisonFailure,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(
            try p3Compare(
                fixture,
                toolReference: toolReference,
                toolAnchor: toolAnchor,
                denial: denial,
                denialAnchor: denialAnchor,
                gitExecutable: gitExecutable,
                gitExecutableAnchor: gitExecutableAnchor,
                selfGuardExecutable: selfGuardExecutable,
                selfGuardExecutableAnchor: selfGuardExecutableAnchor,
                gitClosureExpectation: gitClosureExpectation,
                gitClosureAnchor: gitClosureAnchor,
                gitClosureReference: gitClosureReference,
                gitClosureComparison: gitClosureComparison,
                selfGuardClosureExpectation:
                    selfGuardClosureExpectation,
                selfGuardClosureAnchor: selfGuardClosureAnchor,
                selfGuardClosureReference: selfGuardClosureReference,
                selfGuardClosureComparison: selfGuardClosureComparison
            ),
            file: file,
            line: line
        ) { error in
            XCTAssertEqual(
                error as? FileImageExecutionIdentityComparisonFailure,
                expected,
                file: file,
                line: line
            )
        }
    }

    static let p3ResultScalarAllowlist: Set<String> = [
        "runClaimID",
        "anchorContextID",
        "gitToolPolicySHA256",
        "runtimeDenialPolicySHA256",
        "gitExecutableExpectationSHA256",
        "gitExecutableContentEvidenceID",
        "selfGuardExecutableExpectationSHA256",
        "selfGuardExecutableContentEvidenceID",
        "gitClosureManifestSHA256",
        "gitFileImageRuntimeClosureContentEvidenceID",
        "selfGuardClosureManifestSHA256",
        "selfGuardFileImageRuntimeClosureContentEvidenceID",
        "comparisonSHA256",
        "provesSignedClaimToolPolicyMatch",
        "provesRuntimeDenialMatch",
        "provesExecutableIdentityMatch",
        "provesFileImageRuntimeClosureIdentityMatch",
        "provesSingleCurrentAnchorContext",
        "runtimeDecision",
        "canExecute",
        "canSpawn",
        "canAccessNetwork",
        "canConsumePack",
        "canMutateFileSystem",
        "canImportGitObjects",
        "canBuild",
        "canLoadModel",
        "canReserveOutput",
        "canPublish",
    ]

    static func assertP3NoAuthority(
        _ value: FileImageExecutionIdentityComparisonEvidence,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
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

    static func p3ToolAnchor(
        _ fixture: P3Fixture,
        digest: String? = nil,
        bytes: UInt64? = nil,
        generation: UInt64? = nil,
        time: UInt64? = nil
    ) -> GitToolPolicyV2TrustAnchor {
        GitToolPolicyV2TrustAnchor(
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

    static func p3DenialAnchor(
        _ fixture: P3Fixture,
        digest: String? = nil,
        bytes: UInt64? = nil,
        generation: UInt64? = nil,
        time: UInt64? = nil
    ) -> GitRuntimePolicyDenialV2TrustAnchor {
        GitRuntimePolicyDenialV2TrustAnchor(
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

    static func p3ExecutableAnchor(
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

    static func p3ClosureAnchor(
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

    static func assertP3ToolAnchorScalarFailures(
        _ fixture: P3Fixture,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let invalidDigest = String(repeating: "A", count: 64)
        assertP3Failure(
            fixture,
            toolAnchor: p3ToolAnchor(fixture, digest: invalidDigest),
            expected: .invalidAnchorScalar(
                slot: .gitToolPolicy,
                field: .digest
            ),
            file: file,
            line: line
        )
        assertP3Failure(
            fixture,
            toolAnchor: p3ToolAnchor(fixture, bytes: 0),
            expected: .invalidAnchorScalar(
                slot: .gitToolPolicy,
                field: .byteCount
            ),
            file: file,
            line: line
        )
        assertP3Failure(
            fixture,
            toolAnchor: p3ToolAnchor(fixture, generation: 0),
            expected: .invalidAnchorScalar(
                slot: .gitToolPolicy,
                field: .minimumGeneration
            ),
            file: file,
            line: line
        )
    }

    static func assertP3DenialAnchorScalarFailures(
        _ fixture: P3Fixture,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let invalidDigest = String(repeating: "A", count: 64)
        assertP3Failure(
            fixture,
            denialAnchor: p3DenialAnchor(fixture, digest: invalidDigest),
            expected: .invalidAnchorScalar(
                slot: .runtimeDenial,
                field: .digest
            ),
            file: file,
            line: line
        )
        assertP3Failure(
            fixture,
            denialAnchor: p3DenialAnchor(fixture, bytes: 0),
            expected: .invalidAnchorScalar(
                slot: .runtimeDenial,
                field: .byteCount
            ),
            file: file,
            line: line
        )
        assertP3Failure(
            fixture,
            denialAnchor: p3DenialAnchor(fixture, generation: 0),
            expected: .invalidAnchorScalar(
                slot: .runtimeDenial,
                field: .minimumGeneration
            ),
            file: file,
            line: line
        )
    }

    static func assertP3ExecutableAnchorScalarFailures(
        _ fixture: P3Fixture,
        original: ExecutableIdentityExpectationTrustAnchor,
        slot: FileImageExecutionIdentityComparisonFailure.AnchorSlot,
        git: Bool,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let invalidDigest = String(repeating: "A", count: 64)
        let invoke: (
            ExecutableIdentityExpectationTrustAnchor,
            FileImageExecutionIdentityComparisonFailure
        ) -> Void = { anchor, expected in
            if git {
                assertP3Failure(
                    fixture,
                    gitExecutableAnchor: anchor,
                    expected: expected,
                    file: file,
                    line: line
                )
            } else {
                assertP3Failure(
                    fixture,
                    selfGuardExecutableAnchor: anchor,
                    expected: expected,
                    file: file,
                    line: line
                )
            }
        }
        invoke(
            p3ExecutableAnchor(original, digest: invalidDigest),
            .invalidAnchorScalar(slot: slot, field: .digest)
        )
        invoke(
            p3ExecutableAnchor(original, bytes: 0),
            .invalidAnchorScalar(slot: slot, field: .byteCount)
        )
        invoke(
            p3ExecutableAnchor(original, generation: 0),
            .invalidAnchorScalar(slot: slot, field: .minimumGeneration)
        )
    }

    static func assertP3ClosureAnchorScalarFailures(
        _ fixture: P3Fixture,
        original: RuntimeClosureExpectationTrustAnchor,
        slot: FileImageExecutionIdentityComparisonFailure.AnchorSlot,
        git: Bool,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let invalidDigest = String(repeating: "A", count: 64)
        let invoke: (
            RuntimeClosureExpectationTrustAnchor,
            FileImageExecutionIdentityComparisonFailure
        ) -> Void = { anchor, expected in
            if git {
                assertP3Failure(
                    fixture,
                    gitClosureAnchor: anchor,
                    expected: expected,
                    file: file,
                    line: line
                )
            } else {
                assertP3Failure(
                    fixture,
                    selfGuardClosureAnchor: anchor,
                    expected: expected,
                    file: file,
                    line: line
                )
            }
        }
        invoke(
            p3ClosureAnchor(original, digest: invalidDigest),
            .invalidAnchorScalar(slot: slot, field: .digest)
        )
        invoke(
            p3ClosureAnchor(original, bytes: 0),
            .invalidAnchorScalar(slot: slot, field: .byteCount)
        )
        invoke(
            p3ClosureAnchor(original, generation: 0),
            .invalidAnchorScalar(slot: slot, field: .minimumGeneration)
        )
    }

    static func p3Limits(maxOpenFiles: UInt64)
        -> GitToolPolicyResourceLimits
    {
        let limits = GitToolPolicyVerifier.phase1ResourceCeilings
        return GitToolPolicyResourceLimits(
            maxPackBytes: limits.maxPackBytes,
            maxPackObjects: limits.maxPackObjects,
            maxCommitBytes: limits.maxCommitBytes,
            maxSingleInflatedObjectBytes:
                limits.maxSingleInflatedObjectBytes,
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

    static func fixture(
        role: RuntimeClosureExpectationArtifactRole,
        rootTargets: [String]? = nil,
        specs: [MemberSpec]? = nil,
        cacheSeed: Character = "1",
        loaderUUIDSeed: UInt8 = 0x20
    ) throws -> Fixture {
        let resolvedSpecs = specs ?? [.fileImage(memberName)]
        let resolvedRootTargets = rootTargets ?? [resolvedSpecs[0].name]
        let cacheSet = try SyntheticSharedCacheSetIdentityVerifier.derive(
            records: [
                SyntheticSharedCacheFileRecord(
                    suffixBytes: 0,
                    suffixBase64URL: "",
                    fileSHA256: String(
                        repeating: cacheSeed,
                        count: 64
                    ),
                    fileBytes: 4_096,
                    headerUUID: String(
                        repeating: cacheSeed,
                        count: 32
                    )
                )
            ]
        )
        var inputs: [MemberInput] = []
        for (index, spec) in resolvedSpecs.enumerated() {
            let commands = spec.targets.map {
                FMAFileImageFixture.dylibCommand(
                    command: FMAFileImageFixture.lcLoadDylib,
                    name: $0
                )
            }
            switch spec.kind {
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
            case .sharedCache:
                let snapshot = try sharedCacheSnapshot(
                    name: spec.name,
                    commands: commands,
                    cacheSet: cacheSet,
                    uuidSeed: UInt8(truncatingIfNeeded: 0x70 + index)
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
                installName: FMAFileImageFixture.installName(input.name)
            )
        }
        let inventoryByID = Dictionary(
            uniqueKeysWithValues: inputs.map {
                ($0.contentEvidenceID, $0.inventory)
            }
        )
        let memberInventories = members.map {
            inventoryByID[$0.contentEvidenceID]!
        }
        let memberByName = Dictionary(
            uniqueKeysWithValues: members.map {
                (
                    String(
                        decoding: $0.decodedInstallName,
                        as: UTF8.self
                    ),
                    $0
                )
            }
        )
        let root = try SyntheticRuntimeClosureGraphTests.fmbRootEvidence(
            role: executableRole(role),
            loadCommands: resolvedRootTargets.map {
                FMAFileImageFixture.dylibCommand(
                    command: FMAFileImageFixture.lcLoadDylib,
                    name: $0
                )
            }
        )
        let rootInventory = try
            SyntheticAcceptedDependencyCommandInventoryVerifier.root(root)
        var edgeInputs: [EdgeInput] = []
        for entry in rootInventory.entries {
            let resolved = try XCTUnwrap(
                memberByName[
                    String(
                        decoding: entry.decodedInstallName,
                        as: UTF8.self
                    )
                ]
            )
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
        for (member, inventory) in zip(members, memberInventories) {
            for entry in inventory.entries {
                let resolved = try XCTUnwrap(
                    memberByName[
                        String(
                            decoding: entry.decodedInstallName,
                            as: UTF8.self
                        )
                    ]
                )
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
        let graph = try
            SyntheticFileImageRuntimeClosureGraphVerifier.compare(
                root: root,
                members: members,
                edges: edges,
                collection: collection,
                rootInventory: rootInventory,
                memberInventories: memberInventories
            )
        let loader = try dynamicLoaderEvidence(uuidSeed: loaderUUIDSeed)
        let documentBytes = renderExpectation(
            role: role,
            rootID: root.contentEvidenceID.sha256,
            loaderID: loader.contentEvidenceID.sha256,
            cacheSet: cacheSet,
            members: members,
            edges: edges
        )
        let expectationFile = admittedFile(documentBytes)
        let anchor = RuntimeClosureExpectationTrustAnchor(
            expectedCurrentDocumentSHA256: expectationFile.sha256,
            expectedCurrentDocumentBytes: UInt64(documentBytes.count),
            minimumEvidenceGeneration: 9,
            verificationUnixSeconds: 2_000_000_000,
            expectedArtifactRole: role
        )
        let expectation = try RuntimeClosureExpectationVerifier.anchor(
            expectationFile: expectationFile,
            trustAnchor: anchor
        )
        let reference = try
            FileImageRuntimeClosureExpectationVerifier.reference(
                anchoredExpectation: expectation,
                currentExpectationAnchor: anchor
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
            expectation: expectation,
            reference: reference
        )
    }

    static func executableRole(
        _ role: RuntimeClosureExpectationArtifactRole
    ) -> ExecutableContentArtifactRole {
        switch role {
        case .git:
            .git
        case .selfGuard:
            .selfGuard
        }
    }

    static func allFileImageChain(count: Int) throws -> Fixture {
        let names = (0..<count).map {
            "/usr/lib/libD2C2Chain\(fourDigit($0)).dylib"
        }
        let specs = names.enumerated().map { index, name in
            MemberSpec.fileImage(
                name,
                targets: index + 1 < names.count
                    ? [names[index + 1]]
                    : []
            )
        }
        return try fixture(
            role: .git,
            rootTargets: [names[0]],
            specs: specs
        )
    }

    static func compare(
        _ fixture: Fixture,
        expectationReference:
            FileImageRuntimeClosureExpectationReference? = nil,
        currentAnchor: RuntimeClosureExpectationTrustAnchor? = nil
    ) throws -> FileImageRuntimeClosureContentExpectationComparison {
        try FileImageRuntimeClosureContentIdentityVerifier.compare(
            expectationReference:
                expectationReference ?? fixture.reference,
            currentExpectationAnchor:
                currentAnchor ?? fixture.anchor,
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

    static func assertFailure(
        fixture: Fixture,
        expectationReference:
            FileImageRuntimeClosureExpectationReference? = nil,
        currentAnchor: RuntimeClosureExpectationTrustAnchor? = nil,
        root: ExecutableContentIdentityEvidence? = nil,
        members: [SyntheticRuntimeClosureMemberRecordComparison]? = nil,
        edges: [SyntheticRuntimeClosureEdgeRecordComparison]? = nil,
        collection:
            SyntheticRuntimeClosureRecordCollectionComparison? = nil,
        memberInventories:
            [SyntheticAcceptedDependencyCommandInventoryComparison]? = nil,
        graphComparison:
            SyntheticFileImageRuntimeClosureGraphComparison? = nil,
        expected: FileImageRuntimeClosureContentIdentityFailure,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(
            try FileImageRuntimeClosureContentIdentityVerifier.compare(
                expectationReference:
                    expectationReference ?? fixture.reference,
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
                $0 as? FileImageRuntimeClosureContentIdentityFailure,
                expected,
                file: file,
                line: line
            )
        }
    }

    static func assertResult(
        _ result: FileImageRuntimeClosureContentExpectationComparison,
        fixture: Fixture,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
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
        XCTAssertEqual(
            result.fileImageMemberCount,
            fixture.reference.declaredFileImageMemberCount,
            file: file,
            line: line
        )
        XCTAssertTrue(result.provesExpectationAnchorMatch, file: file, line: line)
        XCTAssertTrue(result.provesManifestContentMatch, file: file, line: line)
        XCTAssertTrue(result.provesDeclaredStaticGraphMatch, file: file, line: line)
        XCTAssertTrue(result.provesSealedFileImageContinuity, file: file, line: line)
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

        let mirror = Mirror(reflecting: result)
        XCTAssertEqual(
            mirror.children.compactMap(\.label),
            [
                "artifactRole",
                "manifestSHA256",
                "manifestBytes",
                "rootExecutableContentEvidenceID",
                "dynamicLoaderContentEvidenceID",
                "sharedCacheSetID",
                "memberCount",
                "edgeCount",
                "fileImageMemberCount",
                "contentEvidenceID",
                "provesExpectationAnchorMatch",
                "provesManifestContentMatch",
                "provesDeclaredStaticGraphMatch",
                "provesSealedFileImageContinuity",
                "isCompleteRuntimeClosure",
                "provesRuntimeLaunchability",
                "runtimeResolutionOutcome",
                "runtimeDecision",
                "canExecute",
                "canSpawn",
                "canAccessNetwork",
                "canConsumePack",
                "canMutateFileSystem",
                "canImportGitObjects",
                "canBuild",
                "canLoadModel",
                "canReserveOutput",
                "canPublish",
                "constructionSeal",
            ],
            file: file,
            line: line
        )
        let forbiddenTypeFragments = [
            "FileImageRuntimeClosureExpectationReference",
            "AnchoredRuntimeClosureExpectationDocument",
            "RuntimeClosureExpectationTrustAnchor",
            "SyntheticFileImageRuntimeClosureGraphComparison",
            "SyntheticRuntimeClosureRecordCollectionComparison",
            "SyntheticRuntimeClosureMemberRecordComparison",
            "SyntheticRuntimeClosureEdgeRecordComparison",
            "SyntheticAcceptedDependencyCommandInventoryComparison",
            "Foundation.Data",
            "Range<",
        ]
        for child in mirror.children {
            let typeName = String(reflecting: type(of: child.value))
            for forbidden in forbiddenTypeFragments {
                XCTAssertFalse(
                    typeName.contains(forbidden),
                    "retained forbidden type \(typeName)",
                    file: file,
                    line: line
                )
            }
        }
    }

    static func reference(
        for fixture: Fixture,
        rootID: String? = nil,
        loaderID: String? = nil,
        cacheSet: SyntheticSharedCacheSetIdentityEvidence? = nil,
        memberFields: [RuntimeClosureExpectationMemberFields]? = nil,
        edgeFields: [RuntimeClosureExpectationEdgeFields]? = nil
    ) throws -> FileImageRuntimeClosureExpectationReference {
        let resolvedCacheSet = cacheSet ?? fixture.cacheSet
        let resolvedRootID =
            rootID ?? fixture.root.contentEvidenceID.sha256
        var resolvedEdgeFields =
            edgeFields ?? Self.edgeFields(fixture.edges)
        if resolvedRootID != fixture.root.contentEvidenceID.sha256 {
            resolvedEdgeFields = resolvedEdgeFields.map { edge in
                RuntimeClosureExpectationEdgeFields(
                    parentContentEvidenceID:
                        edge.parentContentEvidenceID ==
                            fixture.root.contentEvidenceID.sha256
                        ? resolvedRootID
                        : edge.parentContentEvidenceID,
                    loadCommandOrdinal: edge.loadCommandOrdinal,
                    kind: edge.kind,
                    installName: edge.installName,
                    decodedInstallName: edge.decodedInstallName,
                    resolvedContentEvidenceID:
                        edge.resolvedContentEvidenceID
                )
            }
        }
        let bytes = renderExpectation(
            role: fixture.role,
            rootID: resolvedRootID,
            loaderID:
                loaderID ?? fixture.loader.contentEvidenceID.sha256,
            cacheSet: resolvedCacheSet,
            memberFields:
                memberFields ?? Self.memberFields(fixture.members),
            edgeFields: resolvedEdgeFields
        )
        let file = admittedFile(bytes)
        let anchor = Self.anchor(for: file, role: fixture.role)
        let expectation = try RuntimeClosureExpectationVerifier.anchor(
            expectationFile: file,
            trustAnchor: anchor
        )
        return try FileImageRuntimeClosureExpectationVerifier.reference(
            anchoredExpectation: expectation,
            currentExpectationAnchor: anchor
        )
    }

    static func memberFields(
        _ members: [SyntheticRuntimeClosureMemberRecordComparison]
    ) -> [RuntimeClosureExpectationMemberFields] {
        members.map {
            RuntimeClosureExpectationMemberFields(
                contentEvidenceID: $0.contentEvidenceID,
                storage: $0.storage,
                installName: $0.installName,
                decodedInstallName: $0.decodedInstallName,
                machOUUID: $0.machOUUID,
                primaryCodeDirectoryBlobSHA256:
                    $0.primaryCodeDirectoryBlobSHA256,
                loadCommandsSHA256: $0.loadCommandsSHA256
            )
        }
    }

    static func edgeFields(
        _ edges: [SyntheticRuntimeClosureEdgeRecordComparison]
    ) -> [RuntimeClosureExpectationEdgeFields] {
        edges.map {
            RuntimeClosureExpectationEdgeFields(
                parentContentEvidenceID: $0.parentContentEvidenceID,
                loadCommandOrdinal: $0.loadCommandOrdinal,
                kind: $0.kind,
                installName: $0.installName,
                decodedInstallName: $0.decodedInstallName,
                resolvedContentEvidenceID:
                    $0.resolvedContentEvidenceID
            )
        }
    }

    static func memberField(
        _ value: RuntimeClosureExpectationMemberFields,
        contentEvidenceID: String? = nil,
        storage: SyntheticRuntimeClosureMemberStorage? = nil,
        installName: SyntheticRuntimeClosureInstallName? = nil,
        decodedInstallName: Data? = nil,
        machOUUID: String? = nil,
        primaryCodeDirectoryBlobSHA256: String? = nil,
        loadCommandsSHA256: String? = nil
    ) -> RuntimeClosureExpectationMemberFields {
        RuntimeClosureExpectationMemberFields(
            contentEvidenceID:
                contentEvidenceID ?? value.contentEvidenceID,
            storage: storage ?? value.storage,
            installName: installName ?? value.installName,
            decodedInstallName:
                decodedInstallName ?? value.decodedInstallName,
            machOUUID: machOUUID ?? value.machOUUID,
            primaryCodeDirectoryBlobSHA256:
                primaryCodeDirectoryBlobSHA256 ??
                value.primaryCodeDirectoryBlobSHA256,
            loadCommandsSHA256:
                loadCommandsSHA256 ?? value.loadCommandsSHA256
        )
    }

    static func edgeField(
        _ value: RuntimeClosureExpectationEdgeFields,
        installName: SyntheticRuntimeClosureInstallName? = nil,
        decodedInstallName: Data? = nil,
        resolvedContentEvidenceID: String? = nil
    ) -> RuntimeClosureExpectationEdgeFields {
        RuntimeClosureExpectationEdgeFields(
            parentContentEvidenceID: value.parentContentEvidenceID,
            loadCommandOrdinal: value.loadCommandOrdinal,
            kind: value.kind,
            installName: installName ?? value.installName,
            decodedInstallName:
                decodedInstallName ?? value.decodedInstallName,
            resolvedContentEvidenceID:
                resolvedContentEvidenceID ??
                value.resolvedContentEvidenceID
        )
    }

    static func assertMemberExpectationFailure(
        fixture: Fixture,
        member: RuntimeClosureExpectationMemberFields,
        edge: RuntimeClosureExpectationEdgeFields,
        field: RuntimeClosureExpectationMemberRecordField,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let reference = try Self.reference(
            for: fixture,
            memberFields: [member],
            edgeFields: [edge]
        )
        Self.assertFailure(
            fixture: fixture,
            expectationReference: reference,
            currentAnchor: reference.anchoredExpectation.trustAnchor,
            expected: .memberExpectation(index: 0, field: field),
            file: file,
            line: line
        )
    }

    static func independentPreimage(
        role: RuntimeClosureExpectationArtifactRole
    ) -> Data {
        Data(
            ([
                "fast-mlx-proof-control-file-image-runtime-closure-content-evidence-id-v1",
                "artifact_role=\(role.rawValue)",
                "manifest_sha256=" + String(repeating: "1", count: 64),
                "manifest_bytes=123",
                "root_executable_content_evidence_id=" +
                    String(repeating: "2", count: 64),
                "dynamic_loader_content_evidence_id=" +
                    String(repeating: "3", count: 64),
                "shared_cache_set_id=" +
                    String(repeating: "4", count: 64),
                "file_image_member_count=1",
            ].joined(separator: "\n") + "\n").utf8
        )
    }

    static func expectedPreimage(_ fixture: Fixture) -> Data {
        Data(
            ([
                "fast-mlx-proof-control-file-image-runtime-closure-content-evidence-id-v1",
                "artifact_role=\(fixture.role.rawValue)",
                "manifest_sha256=\(fixture.expectation.documentSHA256)",
                "manifest_bytes=\(fixture.expectation.documentBytes)",
                "root_executable_content_evidence_id=" +
                    fixture.root.contentEvidenceID.sha256,
                "dynamic_loader_content_evidence_id=" +
                    fixture.loader.contentEvidenceID.sha256,
                "shared_cache_set_id=" +
                    fixture.cacheSet.sharedCacheSetID.sha256,
                "file_image_member_count=" +
                    String(
                        fixture.reference.declaredFileImageMemberCount
                    ),
            ].joined(separator: "\n") + "\n").utf8
        )
    }

    static func renderExpectation(
        role: RuntimeClosureExpectationArtifactRole,
        rootID: String,
        loaderID: String,
        cacheSet: SyntheticSharedCacheSetIdentityEvidence,
        members: [SyntheticRuntimeClosureMemberRecordComparison],
        edges: [SyntheticRuntimeClosureEdgeRecordComparison]
    ) -> Data {
        renderExpectation(
            role: role,
            rootID: rootID,
            loaderID: loaderID,
            cacheSet: cacheSet,
            memberFields: memberFields(members),
            edgeFields: edgeFields(edges)
        )
    }

    static func renderExpectation(
        role: RuntimeClosureExpectationArtifactRole,
        rootID: String,
        loaderID: String,
        cacheSet: SyntheticSharedCacheSetIdentityEvidence,
        memberFields: [RuntimeClosureExpectationMemberFields],
        edgeFields: [RuntimeClosureExpectationEdgeFields]
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
            lines.append(
                "\(prefix)suffix_base64url=\(record.suffixBase64URL)"
            )
            lines.append("\(prefix)sha256=\(record.fileSHA256)")
            lines.append("\(prefix)bytes=\(record.fileBytes)")
            lines.append("\(prefix)header_uuid=\(record.headerUUID)")
        }
        lines.append(
            "shared_cache_set_id=\(cacheSet.sharedCacheSetID.sha256)"
        )
        lines.append("member_count=\(memberFields.count)")
        for (index, member) in memberFields.enumerated() {
            let prefix = "member_\(fourDigit(index))_"
            lines.append(
                "\(prefix)content_evidence_id=\(member.contentEvidenceID)"
            )
            lines.append("\(prefix)storage=\(member.storage.rawValue)")
            lines.append(
                "\(prefix)install_name_bytes=\(member.installName.bytes)"
            )
            lines.append(
                "\(prefix)install_name_base64url=" +
                    member.installName.base64URL
            )
            lines.append("\(prefix)macho_uuid=\(member.machOUUID)")
            lines.append(
                "\(prefix)primary_code_directory_blob_sha256=" +
                    member.primaryCodeDirectoryBlobSHA256
            )
            lines.append(
                "\(prefix)load_commands_sha256=" +
                    member.loadCommandsSHA256
            )
        }
        lines.append("edge_count=\(edgeFields.count)")
        for (index, edge) in edgeFields.enumerated() {
            let prefix = "edge_\(fourDigit(index))_"
            lines.append(
                "\(prefix)parent_content_evidence_id=" +
                    edge.parentContentEvidenceID
            )
            lines.append(
                "\(prefix)load_command_ordinal=" +
                    String(edge.loadCommandOrdinal)
            )
            lines.append("\(prefix)kind=\(edge.kind.rawValue)")
            lines.append(
                "\(prefix)install_name_bytes=\(edge.installName.bytes)"
            )
            lines.append(
                "\(prefix)install_name_base64url=" +
                    edge.installName.base64URL
            )
            lines.append(
                "\(prefix)resolved_content_evidence_id=" +
                    edge.resolvedContentEvidenceID
            )
        }
        lines.append(
            "runtime_resolution_outcome=unproved-static-comparison-only"
        )
        lines.append("runtime_authority=none")
        return Data((lines.joined(separator: "\n") + "\n").utf8)
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

    static func anchor(
        for file: AdmittedFile,
        role: RuntimeClosureExpectationArtifactRole,
        sha256: String? = nil,
        bytes: UInt64? = nil,
        minimumGeneration: UInt64 = 9,
        verificationUnixSeconds: UInt64 = 2_000_000_000
    ) -> RuntimeClosureExpectationTrustAnchor {
        RuntimeClosureExpectationTrustAnchor(
            expectedCurrentDocumentSHA256: sha256 ?? file.sha256,
            expectedCurrentDocumentBytes:
                bytes ?? UInt64(file.bytes.count),
            minimumEvidenceGeneration: minimumGeneration,
            verificationUnixSeconds: verificationUnixSeconds,
            expectedArtifactRole: role
        )
    }

    static func sharedCacheSnapshot(
        name: String,
        commands: [Data],
        cacheSet: SyntheticSharedCacheSetIdentityEvidence,
        uuidSeed: UInt8
    ) throws -> SyntheticSharedCacheImageLoadCommandSnapshot {
        let installName = FMAFileImageFixture.installName(name)
        let commandBytes = commands.reduce(into: Data()) {
            $0.append($1)
        }
        let evidence = try
            SyntheticSharedCacheImageContentIdentityVerifier.derive(
                cacheSetEvidence: cacheSet,
                facts: SyntheticSharedCacheImageContentFacts(
                    installNameBytes: installName.bytes,
                    installNameBase64URL: installName.base64URL,
                    machOUUID: fixedHex(
                        seed: uuidSeed,
                        count: 16
                    ),
                    primaryCodeDirectory: .absent,
                    loadCommandsSHA256: sha256Hex(commandBytes)
                )
            )
        return try SyntheticSharedCacheImageLoadCommandSnapshotVerifier
            .derive(
                imageEvidence: evidence,
                loadCommandBytes: commandBytes
            )
    }

    static func dynamicLoaderEvidence(uuidSeed: UInt8 = 0x20)
        throws -> DynamicLoaderContentIdentityEvidence
    {
        let signature = superBlob(
            entries: [
                (
                    0,
                    codeDirectory(
                        signingIdentifier:
                            Data("com.example.d2c2.dyld".utf8)
                    )
                )
            ]
        )
        let comparison = try
            SyntheticDynamicLoaderMachOIdentityParser.parse(
                dynamicLoaderMachO(
                    signatureRegion: signature,
                    uuidSeed: uuidSeed
                )
            )
        return try DynamicLoaderContentIdentityVerifier.derive(
            comparison: comparison
        )
    }

    static func dynamicLoaderMachO(
        signatureRegion: Data,
        uuidSeed: UInt8 = 0x20
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
        while !result.count.isMultiple(of: 8) {
            result.append(0)
        }
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
        for (_, blob) in entries {
            result.append(blob)
        }
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

    static func p3ExecutableComparison(
        evidence: ExecutableContentIdentityEvidence,
        role: ExecutableIdentityArtifactRole
    ) throws -> (
        ExecutableContentExpectationComparison,
        ExecutableIdentityExpectationTrustAnchor
    ) {
        let fields = p3ExecutableFields(
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
            verificationUnixSeconds: p3CommonTime,
            expectedArtifactRole: role
        )
        let expectation = try ExecutableIdentityExpectationVerifier.anchor(
            expectationFile: file,
            trustAnchor: anchor
        )
        return (
            try ExecutableContentIdentityVerifier.match(
                evidence: evidence,
                expectation: expectation
            ),
            anchor
        )
    }

    static func p3ExecutableFields(
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
            cpuSubtype: p3Hex8(comparison.cpuSubtype),
            headerFlags: p3Hex8(comparison.headerFlags),
            loadCommandCount: UInt64(comparison.loadCommandCount),
            loadCommandBytes: UInt64(comparison.loadCommandBytes.count),
            loadCommandsSHA256: comparison.loadCommandsSHA256,
            machOUUID: p3Hex(comparison.machOUUID),
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
                    flags: p3Hex8($0.flags),
                    signingIdentifierBytes:
                        UInt64($0.signingIdentifier.count),
                    signingIdentifierBase64URL:
                        p3Base64URL($0.signingIdentifier),
                    teamIdentifierBytes:
                        UInt64($0.teamIdentifier.count),
                    teamIdentifierBase64URL:
                        p3Base64URL($0.teamIdentifier)
                )
            },
            cmsBlobSHA256: comparison.cmsBlobSHA256,
            cmsBlobBytes: UInt64(comparison.cmsBlob?.count ?? 0)
        )
    }

    static func p3SignedClaim(
        toolManifestSHA256: String,
        toolManifestBytes: UInt64,
        runtimePolicySHA256: String,
        signingDynamically: Bool = false
    ) throws -> OperatorSignedRunClaim {
        let rootKey = try p3TestKey(seed: 0x10)
        let runKey = try p3TestKey(seed: 0x30)
        let authorizationKey = try p3TestKey(seed: 0x50)
        let worker = try p3AuthorizedFile(
            Data("worker-stage-c3\n".utf8),
            purpose: .workerBytes,
            key: authorizationKey,
            signatureBase64:
                "snhEEEl8XOOpC2BefJL9gVh96AIg4GjO2wbBt3z+p3Bup0+phwtCBVT4jmiSVyaA" +
                "c6qP0GRflGLI9wRXI/ZtDw=="
        )
        let baseline = try p3AuthorizedFile(
            Data("baseline-stage-c3\n".utf8),
            purpose: .sourceManifest,
            key: authorizationKey,
            signatureBase64:
                "kQHJT43I7e874p/2HEDUoi+Yq6VKxlOdYtXxp/boMtALfMM5eQZxclwOZLCZ1EV7" +
                "3Y81jR27fBa9Kdek23ahAA=="
        )
        let candidate = try p3AuthorizedFile(
            Data("candidate-stage-c3\n".utf8),
            purpose: .sourceManifest,
            key: authorizationKey,
            signatureBase64:
                "tuX5DenUQTWOXIXZIfb+p1ti6T04ERNr/2xRN2aRHM3YrWzUhXBNfLQBhlvKXy11" +
                "dSLvMOBPk0YbMt12ZO/9Cg=="
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
        let keyPolicyBytes = try OperatorKeyPolicyVerifier.policyBytes(
            fields: keyFields
        )
        let keyPolicyFile = admittedFile(keyPolicyBytes)
        let keyPolicySignature =
            "PTOXcmv0GTOCFYoUJ1ll7b+1nNbG8f/8/oZZ+wpsdNJ1Mil7nWrRzim+bnK4ws40" +
            "m5cL0C4i74ceLsCeeZ4JBA=="
        let keyPolicy = try OperatorKeyPolicyVerifier.admit(
            policyFile: keyPolicyFile,
            rootSignatureBase64: keyPolicySignature,
            trustAnchor: OperatorKeyPolicyTrustAnchor(
                rootPublicKeyBase64: rootKey.publicKeyBase64,
                rootKeyID: rootKey.keyID,
                expectedCurrentPolicySHA256: keyPolicyFile.sha256,
                minimumPolicyGeneration: 7,
                verificationUnixSeconds: p3CommonTime
            )
        )
        let runner = admittedFile(Data("runner-stage-c3\n".utf8))
        let expectations = OperatorRunClaimAdmissionExpectations(
            keyPolicy: keyPolicy,
            hostAdmissionID: p3HexByte(0x04),
            runner: runner,
            resultPairID: p3HexByte(0x05),
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
            worker: p3AuthorizedReference(worker),
            policies: OperatorRunClaimPolicyReferences(
                sourceSHA256: p3HexByte(0x10),
                dependencySHA256: p3HexByte(0x11),
                buildSHA256: p3HexByte(0x12),
                runtimeSHA256: runtimePolicySHA256,
                preflightSHA256: p3HexByte(0x14),
                publicationSHA256: p3HexByte(0x15)
            ),
            toolManifest: OperatorRunClaimByteIdentity(
                sha256: toolManifestSHA256,
                byteCount: toolManifestBytes
            ),
            baseline: OperatorRunClaimSourceReference(
                role: .baseline,
                sourceManifest: p3AuthorizedReference(baseline),
                gitCommitSHA1: String(repeating: "a", count: 40),
                gitTreeSHA1: String(repeating: "b", count: 40),
                route: .decompressedDeepSeekV3,
                slot: .baseline,
                buildReceiptID: p3HexByte(0x17),
                binary: OperatorRunClaimByteIdentity(
                    sha256: p3HexByte(0x18),
                    byteCount: 1_001
                )
            ),
            candidate: OperatorRunClaimSourceReference(
                role: .candidate,
                sourceManifest: p3AuthorizedReference(candidate),
                gitCommitSHA1: String(repeating: "c", count: 40),
                gitTreeSHA1: String(repeating: "d", count: 40),
                route: .absorbedMLADeepSeekV3Explicit,
                slot: .candidate,
                buildReceiptID: p3HexByte(0x19),
                binary: OperatorRunClaimByteIdentity(
                    sha256: p3HexByte(0x20),
                    byteCount: 1_002
                )
            ),
            model: OperatorRunClaimAuthorizedPayloadReference(
                authorizationID: p3HexByte(0x21),
                payload: OperatorRunClaimByteIdentity(
                    sha256: p3HexByte(0x22),
                    byteCount: 55
                )
            ),
            tokenizer: OperatorRunClaimAuthorizedPayloadReference(
                authorizationID: p3HexByte(0x23),
                payload: OperatorRunClaimByteIdentity(
                    sha256: p3HexByte(0x24),
                    byteCount: 66
                )
            ),
            workload: OperatorRunClaimAuthorizedPayloadReference(
                authorizationID: p3HexByte(0x25),
                payload: OperatorRunClaimByteIdentity(
                    sha256: p3HexByte(0x26),
                    byteCount: 77
                )
            ),
            resultPairID: expectations.resultPairID
        )
        let claimBytes = try OperatorSignedRunClaimVerifier.claimBytes(
            fields: fields
        )
        let signature: String
        if signingDynamically {
            signature = try runKey.privateKey.signature(
                for: claimBytes
            ).base64EncodedString()
        } else {
            signature =
                "pD5hngf6hXOftJv6fxth7mMlo1zWD1bvCdEPHve2AAuBoD0vlBD83MgwnjTPbR3j" +
                "m++TaxI/tAFu6HgtVA3YBQ=="
        }
        return try OperatorSignedRunClaimVerifier.verify(
            claimBytes: claimBytes,
            signatureBase64: signature,
            expectations: expectations
        )
    }

    static func p3AuthorizedFile(
        _ bytes: Data,
        purpose: OperatorAuthorizationPurpose,
        key: P3TestKey,
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

    static func p3AuthorizedReference(
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

    static func p3TestKey(seed: UInt8) throws -> P3TestKey {
        let key = try Curve25519.Signing.PrivateKey(
            rawRepresentation: Data((0..<32).map {
                seed &+ UInt8($0)
            })
        )
        let publicBytes = key.publicKey.rawRepresentation
        return P3TestKey(
            privateKey: key,
            publicKeyBase64: publicBytes.base64EncodedString(),
            keyID: sha256Hex(publicBytes)
        )
    }

    static func p3ContextPreimage(_ fixture: P3Fixture) -> Data {
        let tool = fixture.toolAnchor
        let denial = fixture.denialAnchor
        let gitExecutable = fixture.gitExecutableAnchor
        let selfGuardExecutable = fixture.selfGuardExecutableAnchor
        let gitClosure = fixture.gitClosure.anchor
        let selfGuardClosure = fixture.selfGuardClosure.anchor
        let lines: [String] = [
                "fast-mlx-proof-control-file-image-execution-identity-anchor-context-id-v1",
                "git_tool_policy_v2_expected_sha256=" +
                    tool.expectedCurrentPolicySHA256,
                "git_tool_policy_v2_expected_bytes=" +
                    String(tool.expectedCurrentPolicyBytes),
                "git_tool_policy_v2_minimum_generation=" +
                    String(tool.minimumPolicyGeneration),
                "git_tool_policy_v2_verification_unix_seconds=" +
                    String(tool.verificationUnixSeconds),
                "runtime_denial_v2_expected_sha256=" +
                    denial.expectedCurrentPolicySHA256,
                "runtime_denial_v2_expected_bytes=" +
                    String(denial.expectedCurrentPolicyBytes),
                "runtime_denial_v2_minimum_generation=" +
                    String(denial.minimumPolicyGeneration),
                "runtime_denial_v2_verification_unix_seconds=" +
                    String(denial.verificationUnixSeconds),
                "git_executable_expectation_sha256=" +
                    gitExecutable.expectedCurrentDocumentSHA256,
                "git_executable_expectation_bytes=" +
                    String(gitExecutable.expectedCurrentDocumentBytes),
                "git_executable_minimum_generation=" +
                    String(gitExecutable.minimumEvidenceGeneration),
                "git_executable_verification_unix_seconds=" +
                    String(gitExecutable.verificationUnixSeconds),
                "self_guard_executable_expectation_sha256=" +
                    selfGuardExecutable.expectedCurrentDocumentSHA256,
                "self_guard_executable_expectation_bytes=" +
                    String(selfGuardExecutable.expectedCurrentDocumentBytes),
                "self_guard_executable_minimum_generation=" +
                    String(selfGuardExecutable.minimumEvidenceGeneration),
                "self_guard_executable_verification_unix_seconds=" +
                    String(selfGuardExecutable.verificationUnixSeconds),
                "git_closure_expectation_sha256=" +
                    gitClosure.expectedCurrentDocumentSHA256,
                "git_closure_expectation_bytes=" +
                    String(gitClosure.expectedCurrentDocumentBytes),
                "git_closure_minimum_generation=" +
                    String(gitClosure.minimumEvidenceGeneration),
                "git_closure_verification_unix_seconds=" +
                    String(gitClosure.verificationUnixSeconds),
                "self_guard_closure_expectation_sha256=" +
                    selfGuardClosure.expectedCurrentDocumentSHA256,
                "self_guard_closure_expectation_bytes=" +
                    String(selfGuardClosure.expectedCurrentDocumentBytes),
                "self_guard_closure_minimum_generation=" +
                    String(selfGuardClosure.minimumEvidenceGeneration),
                "self_guard_closure_verification_unix_seconds=" +
                    String(selfGuardClosure.verificationUnixSeconds),
        ]
        return Data((lines.joined(separator: "\n") + "\n").utf8)
    }

    static func p3FinalPreimage(
        _ fixture: P3Fixture,
        contextID: String
    ) -> Data {
        let lines: [String] = [
                "fast-mlx-proof-control-file-image-execution-identity-comparison-id-v1",
                "run_claim_id=" +
                    fixture.toolReference.signedClaim.claimID.rawValue,
                "anchor_context_id=\(contextID)",
                "git_tool_policy_v2_sha256=" +
                    fixture.toolReference.policyDocument.policySHA256,
                "runtime_denial_policy_v2_sha256=" +
                    fixture.denial.policySHA256,
                "git_executable_expectation_sha256=" +
                    fixture.gitExecutable.expectation.documentSHA256,
                "git_executable_content_evidence_id=" +
                    fixture.gitExecutable.contentEvidence
                    .contentEvidenceID.sha256,
                "self_guard_executable_expectation_sha256=" +
                    fixture.selfGuardExecutable.expectation.documentSHA256,
                "self_guard_executable_content_evidence_id=" +
                    fixture.selfGuardExecutable.contentEvidence
                    .contentEvidenceID.sha256,
                "git_closure_manifest_sha256=" +
                    fixture.gitClosure.expectation.documentSHA256,
                "git_file_image_runtime_closure_content_evidence_id=" +
                    fixture.gitClosureComparison.contentEvidenceID.sha256,
                "self_guard_closure_manifest_sha256=" +
                    fixture.selfGuardClosure.expectation.documentSHA256,
                "self_guard_file_image_runtime_closure_content_evidence_id=" +
                    fixture.selfGuardClosureComparison
                    .contentEvidenceID.sha256,
                "runtime_decision=no-go",
        ]
        return Data((lines.joined(separator: "\n") + "\n").utf8)
    }

    static func p3Hex(_ bytes: Data) -> String {
        bytes.map { String(format: "%02x", $0) }.joined()
    }

    static func p3Hex8(_ value: UInt32) -> String {
        String(format: "%08x", value)
    }

    static func p3HexByte(_ value: UInt8) -> String {
        String(repeating: String(format: "%02x", value), count: 32)
    }

    static func p3Base64URL(_ data: Data) -> String {
        data.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    static func fourDigit(_ value: Int) -> String {
        String(format: "%04d", value)
    }

    static func fixedHex(seed: UInt8, count: Int) -> String {
        (0..<count).map {
            String(format: "%02x", seed &+ UInt8($0))
        }.joined()
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
