import CryptoKit
import Darwin
import Foundation
import XCTest

@testable import HarnessCore
@testable import ProofControl
@testable import ScorecardPairControl
@testable import fastmlx_proof_runner

/// Security-review F2: pins the P5 claim→run-identity mapping directly.
/// The fixture claim carries the GATE'S OWN required digests, so the
/// minted identity must equal the exact constants `validateAuthority`
/// enforces — a swapped or constant-derived mapping would reject every
/// CORRECT operator claim on the real run while leaving black-box tests
/// green; this test makes such a regression fail loudly. Fixture keys are
/// deterministic test seeds only (credential boundary preserved); the
/// fixture chain mirrors the resolver suite's, kept local per the chain's
/// per-file self-containment convention.
final class Qwen38ScorecardRunnerIdentityMintingTests: XCTestCase {
    private static let rootSeed = Data(repeating: 0xE8, count: 32)
    private static let activeSeed = Data(repeating: 0xF9, count: 32)
    private static let claimHarnessGitSHA1 = String(repeating: "1e", count: 20)

    private var caseRoot: URL!

    override func setUpWithError() throws {
        let canonicalTemporaryPath = try XCTUnwrap(
            Darwin.realpath(NSTemporaryDirectory(), nil)
        )
        defer { Darwin.free(canonicalTemporaryPath) }
        caseRoot = URL(
            fileURLWithPath: String(cString: canonicalTemporaryPath),
            isDirectory: true
        )
        .appendingPathComponent(
            "fast-mlx-qwen38-identity-minting-\(UUID().uuidString)"
        )
        try FileManager.default.createDirectory(
            at: caseRoot,
            withIntermediateDirectories: false
        )
    }

    override func tearDownWithError() throws {
        if let caseRoot {
            try? FileManager.default.removeItem(at: caseRoot)
        }
    }

    func testRunIdentityIsMintedFromClaimAndMatchesGateConstantsForCorrectClaim() throws {
        typealias Gate = Qwen38MTPPerformanceScorecardGate
        let authorization = try resolvedAuthorization(
            modelSHA256: Gate.requiredArtifact.targetConfigSHA256,
            tensorManifestSHA256: Gate.requiredArtifact.targetTensorManifestSHA256,
            corpusID: Gate.requiredWorkload.id,
            corpusContentSHA256: Gate.requiredWorkload.contentSHA256
        )
        let hostFacts = Qwen38ScorecardRunnerHostFacts(
            chipBrand: "Apple M3 Ultra",
            ramBytes: 274_877_906_944,
            osBuild: "macOS 26.6.2"
        )

        let identity = qwen38ScorecardRunnerTrustedRunIdentity(
            authorization: authorization,
            hostFacts: hostFacts
        )

        // The P5 property: exactly the constants validateAuthority pins.
        XCTAssertEqual(
            identity.modelConfigHash,
            Gate.requiredArtifact.targetConfigSHA256
        )
        XCTAssertEqual(
            identity.modelCheckpointManifestHash,
            Gate.requiredArtifact.targetTensorManifestSHA256
        )
        XCTAssertEqual(identity.corpusID, Gate.requiredWorkload.id)
        XCTAssertEqual(
            identity.corpusContentHash,
            Gate.requiredWorkload.contentSHA256
        )
        // A swapped config/manifest mapping cannot hide: the two required
        // digests are distinct.
        XCTAssertNotEqual(
            Gate.requiredArtifact.targetConfigSHA256,
            Gate.requiredArtifact.targetTensorManifestSHA256
        )

        XCTAssertEqual(identity.harnessGitSHA, Self.claimHarnessGitSHA1)
        XCTAssertEqual(
            identity.candidateMLXSwiftVersion,
            ScorecardPairControlVersions.mlxSwiftVersion
        )
        XCTAssertEqual(identity.measurementClass, Gate.measurementClass)
        XCTAssertEqual(identity.modelLabel, Gate.modelArtifactLabel)
        XCTAssertEqual(identity.modelQuant, ModelQuantInfo(bits: 8, groupSize: 32))
        XCTAssertNil(identity.referenceMLXVersion)
        XCTAssertNil(identity.referenceMLXLMVersion)

        // Host facts: verbatim runner-collected values plus the recomputed
        // chip:ram:os digest recipe.
        XCTAssertEqual(identity.hardwareChip, hostFacts.chipBrand)
        XCTAssertEqual(identity.hardwareRAMBytes, hostFacts.ramBytes)
        XCTAssertEqual(identity.hardwareOSBuild, hostFacts.osBuild)
        XCTAssertEqual(
            identity.hostIdentityDigest,
            qwen38MTPScorecardSHA256Hex(
                Data(
                    "\(hostFacts.chipBrand):\(hostFacts.ramBytes):\(hostFacts.osBuild)"
                        .utf8))
        )
    }

    // MARK: - Fixtures (deterministic test seeds only)

    private func resolvedAuthorization(
        modelSHA256: String,
        tensorManifestSHA256: String,
        corpusID: String,
        corpusContentSHA256: String
    ) throws -> Qwen38ScorecardResolvedRunAuthorization {
        let root = try Curve25519.Signing.PrivateKey(
            rawRepresentation: Self.rootSeed
        )
        let active = try Curve25519.Signing.PrivateKey(
            rawRepresentation: Self.activeSeed
        )
        let rootKeyID = Self.sha256Hex(root.publicKey.rawRepresentation)
        let policyBytes = try Qwen38ScorecardKeyPolicyVerifier.policyBytes(
            fields: Qwen38ScorecardKeyPolicyFields(
                rootKeyID: rootKeyID,
                policyGeneration: 3,
                validFromUnixSeconds: 1_700_000_000,
                validUntilUnixSeconds: 1_800_000_000,
                activeOperatorKeyID: Self.sha256Hex(
                    active.publicKey.rawRepresentation
                ),
                activeOperatorPublicKeyBase64:
                    active.publicKey.rawRepresentation.base64EncodedString(),
                activeOperatorScope: .scorecardRunClaim,
                allowedClaimSubject: .mtpScorecardResultPair,
                revokedOperatorKeyIDs: []
            )
        )
        let policyPath = caseRoot.appendingPathComponent("policy.txt")
        try policyBytes.write(to: policyPath)
        let policyFile = try AdmittedFile.capture(
            absolutePath: policyPath.path,
            maximumBytes: 4_096
        )
        let rootSignature = try root.signature(for: policyFile.bytes)
        let policy = try Qwen38ScorecardKeyPolicyVerifier.admit(
            policyFile: policyFile,
            rootSignatureBase64: rootSignature.base64EncodedString(),
            trustAnchor: Qwen38ScorecardKeyPolicyTrustAnchor(
                rootPublicKeyBase64:
                    root.publicKey.rawRepresentation.base64EncodedString(),
                rootKeyID: rootKeyID,
                expectedCurrentPolicySHA256: policyFile.sha256,
                minimumPolicyGeneration: 1,
                verificationUnixSeconds: 1_750_000_000
            )
        )

        let claimBytes = try Qwen38ScorecardRunClaimVerifier.claimBytes(
            fields: Qwen38ScorecardRunClaimFields(
                subject: .mtpScorecardResultPair,
                modelSHA256: modelSHA256,
                tokenizerSHA256: String(repeating: "aa", count: 32),
                tensorManifestSHA256: tensorManifestSHA256,
                chatTemplateSHA256: String(repeating: "bb", count: 32),
                quantizationIdentity: "mxfp8",
                target: Qwen38ScorecardModelIdentity(
                    modelID: "mlx-community/Qwen3.8-27B-mxfp8",
                    revision: "1a2b3c4"
                ),
                drafter: Qwen38ScorecardModelIdentity(
                    modelID: "mlx-community/Qwen3.8-27B-MTP-mxfp8",
                    revision: "5d6e7f8"
                ),
                sourceID: String(repeating: "6d", count: 32),
                hostAdmissionID: String(repeating: "ce", count: 32),
                harnessGitSHA1: Self.claimHarnessGitSHA1,
                gdnOnMode: .on,
                gdnOffMode: .off,
                corpusID: corpusID,
                corpusContentSHA256: corpusContentSHA256,
                resultPairID: String(repeating: "2e", count: 32)
            )
        )
        let claimSignature = try active.signature(for: claimBytes)
        return try Qwen38ScorecardRunAuthorizationResolver.resolve(
            claimBytes: claimBytes,
            claimSignatureBase64: claimSignature.base64EncodedString(),
            policy: policy
        )
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
