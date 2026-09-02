import Foundation
import XCTest

/// Slice 4b structural gates over the `fastmlx-proof-runner` sources
/// (design verdict hazards C1 and P4/P5 riders). Like the Slice 3
/// no-self-sign gate, these pass against current sources on their first
/// run — the property holds today and the gate pins it — and every
/// detector carries a negative control proving it actually detects the
/// forbidden shape, so a passing run is evidence of absence, not of a
/// blind scanner.
final class ProofRunnerStructuralGateTests: XCTestCase {
    private static var spikeRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private static var proofRunnerSources: URL {
        spikeRoot.appendingPathComponent("Sources/fastmlx-proof-runner")
    }

    private static let observationsInitToken =
        "Qwen38MTPPerformanceScorecardRunnerLaunchObservations("

    /// The exact spawn-result feed the C1 mitigation requires: the
    /// observations are constructed directly and only from the two
    /// `Qwen38ScorecardSpawnedWorker.evidenceID` values.
    private static let goldenFeedTokens = [
        "candidateProcessIsolationEvidenceID: candidateWorker.evidenceID",
        "referenceProcessIsolationEvidenceID: referenceWorker.evidenceID",
    ]

    /// The echo ingredient: runner code must never read a worker-reported
    /// launch binding's evidence ID (echoing it into the observations
    /// would make the equality check vacuous — hazard C1).
    private static let echoToken = "launchBinding.processIsolationEvidenceID"

    /// P4 rider: production runner code calls ONLY the combined entry.
    private static let combinedEntryToken =
        "validateWithRunnerLaunchObservations("
    private static let forbiddenValidationTokens = [
        "validateRunnerLaunchEquality(",
        "Gate.validate(",
        "validateAuthority(",
        "evaluateCandidate(",
        "validateJSONL(",
    ]

    /// P5 rider: the trusted run identity's model/corpus digests are
    /// minted from the SIGNED CLAIM, never from the gate's constants
    /// (constant-derived minting would make the operator-signed digests
    /// decorative).
    private static let claimDerivedIdentityTokens = [
        "authorization.authorizedHarnessGitSHA1",
        "authorization.authorizedModelConfigSHA256",
        "authorization.authorizedTensorManifestSHA256",
        "authorization.authorizedCorpusID",
        "authorization.authorizedCorpusContentSHA256",
    ]
    private static let forbiddenConstantIdentityTokens = [
        "requiredArtifact.targetConfigSHA256",
        "requiredArtifact.targetTensorManifestSHA256",
        "requiredWorkload.id",
        "requiredWorkload.contentSHA256",
    ]

    // MARK: - Detector negative controls

    func testDetectorFlagsForbiddenShapes() {
        let echoFixture = """
            let observations = Qwen38MTPPerformanceScorecardRunnerLaunchObservations(
                candidateProcessIsolationEvidenceID:
                    evidence.candidate.launchBinding.processIsolationEvidenceID,
                referenceProcessIsolationEvidenceID:
                    evidence.reference.launchBinding.processIsolationEvidenceID)
            """
        XCTAssertTrue(echoFixture.contains(Self.echoToken))
        XCTAssertEqual(
            Self.occurrences(of: Self.observationsInitToken, in: echoFixture),
            1
        )

        let doubledFixture = echoFixture + "\n" + echoFixture
        XCTAssertEqual(
            Self.occurrences(of: Self.observationsInitToken, in: doubledFixture),
            2
        )

        let forbiddenValidationFixture = """
            _ = try Qwen38MTPPerformanceScorecardGate.validate(evidence)
            try Gate.validate(evidence, authority: authority)
            try Qwen38MTPPerformanceScorecardGate.validateRunnerLaunchEquality(
                evidence, observations: observations)
            try Qwen38MTPPerformanceScorecardGate.validateAuthority(authority)
            let verdict = try Gate.evaluateCandidate(evidence)
            let verdicts = try Gate.validateJSONL(data)
            """
        for token in Self.forbiddenValidationTokens {
            XCTAssertTrue(
                forbiddenValidationFixture.contains(token),
                "detector failed to flag \(token)"
            )
        }

        let constantIdentityFixture = """
            modelConfigHash: Qwen38MTPPerformanceScorecardGate
                .requiredArtifact.targetConfigSHA256,
            modelCheckpointManifestHash: Gate.requiredArtifact.targetTensorManifestSHA256,
            corpusID: Qwen38MTPPerformanceScorecardGate.requiredWorkload.id,
            corpusContentHash: Gate.requiredWorkload.contentSHA256,
            """
        for token in Self.forbiddenConstantIdentityTokens {
            XCTAssertTrue(
                constantIdentityFixture.contains(token),
                "detector failed to flag \(token)"
            )
        }
    }

    // MARK: - Scanned-surface sanity

    func testRunnerSourcesAreNonEmpty() throws {
        let files = try Self.swiftFiles(in: Self.proofRunnerSources)
        XCTAssertGreaterThanOrEqual(files.count, 2)
        let names = Set(files.map(\.lastPathComponent))
        XCTAssertTrue(names.contains("main.swift"))
        XCTAssertTrue(names.contains("Qwen38ScorecardRunnerPipeline.swift"))
    }

    // MARK: - C1: single spawn-fed observations site, no echo ingredient

    func testObservationsInitializerAppearsExactlyOnceFedFromSpawnResults() throws {
        var initSites = 0
        var goldenFeedSites = 0
        for file in try Self.swiftFiles(in: Self.proofRunnerSources) {
            let content = try String(contentsOf: file, encoding: .utf8)
            initSites += Self.occurrences(
                of: Self.observationsInitToken,
                in: content
            )
            if Self.goldenFeedTokens.allSatisfy(content.contains) {
                goldenFeedSites += 1
            }
        }
        XCTAssertEqual(
            initSites,
            1,
            "the launch-observations initializer must appear exactly once"
        )
        XCTAssertEqual(
            goldenFeedSites,
            1,
            "the single observations site must be fed from the two "
                + "spawned-worker evidence IDs"
        )
    }

    func testRunnerSourcesNeverReadWorkerReportedLaunchBindingIDs() throws {
        for file in try Self.swiftFiles(in: Self.proofRunnerSources) {
            let content = try String(contentsOf: file, encoding: .utf8)
            XCTAssertFalse(
                content.contains(Self.echoToken),
                "\(file.lastPathComponent) reads a worker-reported launch "
                    + "binding evidence ID (echo seam C1)"
            )
        }
    }

    // MARK: - P4: combined validation entry only

    func testRunnerCallsOnlyTheCombinedValidationEntry() throws {
        var combinedEntrySites = 0
        var violations: [String] = []
        for file in try Self.swiftFiles(in: Self.proofRunnerSources) {
            let content = try String(contentsOf: file, encoding: .utf8)
            combinedEntrySites += Self.occurrences(
                of: Self.combinedEntryToken,
                in: content
            )
            for token in Self.forbiddenValidationTokens
            where content.contains(token) {
                violations.append("\(file.lastPathComponent): \(token)")
            }
        }
        XCTAssertGreaterThanOrEqual(
            combinedEntrySites,
            1,
            "the runner must call the combined validation entry"
        )
        XCTAssertEqual(violations, [], violations.joined(separator: ", "))
    }

    // MARK: - P5: claim-derived identity minting

    func testRunIdentityMintingIsClaimDerivedNotConstantDerived() throws {
        var content = ""
        for file in try Self.swiftFiles(in: Self.proofRunnerSources) {
            content += try String(contentsOf: file, encoding: .utf8)
        }
        for token in Self.claimDerivedIdentityTokens {
            XCTAssertTrue(
                content.contains(token),
                "runner sources must mint identity from the claim: \(token)"
            )
        }
        for token in Self.forbiddenConstantIdentityTokens {
            XCTAssertFalse(
                content.contains(token),
                "runner sources mint identity from a gate constant: \(token)"
            )
        }
    }

    // MARK: - Helpers

    private static func occurrences(of token: String, in content: String) -> Int {
        var count = 0
        var searchRange = content.startIndex ..< content.endIndex
        while let found = content.range(of: token, range: searchRange) {
            count += 1
            searchRange = found.upperBound ..< content.endIndex
        }
        return count
    }

    private static func swiftFiles(in directory: URL) throws -> [URL] {
        try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        )
        .filter { $0.pathExtension == "swift" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }
}
