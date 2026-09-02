import Foundation
import XCTest

@testable import ProofControl

/// Slice 3 structural no-self-sign property: the trust kernel
/// (ProofControl) and the runner executable must contain NO signing
/// capability — they can only VERIFY operator signatures. `promotable`
/// authority exists only as a `Qwen38ScorecardResolvedRunAuthorization`
/// value, whose sole constructor requires a verifying operator Ed25519
/// signature; a decoding conformance on that file would be a forgery
/// constructor and is likewise forbidden.
///
/// This gate passes against current sources on its first run (like the
/// existing copy-vs-in-place equivalence gate precedent): the property
/// holds today and the gate pins it. The detector negative controls below
/// prove the scanner actually detects each forbidden token, so a passing
/// run is evidence of absence, not of a blind scanner. Fixture Ed25519
/// keys live ONLY under Tests/, which this gate deliberately does not
/// scan.
final class ProofControlNoSelfSignStructuralTests: XCTestCase {
    /// Signing-capability tokens forbidden in every production source of
    /// the trust kernel and the runner: CryptoKit private keys and signing
    /// calls, and the Security-framework signing surface.
    private static let forbiddenTokens = [
        "PrivateKey",
        ".signature(",
        "SecKeyCreateSignature",
        "import Security",
    ]

    /// Tokens forbidden specifically in the resolved-authorization file:
    /// any serialization-protocol conformance there could reintroduce a
    /// public constructor for the authorization value.
    private static let resolvedAuthorizationForbiddenTokens = [
        "Codable",
        "Decodable",
    ]

    private static let resolvedAuthorizationFileName =
        "Qwen38ScorecardResolvedRunAuthorization.swift"

    private static var spikeRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private static var proofControlSources: URL {
        spikeRoot.appendingPathComponent("Sources/ProofControl")
    }

    private static var proofRunnerSources: URL {
        spikeRoot.appendingPathComponent("Sources/fastmlx-proof-runner")
    }

    /// Slice 4b: the two-process test fixture is production-shaped source
    /// (an executable target, not test code), so it is scanned too; its
    /// claim identity arrives via argv precisely so it needs no signing.
    private static var fixtureWorkerSources: URL {
        spikeRoot.appendingPathComponent("Sources/qwen38-scorecard-fixture-worker")
    }

    /// Slice 4b security-review N2: the runner-linked pair-control layer
    /// is scanned too — it sits inside the runner's trust boundary.
    private static var scorecardPairControlSources: URL {
        spikeRoot.appendingPathComponent("Sources/ScorecardPairControl")
    }

    // MARK: - Detector negative controls

    /// The scanner must FLAG a fixture containing each forbidden token —
    /// this is the detection-capability proof that keeps the zero-hit
    /// assertions below meaningful.
    func testDetectorFlagsEveryForbiddenToken() {
        let fixture = """
            let key = Curve25519.Signing.PrivateKey()
            let sig = try key.signature(for: data)
            let raw = SecKeyCreateSignature(k, a, d, &e)
            import Security
            """
        for token in Self.forbiddenTokens {
            XCTAssertTrue(
                Self.matches(token, in: fixture),
                "detector failed to flag \(token)"
            )
        }

        let decodingFixture = "struct X: Codable {}\nstruct Y: Decodable {}"
        for token in Self.resolvedAuthorizationForbiddenTokens {
            XCTAssertTrue(
                Self.matches(token, in: decodingFixture),
                "detector failed to flag \(token)"
            )
        }
    }

    // MARK: - Scanned-surface sanity

    /// The gate cannot pass by scanning nothing: both production
    /// directories must exist and contain the expected files.
    func testScannedSurfacesAreNonEmpty() throws {
        let proofControlFiles = try Self.swiftFiles(
            in: Self.proofControlSources
        )
        XCTAssertGreaterThanOrEqual(proofControlFiles.count, 10)
        let proofControlNames = Set(
            proofControlFiles.map(\.lastPathComponent)
        )
        for required in [
            "ProcessIdentity.swift",
            "Qwen38ScorecardWorkerSpawn.swift",
            Self.resolvedAuthorizationFileName,
        ] {
            XCTAssertTrue(
                proofControlNames.contains(required),
                "missing \(required)"
            )
        }

        let runnerFiles = try Self.swiftFiles(in: Self.proofRunnerSources)
        XCTAssertGreaterThanOrEqual(runnerFiles.count, 1)
        XCTAssertTrue(
            runnerFiles.map(\.lastPathComponent).contains("main.swift")
        )

        let fixtureFiles = try Self.swiftFiles(in: Self.fixtureWorkerSources)
        XCTAssertGreaterThanOrEqual(fixtureFiles.count, 1)
        XCTAssertTrue(
            fixtureFiles.map(\.lastPathComponent).contains("main.swift")
        )

        let pairControlFiles = try Self.swiftFiles(
            in: Self.scorecardPairControlSources
        )
        XCTAssertGreaterThanOrEqual(pairControlFiles.count, 6)
    }

    // MARK: - The property

    func testNoSigningAPIInProofControlOrRunnerSources() throws {
        var violations: [String] = []
        for directory in [
            Self.proofControlSources,
            Self.proofRunnerSources,
            Self.fixtureWorkerSources,
            Self.scorecardPairControlSources,
        ] {
            for file in try Self.swiftFiles(in: directory) {
                let content = try String(contentsOf: file, encoding: .utf8)
                for token in Self.forbiddenTokens
                where Self.matches(token, in: content) {
                    violations.append(
                        "\(file.lastPathComponent): \(token)"
                    )
                }
            }
        }
        XCTAssertEqual(violations, [], violations.joined(separator: ", "))
    }

    func testResolvedAuthorizationFileHasNoDecodingConformanceTokens() throws {
        let file = Self.proofControlSources.appendingPathComponent(
            Self.resolvedAuthorizationFileName
        )
        let content = try String(contentsOf: file, encoding: .utf8)
        for token in Self.resolvedAuthorizationForbiddenTokens {
            XCTAssertFalse(
                Self.matches(token, in: content),
                "\(Self.resolvedAuthorizationFileName) contains \(token)"
            )
        }
    }

    // MARK: - Helpers

    private static func matches(_ token: String, in content: String) -> Bool {
        content.range(of: token) != nil
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
