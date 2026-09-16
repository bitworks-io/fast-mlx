import XCTest
@testable import HarnessCore

/// TDD for the quality-guidance moat's Swift-side admission gate (`docs/quality-card-schema-v1.md`,
/// "Admission discriminator rules"). Pins the full 5-row truth table as a pure function — no I/O, no
/// model load — plus the manifest loader's fail-closed-to-"no card" behavior and the additive
/// `--accept-quality` CLI parser.
final class QualityAdmissionTests: XCTestCase {

    // MARK: - fixtures

    private func card(
        id: String = "qwen38-27b-optiq-4bit@m3ultra", repo: String = "mlx-community/Qwen3.8-27B-OptiQ-4bit",
        verdict: QualityVerdict, admissionDefault: Bool = false, optIn: Bool = true,
        tier: String = "Noticeable", headline: String = "About 1 word in 6 differs."
    ) -> QualityCard {
        QualityCard(
            id: id,
            model: .init(repo: repo, hfPin: "b04599de"),
            verdict: verdict,
            admission: .init(default: admissionDefault, optIn: optIn, reason: "test fixture"),
            legible: .init(tier: tier, headline: headline))
    }

    // MARK: - discriminator truth table (docs/quality-card-schema-v1.md)

    func testNoCardAdmitsUnmeasuredNeverRefuses() {
        let outcome = QualityAdmission.decide(card: nil, optIn: false)
        XCTAssertEqual(outcome, .admitUnmeasured)
    }

    func testNoCardAdmitsUnmeasuredRegardlessOfOptIn() {
        // optIn is meaningless with no card; must still never refuse.
        XCTAssertEqual(QualityAdmission.decide(card: nil, optIn: true), .admitUnmeasured)
    }

    func testUnmeasuredVerdictCardAdmitsUnmeasured() {
        let c = card(verdict: .unmeasured)
        XCTAssertEqual(QualityAdmission.decide(card: c, optIn: false), .admitUnmeasured)
    }

    func testPassVerdictAdmitsSilently() {
        let c = card(verdict: .pass)
        XCTAssertEqual(QualityAdmission.decide(card: c, optIn: false), .admit)
    }

    func testReferenceVerdictAdmitsSilently() {
        let c = card(verdict: .reference)
        XCTAssertEqual(QualityAdmission.decide(card: c, optIn: false), .admit)
    }

    func testExactVerdictAdmitsSilently() {
        let c = card(verdict: .exact)
        XCTAssertEqual(QualityAdmission.decide(card: c, optIn: false), .admit)
    }

    func testNoGoWithoutOptInRefusesWithQualityFlaggedMessage() {
        let c = card(
            verdict: .noGo, tier: "Noticeable", headline: "About 1 word in 6 differs from the 8-bit model.")
        let outcome = QualityAdmission.decide(card: c, optIn: false)
        guard case .refuseQualityFlagged(let message) = outcome else {
            return XCTFail("expected .refuseQualityFlagged, got \(outcome)")
        }
        XCTAssertTrue(message.contains("Noticeable"), "message should carry the legible tier: \(message)")
        XCTAssertTrue(
            message.contains("About 1 word in 6 differs from the 8-bit model."),
            "message should carry the legible headline: \(message)")
        XCTAssertTrue(
            message.contains("--accept-quality"), "refusal must say how to opt in: \(message)")
        XCTAssertTrue(message.contains(c.id), "refusal should name the card id to elect: \(message)")
    }

    func testNoGoWithOptInAdmitsWithQualityFlagMessage() {
        let c = card(verdict: .noGo, tier: "Significant", headline: "About 1 word in 6 differs (drift 5.6).")
        let outcome = QualityAdmission.decide(card: c, optIn: true)
        guard case .admitWithQualityFlag(let message) = outcome else {
            return XCTFail("expected .admitWithQualityFlag, got \(outcome)")
        }
        XCTAssertTrue(message.contains("Significant"))
        XCTAssertTrue(message.contains("About 1 word in 6 differs (drift 5.6)."))
        // The flag message (unlike the refusal) does not need to re-explain how to opt in — the
        // operator already opted in.
    }

    func testDiscriminatorKeysOnNoGoSpecificallyNotOnAbsenceOfPassingCard() {
        // A PASS card must not be treated the same as "no NO_GO row found" in some other sense —
        // confirm every non-NO_GO verdict silently admits, and only NO_GO ever produces a
        // flag/refusal outcome.
        for verdict: QualityVerdict in [.pass, .reference, .exact, .unmeasured] {
            let c = card(verdict: verdict)
            let outcome = QualityAdmission.decide(card: c, optIn: false)
            switch outcome {
            case .admit, .admitUnmeasured:
                break
            default:
                XCTFail("verdict \(verdict) must never produce a flag/refusal outcome, got \(outcome)")
            }
        }
    }

    // MARK: - QualityVerdict fail-closed decode

    func testUnknownVerdictStringDecodesToUnmeasuredNotACrash() throws {
        let json = Data("\"SOMETHING_NEW_FROM_A_FUTURE_SCHEMA\"".utf8)
        let decoded = try JSONDecoder().decode(QualityVerdict.self, from: json)
        XCTAssertEqual(decoded, .unmeasured)
    }

    // MARK: - QualityCardStore

    private let manifestJSON = """
    {
      "schema": "fast-mlx-quality-card-v1",
      "generatedAt": "2026-09-02T20:35:47Z",
      "cards": [
        {
          "id": "qwen38-27b-optiq-4bit@m3ultra",
          "model": { "family": "Qwen3.8-27B", "repo": "mlx-community/Qwen3.8-27B-OptiQ-4bit", "hfPin": "b04599de" },
          "config": { "quant": { "bits": 4 }, "enhancement": "none", "hardwareClass": "apple-m3-ultra" },
          "verdict": "NO_GO",
          "admission": { "default": false, "optIn": true, "reason": "quality-degraded vs 8-bit reference" },
          "legible": { "tier": "Noticeable", "headline": "About 1 word in 6 differs." }
        },
        {
          "id": "qwen38-27b-8bit-reference@m3ultra",
          "model": { "family": "Qwen3.8-27B", "repo": "mlx-community/Qwen3.8-27B-mxfp8", "hfPin": "abc12345" },
          "config": { "quant": { "bits": 8 }, "enhancement": "none", "hardwareClass": "apple-m3-ultra" },
          "verdict": "REFERENCE",
          "admission": { "default": true, "optIn": true, "reason": "measurement reference" },
          "legible": { "tier": "Near-lossless", "headline": "Indistinguishable in normal use." }
        }
      ]
    }
    """

    func testStoreFindsCardForKnownRepo() {
        let data = Data(manifestJSON.utf8)
        let c = QualityCardStore.card(forRepo: "mlx-community/Qwen3.8-27B-OptiQ-4bit", in: data)
        XCTAssertEqual(c?.id, "qwen38-27b-optiq-4bit@m3ultra")
        XCTAssertEqual(c?.verdict, .noGo)
    }

    func testStoreFindsSecondCardForKnownRepo() {
        let data = Data(manifestJSON.utf8)
        let c = QualityCardStore.card(forRepo: "mlx-community/Qwen3.8-27B-mxfp8", in: data)
        XCTAssertEqual(c?.id, "qwen38-27b-8bit-reference@m3ultra")
        XCTAssertEqual(c?.verdict, .reference)
    }

    func testStoreReturnsNilForUnknownRepo() {
        let data = Data(manifestJSON.utf8)
        let c = QualityCardStore.card(forRepo: "mlx-community/some-other-repo", in: data)
        XCTAssertNil(c)
    }

    func testStoreReturnsNilOnMalformedJSON() {
        let data = Data("{ this is not valid json".utf8)
        let c = QualityCardStore.card(forRepo: "mlx-community/Qwen3.8-27B-OptiQ-4bit", in: data)
        XCTAssertNil(c)
    }

    func testStoreReturnsNilForMissingManifestFile() {
        let missingURL = URL(fileURLWithPath: "/nonexistent/path/quality-guides-\(UUID()).json")
        let c = QualityCardStore.card(forRepo: "mlx-community/Qwen3.8-27B-OptiQ-4bit", manifestURL: missingURL)
        XCTAssertNil(c)
    }

    func testStoreReadsCardFromRealFile() throws {
        let dir = FileManager.default.temporaryDirectory
        let url = dir.appendingPathComponent("quality-guides-\(UUID()).json")
        try Data(manifestJSON.utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let c = QualityCardStore.card(forRepo: "mlx-community/Qwen3.8-27B-OptiQ-4bit", manifestURL: url)
        XCTAssertEqual(c?.id, "qwen38-27b-optiq-4bit@m3ultra")
    }

    // MARK: - end-to-end: store → decide

    func testEndToEndNoGoCardRefusesWithoutOptIn() {
        let data = Data(manifestJSON.utf8)
        let c = QualityCardStore.card(forRepo: "mlx-community/Qwen3.8-27B-OptiQ-4bit", in: data)
        let outcome = QualityAdmission.decide(card: c, optIn: false)
        guard case .refuseQualityFlagged = outcome else {
            return XCTFail("expected refusal, got \(outcome)")
        }
    }

    func testEndToEndReferenceCardAdmits() {
        let data = Data(manifestJSON.utf8)
        let c = QualityCardStore.card(forRepo: "mlx-community/Qwen3.8-27B-mxfp8", in: data)
        XCTAssertEqual(QualityAdmission.decide(card: c, optIn: false), .admit)
    }

    // MARK: - QualityOptIn CLI parser (additive, mirrors QuantPickPreference.validated's idiom)

    func testOptInParsesAcceptQualityFlag() {
        let optIn = QualityOptIn.parse(["--tier", "balanced", "--accept-quality", "qwen38-27b-optiq-4bit@m3ultra"])
        XCTAssertTrue(optIn.isElected(cardID: "qwen38-27b-optiq-4bit@m3ultra", repoID: nil))
    }

    func testOptInMatchesByRepoIDToo() {
        let optIn = QualityOptIn.parse(["--accept-quality", "mlx-community/Qwen3.8-27B-OptiQ-4bit"])
        XCTAssertTrue(optIn.isElected(cardID: nil, repoID: "mlx-community/Qwen3.8-27B-OptiQ-4bit"))
    }

    func testOptInAbsentFlagIsNotElectedForAnything() {
        let optIn = QualityOptIn.parse(["--tier", "balanced", "--context", "8192"])
        XCTAssertFalse(optIn.isElected(cardID: "anything", repoID: "mlx-community/anything"))
        XCTAssertEqual(optIn, QualityOptIn())
    }

    func testOptInDoesNotRejectAnyValueSilentlyIgnoredMeansNotSet() {
        // No invalid VALUE exists for this flag (any string is a valid id/repo to elect); a
        // dangling flag with no following value is simply not recorded, rather than crashing or
        // silently consuming the next unrelated flag.
        let optIn = QualityOptIn.parse(["--accept-quality"])
        XCTAssertEqual(optIn, QualityOptIn())
    }

    func testOptInCollectsMultipleOccurrences() {
        let optIn = QualityOptIn.parse(["--accept-quality", "card-a", "--accept-quality", "card-b"])
        XCTAssertTrue(optIn.isElected(cardID: "card-a", repoID: nil))
        XCTAssertTrue(optIn.isElected(cardID: "card-b", repoID: nil))
        XCTAssertFalse(optIn.isElected(cardID: "card-c", repoID: nil))
    }

    // MARK: - QualityCardsManifestResolver (defect fix: a default lookup relative to the process
    // CWD silently ran with no gate when launched from another directory, and an explicit
    // --quality-cards typo failed OPEN instead of refusing). Pure, unit-testable: `cwd` and
    // `fileExists` are injected so no test depends on the real filesystem or working directory.

    private let injectedCWD = URL(fileURLWithPath: "/srv/fastmlx", isDirectory: true)

    func testResolverExplicitPathMissingThrows() {
        XCTAssertThrowsError(
            try QualityCardsManifestResolver.resolve(
                explicitPath: "/nonexistent/quality-guides.json",
                cwd: injectedCWD,
                fileExists: { _ in false })
        ) { error in
            guard let notFound = error as? QualityCardsManifestPathNotFound else {
                return XCTFail("expected QualityCardsManifestPathNotFound, got \(error)")
            }
            XCTAssertEqual(notFound.requestedPath, "/nonexistent/quality-guides.json")
            XCTAssertEqual(notFound.resolvedPath, "/nonexistent/quality-guides.json")
        }
    }

    func testResolverExplicitAbsolutePathPresentResolvesToItself() throws {
        let resolution = try QualityCardsManifestResolver.resolve(
            explicitPath: "/opt/cards/quality-guides.json",
            cwd: injectedCWD,
            fileExists: { $0.path == "/opt/cards/quality-guides.json" })
        guard case .active(let path, let explicit) = resolution else {
            return XCTFail("expected .active, got \(resolution)")
        }
        XCTAssertEqual(path.path, "/opt/cards/quality-guides.json")
        XCTAssertTrue(explicit)
    }

    func testResolverExplicitRelativePathIsAbsolutizedAgainstInjectedCWD() throws {
        let resolution = try QualityCardsManifestResolver.resolve(
            explicitPath: "cards/quality-guides.json",
            cwd: injectedCWD,
            fileExists: { _ in true })
        guard case .active(let path, let explicit) = resolution else {
            return XCTFail("expected .active, got \(resolution)")
        }
        XCTAssertEqual(path.path, "/srv/fastmlx/cards/quality-guides.json")
        XCTAssertTrue(explicit)
    }

    func testResolverDefaultAbsentIsInactive() throws {
        let resolution = try QualityCardsManifestResolver.resolve(
            explicitPath: nil,
            cwd: injectedCWD,
            fileExists: { _ in false })
        XCTAssertEqual(resolution, .inactive)
    }

    func testResolverDefaultPresentResolvesAbsoluteAndNotExplicit() throws {
        let resolution = try QualityCardsManifestResolver.resolve(
            explicitPath: nil,
            cwd: injectedCWD,
            fileExists: { $0.path == "/srv/fastmlx/site/quality-guides.json" })
        guard case .active(let path, let explicit) = resolution else {
            return XCTFail("expected .active, got \(resolution)")
        }
        XCTAssertEqual(path.path, "/srv/fastmlx/site/quality-guides.json")
        XCTAssertFalse(explicit)
    }

    // MARK: - QualityCardStore.loadManifest(contentsOf:) — strict decode for an explicit path,
    // unlike card(forRepo:manifestURL:)'s silent-nil fail-open convenience above.

    func testLoadManifestThrowsOnMissingFile() {
        let missingURL = URL(fileURLWithPath: "/nonexistent/path/quality-guides-\(UUID()).json")
        XCTAssertThrowsError(try QualityCardStore.loadManifest(contentsOf: missingURL))
    }

    func testLoadManifestThrowsOnMalformedJSON() throws {
        let dir = FileManager.default.temporaryDirectory
        let url = dir.appendingPathComponent("quality-guides-\(UUID()).json")
        try Data("{ not valid json".utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertThrowsError(try QualityCardStore.loadManifest(contentsOf: url))
    }

    func testLoadManifestReturnsCardsOnValidFile() throws {
        let dir = FileManager.default.temporaryDirectory
        let url = dir.appendingPathComponent("quality-guides-\(UUID()).json")
        try Data(manifestJSON.utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let cards = try QualityCardStore.loadManifest(contentsOf: url)
        XCTAssertTrue(cards.contains { $0.id == "qwen38-27b-optiq-4bit@m3ultra" })
    }
}
