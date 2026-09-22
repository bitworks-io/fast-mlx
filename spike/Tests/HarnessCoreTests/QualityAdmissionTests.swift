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
        tier: String = "Noticeable", headline: String = "About 1 word in 6 differs.",
        hardwareClass: String? = nil, residency: String? = nil
    ) -> QualityCard {
        let config: QualityCard.Config? =
            (hardwareClass != nil || residency != nil)
            ? .init(residency: residency, hardwareClass: hardwareClass) : nil
        return QualityCard(
            id: id,
            model: .init(repo: repo, hfPin: "b04599de"),
            verdict: verdict,
            admission: .init(default: admissionDefault, optIn: optIn, reason: "test fixture"),
            legible: .init(tier: tier, headline: headline),
            config: config)
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

    // MARK: - config.residency (a card measured under SSD expert streaming must never gate a
    // resident Swift `fastmlx-serve` launch — see `QualityCard.matchesResidentLaunch`'s doc comment).

    /// A repo with two cards: an "expert-stream" NO_GO and a resident PASS/NO_GO. `streamingFirst`
    /// controls list order, so the tests prove filtering (not accidental first-match-wins ordering).
    private func residencyManifestJSON(
        streamingFirst: Bool, residentVerdict: String = "PASS"
    ) -> String {
        let streamingCard = """
            {
              "id": "qwen38-27b-optiq-4bit-stream@m3ultra",
              "model": { "repo": "mlx-community/Qwen3.8-27B-OptiQ-4bit", "hfPin": "b04599de" },
              "config": { "quant": { "bits": 4 }, "residency": "expert-stream" },
              "verdict": "NO_GO",
              "admission": { "default": false, "optIn": true, "reason": "streaming changes greedy output" },
              "legible": { "tier": "Noticeable", "headline": "Measured under SSD expert streaming." }
            }
            """
        let residentCard = """
            {
              "id": "qwen38-27b-optiq-4bit-resident@m3ultra",
              "model": { "repo": "mlx-community/Qwen3.8-27B-OptiQ-4bit", "hfPin": "b04599de" },
              "config": { "quant": { "bits": 4 } },
              "verdict": "\(residentVerdict)",
              "admission": { "default": false, "optIn": true, "reason": "resident fixture" },
              "legible": { "tier": "Near-lossless", "headline": "Resident measurement." }
            }
            """
        let cards = streamingFirst ? [streamingCard, residentCard] : [residentCard, streamingCard]
        return """
            {
              "schema": "fast-mlx-quality-card-v1",
              "generatedAt": "2026-09-18T00:00:00Z",
              "cards": [\(cards.joined(separator: ","))]
            }
            """
    }

    func testStoreSkipsStreamingCardListedBeforeResidentCardForSameRepo() {
        let data = Data(residencyManifestJSON(streamingFirst: true).utf8)
        let c = QualityCardStore.card(forRepo: "mlx-community/Qwen3.8-27B-OptiQ-4bit", in: data)
        XCTAssertEqual(
            c?.id, "qwen38-27b-optiq-4bit-resident@m3ultra",
            "a streaming card listed BEFORE the resident card must never win the lookup")
        XCTAssertEqual(c?.verdict, .pass)
    }

    /// The explicit `--quality-cards` serve path decodes the manifest with `loadManifest` and then
    /// selects a card itself; it must apply the same residency rule as the data overload.
    func testLoadedManifestSelectionSkipsStreamingCard() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("quality-residency-\(UUID().uuidString).json")
        try Data(residencyManifestJSON(streamingFirst: true).utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let cards = try QualityCardStore.loadManifest(contentsOf: url)
        XCTAssertEqual(cards.count, 2, "both cards must decode; only selection filters")
        let c = QualityCardStore.card(forRepo: "mlx-community/Qwen3.8-27B-OptiQ-4bit", in: cards)
        XCTAssertEqual(c?.id, "qwen38-27b-optiq-4bit-resident@m3ultra")
    }

    func testStoreReturnsNilWhenOnlyAStreamingCardExistsForRepo() {
        let onlyStreaming = """
            {
              "schema": "fast-mlx-quality-card-v1",
              "generatedAt": "2026-09-18T00:00:00Z",
              "cards": [
                {
                  "id": "qwen38-27b-optiq-4bit-stream@m3ultra",
                  "model": { "repo": "mlx-community/Qwen3.8-27B-OptiQ-4bit", "hfPin": "b04599de" },
                  "config": { "quant": { "bits": 4 }, "residency": "expert-stream" },
                  "verdict": "NO_GO",
                  "admission": { "default": false, "optIn": true, "reason": "streaming changes greedy output" },
                  "legible": { "tier": "Noticeable", "headline": "Measured under SSD expert streaming." }
                }
              ]
            }
            """
        let data = Data(onlyStreaming.utf8)
        let c = QualityCardStore.card(forRepo: "mlx-community/Qwen3.8-27B-OptiQ-4bit", in: data)
        XCTAssertNil(c, "a streaming-only card must never surface to a resident-launch lookup")
        XCTAssertEqual(
            QualityAdmission.decide(card: c, optIn: false), .admitUnmeasured,
            "no resident card found → admit unmeasured, never gated by the streaming card")
    }

    func testStoreReturnsNilForUnrecognizedResidencyString() {
        let bogus = """
            {
              "schema": "fast-mlx-quality-card-v1",
              "generatedAt": "2026-09-18T00:00:00Z",
              "cards": [
                {
                  "id": "qwen38-27b-optiq-4bit-bogus@m3ultra",
                  "model": { "repo": "mlx-community/Qwen3.8-27B-OptiQ-4bit", "hfPin": "b04599de" },
                  "config": { "quant": { "bits": 4 }, "residency": "bogus" },
                  "verdict": "NO_GO",
                  "admission": { "default": false, "optIn": true, "reason": "t" },
                  "legible": { "tier": "Noticeable", "headline": "h" }
                }
              ]
            }
            """
        let data = Data(bogus.utf8)
        let c = QualityCardStore.card(forRepo: "mlx-community/Qwen3.8-27B-OptiQ-4bit", in: data)
        XCTAssertNil(c, "an unrecognized residency string must fail closed, never match a resident launch")
    }

    func testStoreMatchesCardWithExplicitResidentResidency() {
        let explicit = """
            {
              "schema": "fast-mlx-quality-card-v1",
              "generatedAt": "2026-09-18T00:00:00Z",
              "cards": [
                {
                  "id": "qwen38-27b-optiq-4bit-explicit@m3ultra",
                  "model": { "repo": "mlx-community/Qwen3.8-27B-OptiQ-4bit", "hfPin": "b04599de" },
                  "config": { "quant": { "bits": 4 }, "residency": "resident" },
                  "verdict": "PASS",
                  "admission": { "default": false, "optIn": true, "reason": "t" },
                  "legible": { "tier": "Near-lossless", "headline": "h" }
                }
              ]
            }
            """
        let data = Data(explicit.utf8)
        let c = QualityCardStore.card(forRepo: "mlx-community/Qwen3.8-27B-OptiQ-4bit", in: data)
        XCTAssertEqual(c?.id, "qwen38-27b-optiq-4bit-explicit@m3ultra")
    }

    // MARK: - hardwareClass identity/tiebreak (docs/task-inbox/2026-09-22-PREDECLARATION-…): two
    // same-repo, same-residency cards differing only in `config.hardwareClass` are NOT duplicates —
    // the uniqueness key widens to admit them — but `hostHardwareClass` must never FILTER admission,
    // only break a tie deterministically, and only when there IS a tie.

    /// Criterion 7 — THE ORDER TEST. Two same-repo resident cards, different hardwareClass, one PASS
    /// one NO_GO. With no host hint (`hostHardwareClass: nil`), selection must never let manifest
    /// array order decide: both orderings must resolve to the SAME card, and it must be the NO_GO
    /// one (a broken quantization keeps refusing on hardware it was never measured on).
    func testHardwareClassOrderInsensitiveSelectsNoGoRegardlessOfArrayOrder() {
        let passCard = card(
            id: "qwen38-27b-optiq-4bit@apple-m5", verdict: .pass, hardwareClass: "apple-m5")
        let noGoCard = card(
            id: "qwen38-27b-optiq-4bit@apple-m3-ultra", verdict: .noGo, hardwareClass: "apple-m3-ultra")

        let forwardOrder = [passCard, noGoCard]
        let reverseOrder = [noGoCard, passCard]

        let forward = QualityCardStore.card(
            forRepo: "mlx-community/Qwen3.8-27B-OptiQ-4bit", in: forwardOrder)
        let reverse = QualityCardStore.card(
            forRepo: "mlx-community/Qwen3.8-27B-OptiQ-4bit", in: reverseOrder)

        XCTAssertEqual(forward?.id, noGoCard.id, "NO_GO must win regardless of manifest order")
        XCTAssertEqual(reverse?.id, noGoCard.id, "NO_GO must win regardless of manifest order")
        XCTAssertEqual(forward?.id, reverse?.id, "selection must be identical under both orderings")

        // And the end-to-end outcome must actually refuse, not merely "pick a card".
        XCTAssertEqual(
            QualityAdmission.decide(card: forward, optIn: false),
            .refuseQualityFlagged(
                "Noticeable: About 1 word in 6 differs. re-run with --accept-quality \(noGoCard.id) to elect it."
            ))
    }

    /// Criterion 4 — the non-filter invariant. A SINGLE card measured on a different hardware class
    /// than the host must still resolve — hardwareClass never FILTERS, only tiebreaks when there is
    /// more than one candidate.
    func testSingleCardOnDifferentHardwareClassStillResolves() {
        let onlyCard = card(
            id: "qwen38-27b-optiq-4bit@apple-m3-ultra", verdict: .pass, hardwareClass: "apple-m3-ultra")
        let resolved = QualityCardStore.card(
            forRepo: "mlx-community/Qwen3.8-27B-OptiQ-4bit", hostHardwareClass: "apple-m5",
            in: [onlyCard])
        XCTAssertEqual(resolved?.id, onlyCard.id, "a single card must resolve regardless of host class")
    }

    /// The NO_GO form of the non-filter invariant: an Ultra NO_GO card must still refuse on an M5
    /// host — filtering by hardwareClass here would silently disarm the refusal.
    func testSingleNoGoCardOnDifferentHardwareClassStillRefuses() {
        let onlyCard = card(
            id: "qwen38-27b-optiq-4bit@apple-m3-ultra", verdict: .noGo, hardwareClass: "apple-m3-ultra")
        let resolved = QualityCardStore.card(
            forRepo: "mlx-community/Qwen3.8-27B-OptiQ-4bit", hostHardwareClass: "apple-m5",
            in: [onlyCard])
        XCTAssertEqual(resolved?.id, onlyCard.id)
        guard case .refuseQualityFlagged = QualityAdmission.decide(card: resolved, optIn: false) else {
            return XCTFail(
                "an Ultra NO_GO card must still refuse on a non-matching host, not be silently disarmed")
        }
    }

    /// Host-match tiebreak: two candidates, `hostHardwareClass` matches exactly one -> that one wins,
    /// even though it is not the NO_GO card — an exact host match is a stronger signal than the
    /// NO_GO-prefers-safety fallback, which only applies when there is no unique host match.
    func testHostHardwareClassMatchesExactlyOneCandidateSelectsThatOne() {
        let m5Card = card(id: "qwen38-27b-optiq-4bit@apple-m5", verdict: .pass, hardwareClass: "apple-m5")
        let ultraCard = card(
            id: "qwen38-27b-optiq-4bit@apple-m3-ultra", verdict: .pass, hardwareClass: "apple-m3-ultra")
        let resolved = QualityCardStore.card(
            forRepo: "mlx-community/Qwen3.8-27B-OptiQ-4bit", hostHardwareClass: "apple-m5",
            in: [ultraCard, m5Card])
        XCTAssertEqual(resolved?.id, m5Card.id)
    }

    /// A card whose `config` is absent entirely (older card, predates `hardwareClass`) must still
    /// participate safely: nil-safe, never a decode/selection crash, and still order-insensitive
    /// against a card that DOES carry hardwareClass.
    func testCardWithNoConfigAtAllParticipatesSafelyInHardwareClassTiebreak() {
        let noConfigCard = card(id: "qwen38-27b-optiq-4bit@legacy", verdict: .pass)
        XCTAssertNil(noConfigCard.config?.hardwareClass)
        let ultraNoGo = card(
            id: "qwen38-27b-optiq-4bit@apple-m3-ultra", verdict: .noGo, hardwareClass: "apple-m3-ultra")

        let forward = QualityCardStore.card(
            forRepo: "mlx-community/Qwen3.8-27B-OptiQ-4bit", in: [noConfigCard, ultraNoGo])
        let reverse = QualityCardStore.card(
            forRepo: "mlx-community/Qwen3.8-27B-OptiQ-4bit", in: [ultraNoGo, noConfigCard])
        XCTAssertEqual(forward?.id, ultraNoGo.id)
        XCTAssertEqual(reverse?.id, ultraNoGo.id)
    }

    /// `config.hardwareClass` decodes leniently exactly like `residency`: absent or explicit null ->
    /// nil, never a decode failure.
    func testHardwareClassDecodesLenientlyAbsentOrNull() throws {
        let absent = try decodeCard(
            """
            {
              "id": "x@m3ultra",
              "model": { "repo": "mlx-community/x", "hfPin": "abc" },
              "config": { "residency": "resident" },
              "verdict": "PASS",
              "admission": { "default": false, "optIn": true, "reason": "t" },
              "legible": { "tier": "Near-lossless", "headline": "h" }
            }
            """)
        XCTAssertNil(absent.config?.hardwareClass)

        let explicitNull = try decodeCard(
            """
            {
              "id": "x@m3ultra",
              "model": { "repo": "mlx-community/x", "hfPin": "abc" },
              "config": { "residency": "resident", "hardwareClass": null },
              "verdict": "PASS",
              "admission": { "default": false, "optIn": true, "reason": "t" },
              "legible": { "tier": "Near-lossless", "headline": "h" }
            }
            """)
        XCTAssertNil(explicitNull.config?.hardwareClass)

        let present = try decodeCard(
            """
            {
              "id": "x@m3ultra",
              "model": { "repo": "mlx-community/x", "hfPin": "abc" },
              "config": { "residency": "resident", "hardwareClass": "apple-m3-ultra" },
              "verdict": "PASS",
              "admission": { "default": false, "optIn": true, "reason": "t" },
              "legible": { "tier": "Near-lossless", "headline": "h" }
            }
            """)
        XCTAssertEqual(present.config?.hardwareClass, "apple-m3-ultra")
    }

    // MARK: - fail-closed-before-tiebreak (the host-class narrowing step must never outrank a
    // published NO_GO by returning a host-matching PASS card before the NO_GO pool is even
    // consulted; host class may only tiebreak WITHIN whichever verdict pool survives).

    /// T1 — the flip must be unreachable: a host-matching PASS card and a NO_GO measured on other
    /// hardware collide; the OLD ordering returned `hostMatches[0]` (the PASS card) before ever
    /// looking at `.noGo`, so `decide` silently admitted. Pins both the SELECTION and the resulting
    /// ADMISSION outcome, so a resolver-only assertion could not hide a still-broken `decide`.
    func testHostMatchDoesNotOverrideNoGoAcrossDifferentHardware() {
        let passCard = card(id: "qwen38-27b-optiq-4bit@apple-m5", verdict: .pass, hardwareClass: "apple-m5")
        let noGoCard = card(
            id: "qwen38-27b-optiq-4bit@apple-m3-ultra", verdict: .noGo, hardwareClass: "apple-m3-ultra")
        let resolved = QualityCardStore.card(
            forRepo: "mlx-community/Qwen3.8-27B-OptiQ-4bit", hostHardwareClass: "apple-m5",
            in: [passCard, noGoCard])
        XCTAssertEqual(
            resolved?.id, noGoCard.id,
            "fail-closed NO_GO preference must win over an exact host match on a different card")
        guard case .refuseQualityFlagged = QualityAdmission.decide(card: resolved, optIn: false) else {
            return XCTFail(
                "the resolved card must actually refuse -- a selection-only assertion would miss a still-broken decide()"
            )
        }
    }

    /// T2 — anti-vacuity arm: with NO candidate NO_GO, host-class narrowing must still discriminate on
    /// its own. Without this arm, simply deleting the host-class branch entirely would still pass T1
    /// (both cards land in the same NO_GO-only pool there) and the suite would measure nothing about
    /// host-class selection itself.
    func testHostClassStillSelectsAmongAllPassCandidates() {
        let m5Card = card(id: "qwen38-27b-optiq-4bit@apple-m5", verdict: .pass, hardwareClass: "apple-m5")
        let ultraCard = card(
            id: "qwen38-27b-optiq-4bit@apple-m3-ultra", verdict: .pass, hardwareClass: "apple-m3-ultra")
        let resolved = QualityCardStore.card(
            forRepo: "mlx-community/Qwen3.8-27B-OptiQ-4bit", hostHardwareClass: "apple-m5",
            in: [ultraCard, m5Card])
        XCTAssertEqual(
            resolved?.id, m5Card.id, "host class must still discriminate when no candidate is NO_GO")
    }

    /// T3 — host class disambiguates WITHIN the NO_GO pool, not just across it: when every remaining
    /// candidate is already NO_GO, the host match still picks the most relevant one so the
    /// `--accept-quality <id>` refusal names a card actually measured on this box, proving the branch
    /// was subordinated to the fail-closed filter rather than bypassed by it.
    func testHostClassDisambiguatesWithinNoGoPool() {
        let m5NoGo = card(id: "qwen38-27b-optiq-4bit@apple-m5", verdict: .noGo, hardwareClass: "apple-m5")
        let ultraNoGo = card(
            id: "qwen38-27b-optiq-4bit@apple-m3-ultra", verdict: .noGo, hardwareClass: "apple-m3-ultra")
        let resolved = QualityCardStore.card(
            forRepo: "mlx-community/Qwen3.8-27B-OptiQ-4bit", hostHardwareClass: "apple-m5",
            in: [ultraNoGo, m5NoGo])
        XCTAssertEqual(
            resolved?.id, m5NoGo.id,
            "host class must pick the most relevant NO_GO card, not merely any NO_GO card")
    }

    /// T4 — nil host is unchanged: the same collision as T1 with no host hint at all must still
    /// resolve to the NO_GO card exactly as before this fix, so a caller that never passes
    /// hostHardwareClass sees no behavior change from this fix.
    func testNilHostHardwareClassStillSelectsNoGo() {
        let passCard = card(id: "qwen38-27b-optiq-4bit@apple-m5", verdict: .pass, hardwareClass: "apple-m5")
        let noGoCard = card(
            id: "qwen38-27b-optiq-4bit@apple-m3-ultra", verdict: .noGo, hardwareClass: "apple-m3-ultra")
        let resolved = QualityCardStore.card(
            forRepo: "mlx-community/Qwen3.8-27B-OptiQ-4bit", hostHardwareClass: nil,
            in: [passCard, noGoCard])
        XCTAssertEqual(resolved?.id, noGoCard.id)
    }

    // MARK: - QualityCard.matchesResidentLaunch / effectiveResidency (decode-level, independent of
    // QualityCardStore, to isolate decode leniency from the store's filtering).

    private func decodeCard(_ json: String) throws -> QualityCard {
        try JSONDecoder().decode(QualityCard.self, from: Data(json.utf8))
    }

    func testMissingConfigKeyDefaultsToResident() throws {
        let card = try decodeCard(
            """
            {
              "id": "x@m3ultra",
              "model": { "repo": "mlx-community/x", "hfPin": "abc" },
              "verdict": "PASS",
              "admission": { "default": false, "optIn": true, "reason": "t" },
              "legible": { "tier": "Near-lossless", "headline": "h" }
            }
            """)
        XCTAssertEqual(card.effectiveResidency, "resident")
        XCTAssertTrue(card.matchesResidentLaunch)
    }

    func testNullResidencyDefaultsToResident() throws {
        let card = try decodeCard(
            """
            {
              "id": "x@m3ultra",
              "model": { "repo": "mlx-community/x", "hfPin": "abc" },
              "config": { "residency": null },
              "verdict": "PASS",
              "admission": { "default": false, "optIn": true, "reason": "t" },
              "legible": { "tier": "Near-lossless", "headline": "h" }
            }
            """)
        XCTAssertEqual(card.effectiveResidency, "resident")
        XCTAssertTrue(card.matchesResidentLaunch)
    }

    func testUnrecognizedResidencyDecodesWithoutThrowingButFailsClosed() throws {
        let card = try decodeCard(
            """
            {
              "id": "x@m3ultra",
              "model": { "repo": "mlx-community/x", "hfPin": "abc" },
              "config": { "residency": "bogus" },
              "verdict": "NO_GO",
              "admission": { "default": false, "optIn": true, "reason": "t" },
              "legible": { "tier": "Noticeable", "headline": "h" }
            }
            """)
        XCTAssertEqual(card.effectiveResidency, "bogus")
        XCTAssertFalse(card.matchesResidentLaunch)
    }

    func testExpertStreamResidencyFailsToMatchResidentLaunch() throws {
        let card = try decodeCard(
            """
            {
              "id": "x@m3ultra",
              "model": { "repo": "mlx-community/x", "hfPin": "abc" },
              "config": { "residency": "expert-stream" },
              "verdict": "NO_GO",
              "admission": { "default": false, "optIn": true, "reason": "t" },
              "legible": { "tier": "Noticeable", "headline": "h" }
            }
            """)
        XCTAssertEqual(card.effectiveResidency, "expert-stream")
        XCTAssertFalse(card.matchesResidentLaunch)
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
