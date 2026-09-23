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
        XCTAssertTrue(optIn.isElected(cardID: "qwen38-27b-optiq-4bit@m3ultra", repoID: nil, hfPin: nil))
    }

    func testOptInMatchesByRepoIDToo() {
        let optIn = QualityOptIn.parse(["--accept-quality", "mlx-community/Qwen3.8-27B-OptiQ-4bit"])
        XCTAssertTrue(
            optIn.isElected(cardID: nil, repoID: "mlx-community/Qwen3.8-27B-OptiQ-4bit", hfPin: nil))
    }

    func testOptInAbsentFlagIsNotElectedForAnything() {
        let optIn = QualityOptIn.parse(["--tier", "balanced", "--context", "8192"])
        XCTAssertFalse(
            optIn.isElected(cardID: "anything", repoID: "mlx-community/anything", hfPin: nil))
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
        XCTAssertTrue(optIn.isElected(cardID: "card-a", repoID: nil, hfPin: nil))
        XCTAssertTrue(optIn.isElected(cardID: "card-b", repoID: nil, hfPin: nil))
        XCTAssertFalse(optIn.isElected(cardID: "card-c", repoID: nil, hfPin: nil))
    }

    // MARK: - QualityOptIn.isElected(card:) escape hatch for a card with no `model.repo` (mirrors
    // Python `is_opted_in`, `scripts/fastmlx_launch.py:1099-1109`, which elects by id, repo, OR
    // hfPin). Real pin string taken from a shipped NO_GO card whose `model.repo` is nil.

    func testOptInElectsRepolessCardByHfPin() {
        // A1: a NO_GO card with model.repo == nil, elected via its hfPin, must admit with the
        // quality flag rather than refuse -- the escape hatch this increment adds.
        let card = QualityCard(
            id: "repoless-card@m3ultra",
            model: .init(repo: nil, hfPin: "ef5b919d31534faa1997666f1a22d362cd6383cd"),
            verdict: .noGo,
            admission: .init(default: false, optIn: true, reason: "test fixture"),
            legible: .init(tier: "Noticeable", headline: "h"))
        let optIn = QualityOptIn.parse([
            "--accept-quality", "ef5b919d31534faa1997666f1a22d362cd6383cd",
        ])
        XCTAssertTrue(optIn.isElected(card: card))
        let outcome = QualityAdmission.decide(card: card, optIn: optIn.isElected(card: card))
        XCTAssertEqual(outcome, .admitWithQualityFlag("Noticeable: h"))
    }

    func testOptInElectsRepolessCardByCardIDToo() {
        // A2: the same repoless card is still elected by its card id (no regression from adding
        // the hfPin arm).
        let card = QualityCard(
            id: "repoless-card@m3ultra",
            model: .init(repo: nil, hfPin: "ef5b919d31534faa1997666f1a22d362cd6383cd"),
            verdict: .noGo,
            admission: .init(default: false, optIn: true, reason: "test fixture"),
            legible: .init(tier: "Noticeable", headline: "h"))
        let optIn = QualityOptIn.parse(["--accept-quality", "repoless-card@m3ultra"])
        XCTAssertTrue(optIn.isElected(card: card))
    }

    func testOptInElectsCardWithRepoByRepoIDStillWorks() {
        // A3: a card WITH a repo is still elected by its repo id -- no regression from the widened
        // election logic.
        let cardWithRepo = card(verdict: .noGo)
        let optIn = QualityOptIn.parse(["--accept-quality", "mlx-community/Qwen3.8-27B-OptiQ-4bit"])
        XCTAssertTrue(optIn.isElected(card: cardWithRepo))
    }

    func testOptInUnrelatedValueElectsNeitherCard() {
        // A4: an unrelated accepted value elects none of id/repo/hfPin -- the widened set did not
        // become permissive-by-default.
        let repolessCard = QualityCard(
            id: "repoless-card@m3ultra",
            model: .init(repo: nil, hfPin: "ef5b919d31534faa1997666f1a22d362cd6383cd"),
            verdict: .noGo,
            admission: .init(default: false, optIn: true, reason: "test fixture"),
            legible: .init(tier: "Noticeable", headline: "h"))
        let cardWithRepo = card(verdict: .noGo)
        let optIn = QualityOptIn.parse(["--accept-quality", "something-else"])
        XCTAssertFalse(optIn.isElected(card: repolessCard))
        XCTAssertFalse(optIn.isElected(card: cardWithRepo))
        XCTAssertEqual(
            QualityAdmission.decide(card: repolessCard, optIn: optIn.isElected(card: repolessCard)),
            .refuseQualityFlagged(
                "Noticeable: h re-run with --accept-quality repoless-card@m3ultra to elect it."))
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

    // MARK: - QualityCardStore.canonicalHardwareClass (docs/task-inbox/2026-09-22-PREDECLARATION-
    // swift-serve-never-learns-its-host-class.md, criteria 1–4) — pure, mirrors
    // `fastmlx_launch.host_hardware_class()`'s normalization generalized the way
    // `emit_quality_card.validate_hardware_class` does.

    /// Criterion 1: the exact examples Python's docstring names.
    func testCanonicalHardwareClassNormalizesAppleM3Ultra() {
        XCTAssertEqual(
            QualityCardStore.canonicalHardwareClass(fromChipBrand: "Apple M3 Ultra"), "apple-m3-ultra")
    }

    func testCanonicalHardwareClassNormalizesAppleM5() {
        XCTAssertEqual(QualityCardStore.canonicalHardwareClass(fromChipBrand: "Apple M5"), "apple-m5")
    }

    /// Criterion 2: nil in -> nil out; empty/whitespace-only -> nil after trimming.
    func testCanonicalHardwareClassNilForNilEmptyAndWhitespaceOnly() {
        XCTAssertNil(QualityCardStore.canonicalHardwareClass(fromChipBrand: nil))
        XCTAssertNil(QualityCardStore.canonicalHardwareClass(fromChipBrand: ""))
        XCTAssertNil(QualityCardStore.canonicalHardwareClass(fromChipBrand: "   "))
    }

    /// Criterion 3: the chip-probe-failed sentinel is refused case-insensitively — compared AFTER
    /// lowercasing, so "UNKNOWN" / " Unknown " are caught too, not only the exact lowercase spelling.
    func testCanonicalHardwareClassNilForSentinelCaseInsensitive() {
        XCTAssertNil(QualityCardStore.canonicalHardwareClass(fromChipBrand: "unknown"))
        XCTAssertNil(QualityCardStore.canonicalHardwareClass(fromChipBrand: "UNKNOWN"))
        XCTAssertNil(QualityCardStore.canonicalHardwareClass(fromChipBrand: " Unknown "))
    }

    /// Criterion 4: the `hw.model` fallback shape (e.g. `"Mac17,3"`) must not be silently laundered
    /// into something that could match a card — normalization only trims/lowercases/hyphenates
    /// whitespace, it never strips punctuation, so the comma (and hence the non-canonical shape)
    /// survives verbatim.
    func testCanonicalHardwareClassDoesNotLaunderHwModelFallbackShape() {
        let result = QualityCardStore.canonicalHardwareClass(fromChipBrand: "Mac17,3")
        XCTAssertEqual(result, "mac17,3")
        XCTAssertTrue(
            result?.contains(",") ?? false,
            "the comma must survive normalization — it is what makes this shape non-canonical")
        XCTAssertNotEqual(result, "apple-m3-ultra", "must not collide with a plausible class")
        XCTAssertNotEqual(result, "apple-m5", "must not collide with a plausible class")
    }

    // MARK: - hostHardwareClass wiring into card resolution (criteria 5–6, 9) — proves the
    // `hostHardwareClass:` parameter actually discriminates, not merely that it type-checks.

    /// Criterion 5 — THE discrimination test. Two cards, same repo, same (non-NO_GO) verdict,
    /// differing only in `config.hardwareClass`. Card ids are chosen so the LEXICOGRAPHICALLY-FIRST
    /// id ("alpha-ultra") is the one whose hardwareClass is NOT the host we pass — so a broken
    /// implementation that silently ignored `hostHardwareClass` and always fell through to step 4
    /// (lexicographic-by-id) would return the WRONG card on the host-supplied arm, and this test
    /// would catch it. Both arms are asserted so the wiring cannot pass vacuously.
    func testHostHardwareClassDiscriminatesBetweenTwoAdmittingCards() {
        let ultraCard = card(id: "alpha-ultra", verdict: .pass, hardwareClass: "apple-m3-ultra")
        let m5Card = card(id: "zulu-m5", verdict: .pass, hardwareClass: "apple-m5")
        let cards = [ultraCard, m5Card]

        let withHost = QualityCardStore.card(
            forRepo: "mlx-community/Qwen3.8-27B-OptiQ-4bit", hostHardwareClass: "apple-m5", in: cards)
        XCTAssertEqual(
            withHost?.id, m5Card.id,
            "hostHardwareClass: \"apple-m5\" must select the card measured on apple-m5")

        let withoutHost = QualityCardStore.card(
            forRepo: "mlx-community/Qwen3.8-27B-OptiQ-4bit", hostHardwareClass: nil, in: cards)
        XCTAssertEqual(
            withoutHost?.id, ultraCard.id,
            "hostHardwareClass: nil must fall back to the lexicographically-first card id")
    }

    /// Criterion 6: fail-closed still outranks host match — a NO_GO card on a NON-matching host must
    /// still be returned over an admitting card that DOES match the host.
    func testFailClosedNoGoOutranksHostMatchingAdmittingCard() {
        let noGoOffHost = card(id: "no-go-off-host", verdict: .noGo, hardwareClass: "apple-m3-ultra")
        let admitOnHost = card(id: "admit-on-host", verdict: .pass, hardwareClass: "apple-m5")
        let resolved = QualityCardStore.card(
            forRepo: "mlx-community/Qwen3.8-27B-OptiQ-4bit", hostHardwareClass: "apple-m5",
            in: [noGoOffHost, admitOnHost])
        XCTAssertEqual(
            resolved?.id, noGoOffHost.id,
            "a NO_GO card must outrank a host-matching admitting card — fail-closed stays ahead of the tiebreak")
    }

    /// Criterion 9: single-card resolution is unchanged byte-for-byte — a repo with exactly one
    /// resident match returns it regardless of whether the host class matches, is nil, or differs.
    func testSingleCardResolutionUnchangedRegardlessOfHostHardwareClass() {
        let onlyCard = card(id: "solo-card", verdict: .pass, hardwareClass: "apple-m3-ultra")
        for host: String? in ["apple-m3-ultra", "apple-m5", nil] {
            let resolved = QualityCardStore.card(
                forRepo: "mlx-community/Qwen3.8-27B-OptiQ-4bit", hostHardwareClass: host, in: [onlyCard])
            XCTAssertEqual(
                resolved?.id, onlyCard.id,
                "single match must resolve regardless of host (host=\(String(describing: host)))")
        }
    }

    // MARK: - call-site pin (criterion 7) — `FastMLXServe.swift`'s admission call must pass
    // `hostHardwareClass:`. Reads the real source file off disk (not `@testable import` — the
    // parameter's PRESENCE at the call site is what's being pinned, which only source text proves),
    // comment-stripped so a comment mentioning the symbol cannot satisfy the assertion.
    //
    // Updated for Task B/C: the call site now goes through `QualityCardStore.resolve(repo:revision:
    // hostHardwareClass:in:)` rather than `card(forRepo:hostHardwareClass:in:)` directly — this test
    // is re-pinned to the new call shape rather than left checking a call that no longer exists.

    private static let fastMLXServeRelativePath = "spike/Sources/fastmlx-serve/FastMLXServe.swift"

    /// Walks parent directories upward from this test file's own location looking for
    /// `spike/Sources/fastmlx-serve/FastMLXServe.swift`, rather than hand-counting `../` levels from
    /// `#filePath` — so the pin survives a different checkout/worktree layout.
    private func locateFastMLXServeSwift() -> URL? {
        var dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        while true {
            let candidate = dir.appendingPathComponent(Self.fastMLXServeRelativePath)
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
            let parent = dir.deletingLastPathComponent()
            if parent.path == dir.path {
                return nil
            }
            dir = parent
        }
    }

    func testCallSitePassesHostHardwareClassToAdmission() throws {
        guard let url = locateFastMLXServeSwift() else {
            XCTFail(
                "could not locate \(Self.fastMLXServeRelativePath) by walking up from #filePath — checkout layout may have changed"
            )
            return
        }
        let source = try String(contentsOf: url, encoding: .utf8)
        // Strip `//` line comments so a comment mentioning the symbol cannot satisfy this assertion.
        let stripped =
            source
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { line -> Substring in
                if let range = line.range(of: "//") {
                    return line[line.startIndex..<range.lowerBound]
                }
                return line
            }
            .joined(separator: "\n")
        let pattern =
            "QualityCardStore\\.resolve\\(\\s*repo:\\s*model,\\s*revision:[^,]+,\\s*hostHardwareClass:"
        XCTAssertNotNil(
            stripped.range(of: pattern, options: .regularExpression),
            "the admission call site (QualityCardStore.resolve(repo: model, revision:, ...)) must pass hostHardwareClass: — comment-stripped source did not match"
        )
    }

    // MARK: - QualityCardStore.resolve(repo:revision:hostHardwareClass:in:) — hfPin-prefix lookup +
    // ambiguity (Task B), mirroring Python `resolve_card` (`scripts/fastmlx_launch.py:952-1082`) and
    // its helpers `_is_full_hex_revision`/`_is_usable_hf_pin`/`_hf_pin_matches_revision`/
    // `find_cards_by_pin`/`find_cards_by_repo` exactly.

    private func repolessCard(
        id: String = "repoless-card@m3ultra",
        hfPin: String?,
        verdict: QualityVerdict = .noGo,
        hardwareClass: String? = nil,
        residency: String? = nil
    ) -> QualityCard {
        let config: QualityCard.Config? =
            (hardwareClass != nil || residency != nil)
            ? .init(residency: residency, hardwareClass: hardwareClass) : nil
        return QualityCard(
            id: id,
            model: .init(repo: nil, hfPin: hfPin),
            verdict: verdict,
            admission: .init(default: false, optIn: true, reason: "test fixture"),
            legible: .init(tier: "Noticeable", headline: "h"),
            config: config)
    }

    /// B1: a card with `repo: nil`, pin `ef5b919d31534faa1997666f1a22d362cd6383cd` (the real shipped
    /// repoless NO_GO card's pin — `site/quality-guides.json`), resolves from that exact 40-hex
    /// revision (a trivial prefix match: the pin IS the full revision here).
    func testResolveB1RepolessCardResolvesFromExact40HexRevision() {
        let c = repolessCard(hfPin: "ef5b919d31534faa1997666f1a22d362cd6383cd")
        let resolution = QualityCardStore.resolve(
            repo: nil, revision: "ef5b919d31534faa1997666f1a22d362cd6383cd", in: [c])
        guard case .resolved(let resolved) = resolution else {
            return XCTFail("expected .resolved, got \(resolution)")
        }
        XCTAssertEqual(resolved.id, c.id)
    }

    /// B2: an 8-hex pin resolves from a 40-hex revision it prefixes.
    func testResolveB2ShortPinResolvesFromPrefixedFullRevision() {
        let c = repolessCard(hfPin: "73e3e38d", verdict: .pass)
        let resolution = QualityCardStore.resolve(
            repo: nil, revision: "73e3e38d981303bc594367cd910ea6eb48349da8", in: [c])
        guard case .resolved(let resolved) = resolution else {
            return XCTFail("expected .resolved, got \(resolution)")
        }
        XCTAssertEqual(resolved.id, c.id)
    }

    /// B3: a revision that is not full 40-hex resolves nothing, whether it is too short or non-hex.
    func testResolveB3NonFullHexRevisionResolvesNothing() {
        let c = repolessCard(hfPin: "73e3e38d", verdict: .pass)
        for revision in ["73e3e38d", "main"] {
            let resolution = QualityCardStore.resolve(repo: nil, revision: revision, in: [c])
            XCTAssertEqual(
                resolution, .none, "revision \(revision) is not full 40-hex, must resolve nothing")
        }
    }

    /// B4: a pin shorter than 8 hex chars never matches, even from a 40-hex revision it prefixes.
    func testResolveB4TooShortPinResolvesNothing() {
        let c = repolessCard(hfPin: "73e3", verdict: .pass)
        let resolution = QualityCardStore.resolve(
            repo: nil, revision: "73e3e38d981303bc594367cd910ea6eb48349da8", in: [c])
        XCTAssertEqual(resolution, .none)
    }

    /// B5: repo lookup and pin lookup naming DIFFERENT cards is ambiguous — neither `.resolved` nor
    /// `.none`.
    func testResolveB5RepoAndPinNamingDifferentCardsIsAmbiguous() {
        let repoCard = card(id: "repo-card@m3ultra", verdict: .pass)
        let pinCard = repolessCard(id: "pin-card@m3ultra", hfPin: "abcdefab", verdict: .pass)
        // 40-hex, prefixed by pinCard's "abcdefab" pin.
        let revision = "abcdefab0123456789abcdef0123456789abcdef"
        let resolution = QualityCardStore.resolve(
            repo: repoCard.model.repo, revision: revision, in: [repoCard, pinCard])
        switch resolution {
        case .resolved, .none:
            XCTFail("expected .ambiguous, got \(resolution)")
        case .ambiguous(let repoCardIDs, let pinCardIDs):
            XCTAssertEqual(repoCardIDs, [repoCard.id])
            XCTAssertEqual(pinCardIDs, [pinCard.id])
        }
    }

    /// B6: two cards sharing pin `73e3e38d` but differing in `config.hardwareClass` (real shipped
    /// ids: `qwen3-0p6b-4bit@m3ultra` / `qwen3-0p6b-4bit@m5`), resolved by pin with
    /// `hostHardwareClass: "apple-m5"`, must return the `@m5` card. This is discriminating: the ids
    /// sort `"...@m3ultra" < "...@m5"` lexicographically, so a lexicographic fallback (step 4, if the
    /// hardwareClass tiebreak (step 3) were skipped) would return the WRONG (m3ultra) card.
    func testResolveB6PinPoolHardwareClassTiebreakIsNotLexicographic() {
        let m3ultraCard = repolessCard(
            id: "qwen3-0p6b-4bit@m3ultra", hfPin: "73e3e38d", verdict: .pass,
            hardwareClass: "apple-m3-ultra", residency: "resident")
        let m5Card = repolessCard(
            id: "qwen3-0p6b-4bit@m5", hfPin: "73e3e38d", verdict: .pass, hardwareClass: "apple-m5",
            residency: "resident")
        let revision = "73e3e38d981303bc594367cd910ea6eb48349da8"
        let resolution = QualityCardStore.resolve(
            repo: nil, revision: revision, hostHardwareClass: "apple-m5", in: [m3ultraCard, m5Card])
        guard case .resolved(let resolved) = resolution else {
            return XCTFail("expected .resolved, got \(resolution)")
        }
        XCTAssertEqual(resolved.id, m5Card.id)
    }

    /// B7: a pin-resolved pool with one NO_GO card on non-matching hardware and one PASS card whose
    /// hardwareClass matches the host must still return the NO_GO card — the fail-closed-first rule
    /// is not reordered by going through the pin path instead of the repo path.
    func testResolveB7PinPoolStaysFailClosedFirst() {
        let noGoOffHost = repolessCard(
            id: "off-host-nogo@m3ultra", hfPin: "73e3e38d", verdict: .noGo,
            hardwareClass: "apple-m3-ultra", residency: "resident")
        let passOnHost = repolessCard(
            id: "on-host-pass@m5", hfPin: "73e3e38d", verdict: .pass, hardwareClass: "apple-m5",
            residency: "resident")
        let revision = "73e3e38d981303bc594367cd910ea6eb48349da8"
        let resolution = QualityCardStore.resolve(
            repo: nil, revision: revision, hostHardwareClass: "apple-m5",
            in: [noGoOffHost, passOnHost])
        guard case .resolved(let resolved) = resolution else {
            return XCTFail("expected .resolved, got \(resolution)")
        }
        XCTAssertEqual(
            resolved.id, noGoOffHost.id,
            "fail-closed NO_GO-first must win even when the pool was assembled via the pin path")
    }

    // MARK: - resolve(...) over the REAL shipped manifest (site/quality-guides.json) — the
    // no-behavior-change assertion for the no-revision, repo-only path.

    /// Walks parent directories upward from this test file's own location looking for
    /// `site/quality-guides.json`, the same idiom `locateFastMLXServeSwift()` above uses — so the
    /// fixture load survives a different checkout/worktree layout. No existing helper in this file
    /// locates a fixture by repo-root walk for a *data* file (only for `FastMLXServe.swift`'s source
    /// text), so this repeats that idiom rather than introducing a new one.
    private func locateSiteQualityGuidesJSON() -> URL? {
        var dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        while true {
            let candidate = dir.appendingPathComponent("site/quality-guides.json")
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
            let parent = dir.deletingLastPathComponent()
            if parent.path == dir.path {
                return nil
            }
            dir = parent
        }
    }

    /// C1: for every repo present in the real shipped manifest, `resolve(repo:revision: nil, ...)`
    /// returns `.resolved` with the SAME card `card(forRepo:...)` returns; an unknown repo returns
    /// `.none` exactly where the old call returns `nil`.
    func testResolveC1MatchesCardForRepoAcrossRealManifest() throws {
        guard let url = locateSiteQualityGuidesJSON() else {
            XCTFail("could not locate site/quality-guides.json by walking up from #filePath")
            return
        }
        let cards = try QualityCardStore.loadManifest(contentsOf: url)
        let repos = Set(cards.compactMap { $0.model.repo })
        XCTAssertFalse(repos.isEmpty, "sanity: the real manifest must carry at least one repo-identified card")

        for repo in repos {
            let old = QualityCardStore.card(forRepo: repo, in: cards)
            let resolution = QualityCardStore.resolve(repo: repo, revision: nil, in: cards)
            guard let old else {
                XCTFail("unexpected nil from card(forRepo:) for repo \(repo) in the real manifest")
                continue
            }
            guard case .resolved(let resolvedCard) = resolution else {
                XCTFail("expected .resolved for repo \(repo), got \(resolution)")
                continue
            }
            XCTAssertEqual(
                resolvedCard.id, old.id,
                "resolve(repo: \(repo), revision: nil, ...) must match card(forRepo: \(repo), ...)")
        }

        let unknownRepo = "mlx-community/this-repo-does-not-exist-in-the-manifest"
        XCTAssertNil(QualityCardStore.card(forRepo: unknownRepo, in: cards))
        XCTAssertEqual(
            QualityCardStore.resolve(repo: unknownRepo, revision: nil, in: cards), .none)
    }

    /// C3: the production shape — a friendly model id (not an HF repo path), no revision — resolves
    /// nothing, i.e. still admits unmeasured, exactly like `card(forRepo:...)` would for the same
    /// non-matching string.
    func testResolveC3FriendlyIDWithNoRevisionResolvesNothing() throws {
        guard let url = locateSiteQualityGuidesJSON() else {
            XCTFail("could not locate site/quality-guides.json by walking up from #filePath")
            return
        }
        let cards = try QualityCardStore.loadManifest(contentsOf: url)
        let resolution = QualityCardStore.resolve(
            repo: "Qwen3.8-Flash-Next-MLX-oQ4-MTP", revision: nil, in: cards)
        XCTAssertEqual(resolution, .none)
    }

    // MARK: - QualityAdmission.announceFragment(card:) — closes the other half of the "operator can
    // always SEE, never merely infer" contract: `quality_cards=<path>` only ever said whether a
    // MANIFEST was consulted, never whether a CARD resolved for this launch. `.none` (no card) and
    // `.resolved` were externally identical before this fragment existed.

    func testAnnounceFragmentNilCardIsQualityCardNone() {
        XCTAssertEqual(QualityAdmission.announceFragment(card: nil), "quality_card=none")
    }

    func testAnnounceFragmentNoGoCardNamesIDAndVerdict() {
        let c = card(id: "qwen38-27b-optiq-4bit@m3ultra", verdict: .noGo)
        XCTAssertEqual(
            QualityAdmission.announceFragment(card: c),
            "quality_card=qwen38-27b-optiq-4bit@m3ultra verdict=NO_GO")
    }

    func testAnnounceFragmentPassCardNamesIDAndVerdict() {
        let c = card(id: "qwen38-27b-8bit-reference@m3ultra", verdict: .pass)
        XCTAssertEqual(
            QualityAdmission.announceFragment(card: c),
            "quality_card=qwen38-27b-8bit-reference@m3ultra verdict=PASS")
    }

    /// Positive control against a vacuous/constant implementation: a function returning `""` or a
    /// fixed string could otherwise satisfy the equality assertions above by accident if their
    /// literals were ever copy-pasted wrong. Assert the nil-card and NO_GO-card fragments actually
    /// DIFFER, and that neither is empty.
    func testAnnounceFragmentNilAndCardFragmentsDifferAndAreNonEmpty() {
        let none = QualityAdmission.announceFragment(card: nil)
        let withCard = QualityAdmission.announceFragment(
            card: card(id: "qwen38-27b-optiq-4bit@m3ultra", verdict: .noGo))
        XCTAssertNotEqual(none, withCard)
        XCTAssertFalse(none.isEmpty)
        XCTAssertFalse(withCard.isEmpty)
    }

    /// The fragment is appended to a launchd/nohup-parsed startup line alongside `quality_cards=
    /// <absolute path>`; it must never itself carry a `/` — a manifest path, a model repo path, or a
    /// revision leaking into what is documented as an id+verdict-only token. Card ids and verdicts are
    /// already public in `site/quality-guides.json`; paths and revisions are not.
    func testAnnounceFragmentNeverContainsSlash() {
        let fragments = [
            QualityAdmission.announceFragment(card: nil),
            QualityAdmission.announceFragment(
                card: card(id: "qwen38-27b-optiq-4bit@m3ultra", verdict: .noGo)),
            QualityAdmission.announceFragment(
                card: card(id: "qwen38-27b-8bit-reference@m3ultra", verdict: .pass)),
        ]
        for fragment in fragments {
            XCTAssertFalse(
                fragment.contains("/"), "fragment must never carry a path or repo: \(fragment)")
        }
    }

    /// F — the production-identity arm. The pack live in production
    /// (`Vontra/Qwen3.8-Flash-Next-MLX-oQ4-MTP` @ `43a82b3f0ff64fa417fd09ca046580f08d19b0d6`) is
    /// UNCARDED: no shipped card names that repo, and no card's `hfPin` prefix-matches that revision.
    /// Before this fragment existed, that resolved `.none` -> `admitUnmeasured` -> the SAME startup
    /// line an admitted, carded-and-passing launch would print — an operator could not tell "no card
    /// consulted this model at all" from "a card resolved and it's fine". This fixture reproduces that
    /// exact shape with the two real shipped repo-less NO_GO pins, plus a repo-identified card, so the
    /// non-match is not an artifact of an empty/degenerate manifest.
    func testAnnounceFragmentProductionIdentityResolvesNoneAndFragmentSaysNone() {
        let repolessA = repolessCard(
            id: "repoless-a@m3ultra", hfPin: "ef5b919d31534faa1997666f1a22d362cd6383cd", verdict: .noGo)
        let repolessB = repolessCard(
            id: "repoless-b@m3ultra", hfPin: "2b7da62be0151a7932e4dfcab1d73c93ccf83f64", verdict: .noGo)
        let repoIdentified = card(verdict: .pass)
        let fixture = [repolessA, repolessB, repoIdentified]

        let resolution = QualityCardStore.resolve(
            repo: "Vontra/Qwen3.8-Flash-Next-MLX-oQ4-MTP",
            revision: "43a82b3f0ff64fa417fd09ca046580f08d19b0d6",
            hostHardwareClass: "apple-m3-ultra", in: fixture)
        XCTAssertEqual(
            resolution, .none,
            "the production pack's repo+revision must not match either repo-less pin or the unrelated repo card")

        let resolvedCard: QualityCard?
        if case .resolved(let c) = resolution { resolvedCard = c } else { resolvedCard = nil }
        XCTAssertEqual(
            QualityAdmission.announceFragment(card: resolvedCard), "quality_card=none",
            "uncarded must announce quality_card=none, not be indistinguishable from a passing card")

        // Contrast arm: the fixture is not inert -- resolving BY the real repo-less pin's own
        // revision must actually resolve that card and name it, proving the .none result above is a
        // genuine non-match rather than a fixture that can never resolve anything.
        let contrastResolution = QualityCardStore.resolve(
            repo: "Vontra/Qwen3.8-Flash-Next-MLX-oQ4-MTP",
            revision: "ef5b919d31534faa1997666f1a22d362cd6383cd",
            hostHardwareClass: "apple-m3-ultra", in: fixture)
        guard case .resolved(let contrastCard) = contrastResolution else {
            return XCTFail("expected .resolved for the matching pin, got \(contrastResolution)")
        }
        XCTAssertEqual(contrastCard.id, repolessA.id)
        XCTAssertEqual(
            QualityAdmission.announceFragment(card: contrastCard),
            "quality_card=repoless-a@m3ultra verdict=NO_GO")
    }
}
