import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// The quality-guidance moat's admission verdict for a resolved model+config, as authored into
/// `site/quality-guides.json` by `scripts/emit_quality_card.py`. See
/// `docs/quality-card-schema-v1.md` — this file is the contract; the discriminator table there and
/// `QualityAdmission.decide` below MUST agree.
public enum QualityVerdict: String, Sendable, Codable {
    case noGo = "NO_GO"
    case pass = "PASS"
    case reference = "REFERENCE"
    case exact = "EXACT"
    case unmeasured = "UNMEASURED"

    /// Fail-closed decode: an unrecognized verdict string (a future schema addition, a corrupted
    /// manifest row) never crashes the serve path — it decodes to `.unmeasured`, which the
    /// discriminator table treats identically to "no card" (admit, never refuse). Fail-closed applies
    /// to *claims* here (never trust an unrecognized string as a passing verdict), not to admission.
    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = QualityVerdict(rawValue: raw) ?? .unmeasured
    }
}

/// The minimal slice of a quality card the Swift serve gate + announce need: the admission
/// discriminator inputs (`verdict`, `admission`) and the legible one-line summary the refusal/flag
/// message quotes. Decoded via an explicit `CodingKeys` + hand-rolled `init(from:)` (not synthesized
/// `Decodable` — see below): `id`, `model`, `verdict`, and `legible` stay REQUIRED, exactly as strict
/// synthesis would decode them (any missing or malformed value still throws, dropping the whole card
/// at the `LenientQualityCard` layer above); `config` keeps synthesis's `decodeIfPresent` semantics
/// verbatim (present-but-wrong-type still throws). Only `admission` is decoded via
/// `try? container.decodeIfPresent(Admission.self, forKey: .admission)`, so a missing, null, or
/// STRUCTURALLY CORRUPT `admission` value (not only a missing `optIn`) yields `nil` instead of
/// throwing — see docs/task-inbox/2026-09-23-PREDECLARATION-a-dropped-card-still-admits.md (D2: a
/// synthesized `Admission?` alone does NOT achieve this, since synthesized `decodeIfPresent` only
/// returns `nil` for an absent/null key, not a present-but-malformed one; D3: this is deliberately
/// wider than "missing `optIn`" alone; D4: safe only because nothing in Swift ever reads a decoded
/// `.admission` value). New emitter-side fields (`rawMetrics`, `provenance`, `boundary`, …) never
/// break this decode simply because `CodingKeys` never names them — ordinary `Decodable` behavior,
/// not special leniency.
public struct QualityCard: Sendable, Decodable, Equatable {
    public struct Model: Sendable, Decodable, Equatable {
        /// `nil` for a non-checkpoint-specific card (e.g. an "enhancement" card like native MTP,
        /// which applies to whatever base model is already resolved rather than naming its own HF
        /// repo) — such a card can never match a `forRepo:` lookup, which is correct: it is not
        /// admission-gated by repo identity.
        public let repo: String?
        public let hfPin: String?

        public init(repo: String?, hfPin: String? = nil) {
            self.repo = repo
            self.hfPin = hfPin
        }
    }

    public struct Admission: Sendable, Decodable, Equatable {
        public let `default`: Bool
        public let optIn: Bool
        public let reason: String

        public init(default defaultValue: Bool, optIn: Bool, reason: String) {
            self.default = defaultValue
            self.optIn = optIn
            self.reason = reason
        }
    }

    public struct Legible: Sendable, Decodable, Equatable {
        public let tier: String
        public let headline: String

        public init(tier: String, headline: String) {
            self.tier = tier
            self.headline = headline
        }
    }

    /// The slice of `config` this gate consults: `residency` (the only field that changes whether a
    /// card is even ELIGIBLE to gate a resident Swift launch) and `hardwareClass` (a uniqueness-key /
    /// tiebreak input — see `QualityCardStore.card(forRepo:hostHardwareClass:in:)` — never an
    /// eligibility filter). Both decode leniently: a missing or `null` value is `nil`, never a decode
    /// failure. Any other emitter-side `config` field (`quant`, `enhancement`, …) is ignored, never
    /// failing this decode.
    public struct Config: Sendable, Decodable, Equatable {
        public let residency: String?
        /// The hardware class this card was MEASURED on (e.g. `"apple-m3-ultra"`), or `nil` for a
        /// card that predates this field. A host property, not a launch parameter: it is a
        /// uniqueness-key component and a multi-candidate tiebreak, and it must NEVER filter
        /// admission — see the doc comment on `QualityCardStore.card(forRepo:hostHardwareClass:in:)`.
        public let hardwareClass: String?

        public init(residency: String? = nil, hardwareClass: String? = nil) {
            self.residency = residency
            self.hardwareClass = hardwareClass
        }
    }

    public let id: String
    public let model: Model
    public let verdict: QualityVerdict
    public let admission: Admission?
    public let legible: Legible
    /// `nil` when the card predates the `config.residency` field, or when its `config` object omits
    /// `residency` — both mean "measured resident" per `effectiveResidency` below.
    public let config: Config?

    public init(
        id: String, model: Model, verdict: QualityVerdict, admission: Admission?, legible: Legible,
        config: Config? = nil
    ) {
        self.id = id
        self.model = model
        self.verdict = verdict
        self.admission = admission
        self.legible = legible
        self.config = config
    }

    private enum CodingKeys: String, CodingKey {
        case id, model, verdict, admission, legible, config
    }

    /// Hand-rolled rather than synthesized so `admission` alone can tolerate a present-but-malformed
    /// value (see the type's doc comment above for why the naive `Admission?` + synthesized decode is
    /// NOT sufficient). Every other property is decoded with the exact strictness synthesis would have
    /// given it -- this decoder is a narrowing of leniency to one field, not a wholesale relaxation.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        model = try container.decode(Model.self, forKey: .model)
        verdict = try container.decode(QualityVerdict.self, forKey: .verdict)
        admission = try? container.decodeIfPresent(Admission.self, forKey: .admission)
        legible = try container.decode(Legible.self, forKey: .legible)
        config = try container.decodeIfPresent(Config.self, forKey: .config)
    }

    /// The residency this card was measured under: `config.residency` verbatim, or `"resident"` when
    /// `config` or `config.residency` is absent/null. Kept verbatim (never normalized/validated) so an
    /// unrecognized future value is visible in logs/tests rather than silently coerced.
    public var effectiveResidency: String {
        config?.residency ?? "resident"
    }

    /// `true` only when this card was measured under resident weights — the ONLY residency the Swift
    /// `fastmlx-serve` engine can serve (it cannot stream experts from SSD). A card measured under SSD
    /// expert streaming changes greedy-decode output relative to a resident load, so an
    /// `"expert-stream"` card must never gate a resident launch. Fail-closed: any residency string
    /// other than exactly `"resident"` — including an unrecognized future value — never matches, so a
    /// typo'd or forward-incompatible `config.residency` can only ever make a card LESS eligible to
    /// gate, never silently trusted as resident.
    public var matchesResidentLaunch: Bool {
        effectiveResidency == "resident"
    }
}

/// The outcome of admitting a resolved model+config against its quality card (or its absence). Pure
/// data — no I/O, no process exit; the caller (`FastMLXServe`) maps this onto stderr + exit code.
public enum QualityAdmissionOutcome: Sendable, Equatable {
    /// No applicable NO_GO verdict: serve silently. Covers both "no card" and a PASS/REFERENCE/EXACT
    /// card — the default announce must stay byte-identical in both cases.
    case admit
    /// No card exists, or the card's verdict is UNMEASURED. Serve silently — unmeasured never
    /// refuses; this is distinguished from `.admit` only so callers/tests can see WHY admission
    /// happened, never so it behaves differently.
    case admitUnmeasured
    /// NO_GO, and the operator explicitly opted in for this card: serve, but surface the one-line
    /// quality flag in the announce.
    case admitWithQualityFlag(String)
    /// NO_GO, no opt-in: refuse. The message names the tier + headline and how to opt in.
    case refuseQualityFlagged(String)
}

/// Implements the "Admission discriminator rules" table in `docs/quality-card-schema-v1.md` exactly.
/// Pure function: given the (possibly absent) card for the resolved model+config and whether the
/// operator opted in for it, decides whether to serve. Keys on `verdict == .noGo` SPECIFICALLY —
/// never on "absence of a passing card" — so a model with no card, or an UNMEASURED card, always
/// admits.
public enum QualityAdmission {
    /// The `quality_card=` startup-line fragment: closes the OTHER half of the "operator can always
    /// SEE, never merely infer" contract that `QualityCardsManifestResolution`'s `quality_cards=`
    /// fragment left open. `quality_cards=<path>` / `quality_cards=none` only ever say whether a
    /// MANIFEST was consulted; they say nothing about whether a CARD resolved for the launched
    /// model+revision. Before this fragment existed, `.none` (no card matched) and `.resolved` were
    /// externally identical -- both fell through the discriminator switch in
    /// `applyQualityAdmissionGate` silently -- so an operator seeing `quality_cards=/…/quality-guides.json`
    /// could not tell "this pack is uncarded" from "this pack is carded and PASS".
    ///
    /// `nil` card -> exactly `"quality_card=none"`. Non-nil -> exactly
    /// `"quality_card=<card.id> verdict=<card.verdict.rawValue>"`. Total over its input (every
    /// `QualityCard` decodes to a non-optional `id` and `verdict`); no I/O, no force-unwrap.
    ///
    /// Deliberately carries `id` and `verdict` ONLY -- never `model.repo`, `model.hfPin`, or the
    /// manifest path. Both of those already appear verbatim in the published `site/quality-guides.json`,
    /// so naming a card by `id`/`verdict` in a log line discloses nothing new; a resolved HF repo path
    /// or revision hash is host/deployment-specific and does not belong in this token. The fragment
    /// must never contain `/` for the same reason `quality_cards=` is kept as a separate, path-shaped
    /// token: a supervisor line-parser (launchd/nohup) must be able to split on whitespace without a
    /// path embedding an ambiguous separator into what is documented as an id+verdict-only field.
    public static func announceFragment(card: QualityCard?) -> String {
        guard let card else { return "quality_card=none" }
        return "quality_card=\(card.id) verdict=\(card.verdict.rawValue)"
    }

    /// Builds the `quality_card_ambiguous` refusal detail string for
    /// `FastMLXServe.applyQualityAdmissionGate`'s `.ambiguous` case, given the launch's `model` repo,
    /// its `--model-revision` (or `nil`), and the ambiguous resolution's sorted repo/pin card id
    /// lists. Pulled out as a pure function so the CLI's message content is independently testable
    /// (see `QualityAdmissionTests`) without exercising the `exit(2)` process-termination path.
    ///
    /// When the two id lists are equal AS SETS -- a duplicate `id` published on two structurally
    /// different cards, see `QualityCardStore`'s `CardIdentity` doc comment -- the plain wording
    /// self-contradicts ("matches card(s) [X] but ... matches different card(s) [X]"); a
    /// `duplicate_card_id=<id>` token is appended so the refusal names the actual defect. Wording is
    /// otherwise unchanged from before this token existed.
    public static func ambiguousRefusalDetail(
        model: String, revision: String?, repoCardIDs: [String], pinCardIDs: [String]
    ) -> String {
        let duplicateToken =
            Set(repoCardIDs) == Set(pinCardIDs)
            ? " duplicate_card_id=\(repoCardIDs.sorted().joined(separator: ","))" : ""
        return
            "repo \(model) matches card(s) \(repoCardIDs) but --model-revision \(revision ?? "nil") matches different card(s) \(pinCardIDs)\(duplicateToken)"
    }

    /// The D-B visibility fragment: `" quality_cards_count=0"` when the manifest decoded to zero
    /// cards (a truncated write, a hand-authored `{"cards": []}`), or `""` otherwise. Mirrors the
    /// existing `quality_cards_dropped` idiom in `FastMLXServe.applyQualityAdmissionGate`: never emit
    /// `quality_cards_count=0` for a well-formed non-empty manifest, so the happy-path announce line
    /// for every existing launch stays byte-identical. Pulled out as a pure, independently testable
    /// function (see `QualityAdmissionTests`) rather than inlined at its one call site.
    public static func cardsCountFragment(cards: [QualityCard]) -> String {
        cards.isEmpty ? " quality_cards_count=0" : ""
    }

    public static func decide(card: QualityCard?, optIn: Bool) -> QualityAdmissionOutcome {
        guard let card else { return .admitUnmeasured }
        switch card.verdict {
        case .unmeasured:
            return .admitUnmeasured
        case .pass, .reference, .exact:
            return .admit
        case .noGo:
            let summary = "\(card.legible.tier): \(card.legible.headline)"
            if optIn {
                return .admitWithQualityFlag(summary)
            }
            return .refuseQualityFlagged(
                summary + " re-run with --accept-quality \(card.id) to elect it.")
        }
    }
}

/// A single element of the manifest's `cards` array, decoded leniently: `card` is `nil` when this
/// element fails to decode as a `QualityCard` for ANY reason (a missing required field such as
/// `verdict`, a malformed value, or a non-object element) rather than throwing out of the
/// surrounding array decode. `admission` is the deliberate EXCEPTION: `QualityCard.init(from:)`
/// decodes it via `try?`, so a missing or malformed `admission` no longer drops the card here.
/// `init(from:)` here must NEVER throw -- CRITICAL, do not "fix" this by removing the `try?` below.
///
/// `UnkeyedDecodingContainer.decode(_:)` does NOT advance `currentIndex` when the element decode
/// throws (probed directly on Apple Swift 6.4). A naive
/// `while !container.isAtEnd { _ = try? container.decode(QualityCard.self) }` would therefore
/// INFINITE-LOOP on the first malformed element: the failed decode never consumes it, so `isAtEnd`
/// never becomes true. Wrapping the fallible decode INSIDE this always-succeeding `Decodable`
/// conformance is what makes `[LenientQualityCard]`'s own array decode -- which DOES always advance,
/// because `LenientQualityCard.init(from:)` never throws -- skip a malformed element exactly once,
/// the same way it advances past a well-formed one.
private struct LenientQualityCard: Decodable {
    let card: QualityCard?
    init(from decoder: Decoder) throws { card = try? QualityCard(from: decoder) }
}

/// The top-level `site/quality-guides.json` manifest envelope
/// (`docs/quality-card-schema-v1.md`'s "Top-level manifest"). Decoded leniently — `schema` and
/// `generatedAt` are carried but not asserted here; only `cards` is consulted.
///
/// `cards` itself is decoded per-element leniently via `LenientQualityCard`: one malformed card
/// (e.g. missing `verdict`) is dropped and counted in `droppedCardCount`, never thrown out of
/// the whole manifest decode -- closing the defect where one bad card silently disarmed every other
/// card's admission gate. `admission` is the exception: it decodes leniently inside `QualityCard`
/// itself (see its doc comment), so a missing/malformed `admission` alone never drops a card here.
/// The ENVELOPE stays strict: `cards` must still decode as a JSON array (the
/// keyed `container.decode([LenientQualityCard].self, forKey: .cards)` call below still throws on a
/// structurally corrupt envelope, e.g. `cards` as an object) -- leniency must never turn a corrupt
/// manifest into "0 cards, gate armed, everything silently admits".
struct QualityCardManifest: Decodable {
    let cards: [QualityCard]
    /// Count of `cards` array elements that failed to decode as a `QualityCard` and were dropped.
    /// `0` for a fully well-formed manifest (including every manifest decoded before this type
    /// gained per-element leniency).
    let droppedCardCount: Int

    private enum CodingKeys: String, CodingKey {
        case cards
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let lenientCards = try container.decode([LenientQualityCard].self, forKey: .cards)
        cards = lenientCards.compactMap(\.card)
        droppedCardCount = lenientCards.count - cards.count
    }
}

/// The public result of `QualityCardStore.loadManifestDetailed(contentsOf:)`: the cards that
/// survived per-element decode, plus how many did not. `droppedCardCount == 0` for a fully
/// well-formed manifest — including every manifest that predates per-card leniency.
public struct QualityCardManifestLoadResult: Sendable, Equatable {
    public let cards: [QualityCard]
    public let droppedCardCount: Int

    public init(cards: [QualityCard], droppedCardCount: Int) {
        self.cards = cards
        self.droppedCardCount = droppedCardCount
    }
}

/// Loads a `QualityCard` for a resolved repo out of a `fast-mlx-quality-card-v1` manifest. Fails
/// closed to "no card" (→ `QualityAdmission.decide` → `.admitUnmeasured`) on any decode failure or
/// missing file, so a serve with no manifest, an unreadable manifest, or a manifest that fails to
/// parse behaves EXACTLY as today — never a refusal caused by a broken loader.
public enum QualityCardStore {
    /// The producer-side fail-closed sentinel a chip probe returns when identification fails (e.g.
    /// `ProvenanceCLI.chipBrand()`, `fastmlx_bench._chip_identity`). Lowercase alphanumeric, so it
    /// would otherwise look like a plausible class -- this guard exists so it can never host-match.
    private static let hardwareClassSentinel = "unknown"

    /// Normalizes a raw chip-brand string into the canonical `config.hardwareClass` / host-probe
    /// shape: trim, lowercase, then collapse each run of whitespace to a single `-`. `nil` in, `nil`
    /// out; `nil` for an empty/whitespace-only trimmed result; `nil` for the chip-probe-failed
    /// sentinel `"unknown"` (checked case-insensitively, i.e. AFTER lowercasing, so `"UNKNOWN"` /
    /// `" Unknown "` are caught too).
    ///
    /// Mirrors `fastmlx_launch.host_hardware_class()`'s normalization (`scripts/fastmlx_launch.py`,
    /// `brand.lower().replace(" ", "-")`), generalized to arbitrary whitespace runs the way
    /// `emit_quality_card.validate_hardware_class` does (`scripts/emit_quality_card.py:97`,
    /// `re.sub(r"\s+", "-", value.strip().lower())`). Deliberately does NOT add canonical-regex
    /// validation beyond the sentinel check: Python's live detector (`fastmlx_launch.py:775-778`)
    /// does not validate its own output against the `^[a-z0-9]+(-[a-z0-9]+)*$` shape either -- a
    /// non-canonical value (e.g. one carrying a comma) is not "fixed up", it simply fails to match
    /// any card, which is the correct and sufficient outcome. Keeping the rule identical to the
    /// Python detector -- not to the stricter emitter-side validator -- is the point.
    ///
    /// This is a deliberate separate copy rather than an import of the Python normalizer or of
    /// `SystemProfile`'s own sysctl helper: the codebase already keeps independent copies of this
    /// rule across `fastmlx_launch.py`, `build_public_site.py`, and `validate_public_site.py`
    /// (`scripts/emit_quality_card.py:68-74`), and `emit_quality_card.py:82-86` documents directly why
    /// the sentinel check must survive any such copy. A fourth copy in Swift follows that established
    /// split rather than reaching across the Python/Swift boundary for a few lines of string logic.
    public static func canonicalHardwareClass(fromChipBrand brand: String?) -> String? {
        guard let brand else { return nil }
        let trimmed = brand.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let lowered = trimmed.lowercased()
        let normalized = lowered.replacingOccurrences(
            of: "\\s+", with: "-", options: .regularExpression)
        guard normalized != hardwareClassSentinel else { return nil }
        return normalized
    }

    /// Probes THIS host's hardware class for the admission tiebreak (step 3 of
    /// `card(forRepo:hostHardwareClass:in:)`). Reads ONLY `machdep.cpu.brand_string` via
    /// `sysctlbyname` and fails closed to `nil` on any failure (a nonzero/absent sysctl, or an empty
    /// result) -- mirroring `fastmlx_launch.host_hardware_class()`'s fail-closed contract on the
    /// Swift side.
    ///
    /// Deliberately does NOT reuse `SystemProfile.detectHost()`: that detector's `chip` field falls
    /// back from `machdep.cpu.brand_string` to `hw.model` (e.g. `"Mac17,3"`, which is not canonical
    /// and carries a comma) and finally to the literal `"unknown"` -- the exact chip-probe-failed
    /// sentinel `canonicalHardwareClass(fromChipBrand:)` refuses by name. That fallback chain is
    /// right for the sizer, which wants a best-effort human-readable chip name, and wrong for card
    /// identity, which must fail closed to `nil` rather than silently emit a non-canonical or
    /// sentinel value a card validator would reject. Mirrors the narrow, private
    /// `SystemProfile.sysctlString` reader (`SystemProfile.swift:334+`) rather than making it public
    /// or importing it, since `SystemProfile` deliberately stays MLX/GPU-toolchain-free and this is a
    /// one-sysctl probe.
    public static func hostHardwareClass() -> String? {
        guard let brand = sysctlString("machdep.cpu.brand_string") else { return nil }
        return canonicalHardwareClass(fromChipBrand: brand)
    }

    /// Minimal C-string sysctl read, mirroring `SystemProfile.sysctlString` narrowly rather than
    /// reusing it (that helper is `private` to `SystemProfile` by design).
    private static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
        let bytes = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self)
    }

    /// Pure core: decode `manifestData` as a `fast-mlx-quality-card-v1` manifest and return the first
    /// card whose `model.repo` matches `repoID` AND whose `effectiveResidency` is `"resident"`
    /// (`QualityCard.matchesResidentLaunch`). This engine only serves resident weights — it cannot
    /// stream experts from SSD — and streaming a model changes its greedy-decode output relative to a
    /// resident load, so a card measured under `"expert-stream"` (or any unrecognized residency
    /// string) must never match here, however it is ordered in the manifest: it is treated exactly
    /// like "no card for this repo", i.e. `.admitUnmeasured`, never as a resident admission input.
    /// `nil` on an unknown repo, a repo with only non-resident cards, or a decode failure.
    public static func card(
        forRepo repoID: String, hostHardwareClass: String? = nil, in manifestData: Data
    ) -> QualityCard? {
        guard let manifest = try? JSONDecoder().decode(QualityCardManifest.self, from: manifestData)
        else {
            return nil
        }
        return card(forRepo: repoID, hostHardwareClass: hostHardwareClass, in: manifest.cards)
    }

    /// The single card-selection rule over an already-narrowed candidate pool (a repo match, a pin
    /// match, or any other identity-matched set) — shared by `card(forRepo:hostHardwareClass:in:)`
    /// and `resolve(repo:revision:hostHardwareClass:in:)` so the 4-step rule documented on
    /// `card(forRepo:hostHardwareClass:in:)` below is implemented exactly once.
    ///
    /// 1. 0 or 1 match — return it. This is TODAY'S BEHAVIOR, byte-for-byte identical to
    ///    `candidates.first`, and must stay exactly that: it is what preserves every existing user's
    ///    resolution. (An Ultra card and an M5 card for the same pack are legitimately distinct
    ///    measurements, not duplicates — see
    ///    `docs/task-inbox/2026-09-22-PREDECLARATION-hardwareclass-joins-identity-never-filters.md`
    ///    — so reaching >1 match here is an expected, supported case, not an error.)
    /// 2. More than one match: manifest array order must NEVER decide the outcome (order is
    ///    emitter-determined and not stable under re-emission). Fail-closed FIRST — if any candidate's
    ///    `verdict == .noGo`, narrow the pool to just those: `QualityAdmission.decide` refuses ONLY on
    ///    `.noGo` (`:141-163`), so a broken quantization must keep refusing even on hardware it was
    ///    never measured on — a card measured on other hardware is WEAKER evidence, not VOID evidence,
    ///    and hardwareClass must never act as an eligibility filter (that would silently disarm every
    ///    published NO_GO card on every non-matching host, converting a fail-closed gate into a
    ///    fail-open one). If no candidate is `.noGo`, the pool stays every remaining candidate — none
    ///    of them can change the outcome, since they all admit.
    /// 3. Host class is a TIEBREAK WITHIN that pool only, never a filter and never a way to escape the
    ///    pool chosen in step 2: if `hostHardwareClass` is non-nil and exactly one of the pool's
    ///    candidates has `config?.hardwareClass` equal to it, return that one. An exact host match is
    ///    the strongest available signal for choosing WHICH card of the surviving pool is most
    ///    relevant — but it must never reach across pools to prefer an admitting card over a NO_GO one;
    ///    doing so would defeat step 2's fail-closed preference by ordering, not by filtering.
    /// 4. Otherwise (no unique host match, or no host hint), pick deterministically by the
    ///    lexicographically smallest card `id` within the pool, so the `--accept-quality <id>` refusal
    ///    message is stable across manifest re-emissions.
    private static func select(
        from candidates: [QualityCard], hostHardwareClass: String?
    ) -> QualityCard? {
        if candidates.count <= 1 {
            return candidates.first
        }
        let noGoMatches = candidates.filter { $0.verdict == .noGo }
        let tiePool = noGoMatches.isEmpty ? candidates : noGoMatches
        if let hostHardwareClass {
            let hostMatches = tiePool.filter { $0.config?.hardwareClass == hostHardwareClass }
            if hostMatches.count == 1 {
                return hostMatches[0]
            }
        }
        return tiePool.min { $0.id < $1.id }
    }

    /// The single card-selection rule over already-decoded cards. Every Swift call site that picks a
    /// card by identity goes through here, so the residency filter cannot be bypassed by a caller
    /// that loaded the manifest itself.
    ///
    /// Filters to `model.repo == repoID && matchesResidentLaunch`, then runs the shared 4-step
    /// selection rule documented on `select(from:hostHardwareClass:)` above.
    public static func card(
        forRepo repoID: String, hostHardwareClass: String? = nil, in cards: [QualityCard]
    ) -> QualityCard? {
        let matches = cards.filter { $0.model.repo == repoID && $0.matchesResidentLaunch }
        return select(from: matches, hostHardwareClass: hostHardwareClass)
    }

    /// Full hex-digit ASCII check mirroring Python's `_HEX_DIGITS_RE = re.compile(r"^[0-9a-fA-F]+$")`
    /// exactly — deliberately NOT `Character.isHexDigit`, which also accepts non-ASCII Unicode
    /// "Hex_Digit"-property characters (e.g. fullwidth digit forms) that Python's regex would reject.
    private static func isAllASCIIHexDigits(_ value: String) -> Bool {
        !value.isEmpty
            && value.utf8.allSatisfy { byte in
                (0x30...0x39).contains(byte) || (0x41...0x46).contains(byte)
                    || (0x61...0x66).contains(byte)
            }
    }

    /// Mirrors Python `_is_full_hex_revision` (`scripts/fastmlx_launch.py:628`): exactly 40 hex
    /// characters, else `false` — including `nil`.
    private static func isFullHexRevision(_ value: String?) -> Bool {
        guard let value, value.count == 40 else { return false }
        return isAllASCIIHexDigits(value)
    }

    /// Mirrors Python `_is_usable_hf_pin` (`scripts/fastmlx_launch.py:636`): at least 8 hex
    /// characters, else `false` — including `nil`.
    private static func isUsableHfPin(_ value: String?) -> Bool {
        guard let value, value.count >= 8 else { return false }
        return isAllASCIIHexDigits(value)
    }

    /// Mirrors Python `_hf_pin_matches_revision` (`scripts/fastmlx_launch.py:644`): `revision` must
    /// be a full 40-hex sha, `hfPin` must be a usable (>= 8 hex) pin, and `revision` must
    /// case-insensitively PREFIX-match `hfPin` (not equal it — a card's `hfPin` may itself be a short
    /// prefix of the pinned commit).
    private static func hfPinMatchesRevision(hfPin: String?, revision: String?) -> Bool {
        guard isFullHexRevision(revision), isUsableHfPin(hfPin), let hfPin, let revision else {
            return false
        }
        return revision.lowercased().hasPrefix(hfPin.lowercased())
    }

    /// Every resident card whose `hfPin` prefix-matches `revision`, mirroring Python
    /// `find_cards_by_pin` (`scripts/fastmlx_launch.py:659`). Empty when `revision` is not a full
    /// 40-hex sha (`isFullHexRevision`), or when no card's `hfPin` is a usable prefix of it.
    private static func pinMatches(revision: String?, in residentCards: [QualityCard]) -> [QualityCard]
    {
        guard isFullHexRevision(revision) else { return [] }
        return residentCards.filter { hfPinMatchesRevision(hfPin: $0.model.hfPin, revision: revision) }
    }

    /// Convenience: read `manifestURL` and decode it. Returns `nil` (never throws) when the file is
    /// missing or unreadable — the no-manifest-today behavior is unchanged.
    public static func card(
        forRepo repoID: String, hostHardwareClass: String? = nil, manifestURL: URL
    ) -> QualityCard? {
        guard let data = try? Data(contentsOf: manifestURL) else { return nil }
        return card(forRepo: repoID, hostHardwareClass: hostHardwareClass, in: data)
    }

    /// Strict decode: throws `QualityCardsManifestUndecodable` on an unreadable file or a payload
    /// whose ENVELOPE fails `fast-mlx-quality-card-v1` decoding, instead of
    /// `card(forRepo:manifestURL:)`'s silent nil. Used only for an EXPLICITLY supplied
    /// `--quality-cards` path — see `QualityCardsManifestResolver`'s doc comment for why the
    /// conventional default keeps the silent fail-open behavior via the convenience method above
    /// instead.
    ///
    /// A malformed INDIVIDUAL card no longer throws here (see `QualityCardManifest`'s per-element
    /// leniency) -- it is silently dropped, exactly like `loadManifest(contentsOf:)` below, which is
    /// why that entry point's four existing call sites need no change. A caller that must react to a
    /// dropped card (the D2 explicit-path-refuses decision — see `FastMLXServe.applyQualityAdmissionGate`)
    /// needs `droppedCardCount`, which `loadManifest`'s `[QualityCard]`-only return type cannot carry;
    /// this is the entry point that exposes it.
    public static func loadManifestDetailed(contentsOf url: URL) throws -> QualityCardManifestLoadResult
    {
        guard let data = try? Data(contentsOf: url),
            let manifest = try? JSONDecoder().decode(QualityCardManifest.self, from: data)
        else {
            throw QualityCardsManifestUndecodable(path: url.path)
        }
        return QualityCardManifestLoadResult(
            cards: manifest.cards, droppedCardCount: manifest.droppedCardCount)
    }

    /// `loadManifestDetailed(contentsOf:)`'s `.cards` only, for the four call sites that never needed
    /// `droppedCardCount` and predate it. Behavior is byte-identical to before per-card leniency
    /// existed: a fully well-formed manifest returns every card; the ENVELOPE throw behavior
    /// (missing file, corrupt JSON, corrupt `cards` shape) is unchanged; a malformed individual card
    /// is now dropped rather than throwing (this entry point simply does not expose the count).
    public static func loadManifest(contentsOf url: URL) throws -> [QualityCard] {
        try loadManifestDetailed(contentsOf: url).cards
    }

    /// The implicit (no `--card-id`) lookup: a repo match (`card(forRepo:hostHardwareClass:in:)`'s
    /// semantics) OR an `hfPin`-prefix match against `revision`. Mirrors Python `resolve_card`
    /// (`scripts/fastmlx_launch.py:952-1082`) exactly, generalized to return a `QualityCardResolution`
    /// rather than `Optional<QualityCard>`: `card(forRepo:hostHardwareClass:in:)`'s `nil` already
    /// means "no card" (→ `QualityAdmission.decide` → `.admitUnmeasured`), so an ambiguous
    /// repo-vs-pin collision cannot be expressed by that return type without silently admitting a
    /// launch this lookup could not actually disambiguate — `.ambiguous` is a THIRD outcome, distinct
    /// from both `.none` and `.resolved`, precisely so a caller is forced to refuse rather than guess.
    ///
    /// 1. `repoCards`: resident cards with `model.repo == repo` (empty when `repo == nil`).
    /// 2. `pinCards`: resident cards whose `hfPin` prefix-matches `revision`
    ///    (`pinMatches(revision:in:)`; empty when `revision == nil` or not a full 40-hex sha).
    /// 3. If both are non-empty AND name different card-id sets, the lookup is ambiguous: a manifest
    ///    naming the same model twice, once keyed by repo and once only by pin, cannot be resolved by
    ///    silently preferring one over the other.
    /// 4. Otherwise, `candidates` = `repoCards` if non-empty, else `pinCards`; `.none` when empty.
    /// 5. The shared `select(from:hostHardwareClass:)` 4-step rule runs over `candidates` — identical
    ///    fail-closed-NO_GO-first narrowing and hardwareClass tiebreak whether the pool came from the
    ///    repo path or the pin path.
    /// The identity fields ambiguity comparison narrows to — exactly the five fields that can change
    /// `select`'s outcome: `id`, `model.repo`, `model.hfPin`, `verdict`, and `config?.hardwareClass`.
    /// This is a DELIBERATE NARROWING, not an oversight: two cards that differ only in `legible`,
    /// `admission`, or `rawMetrics` (a field this decode never even keeps) compare EQUAL here and
    /// raise no ambiguity, because neither field can change which card `select` would choose. Two
    /// cards sharing `id` but differing in any of these five fields (e.g. a repo-matched `PASS` and a
    /// pin-matched `NO_GO` published under the same `id` by mistake) compare DIFFERENT, so the id
    /// string alone can no longer hide a real ambiguity behind an accidental id collision.
    ///
    /// Do NOT add `admission` to this tuple. `QualityCard.admission` is decoded via `try?` specifically
    /// so nothing in Swift ever reads a decoded value from it (see `QualityCard`'s doc comment, D4) —
    /// folding it in here would read it and silently make a `try?`-decoded field outcome-bearing.
    private struct CardIdentity: Hashable {
        let id: String
        let repo: String?
        let hfPin: String?
        let verdict: QualityVerdict
        let hardwareClass: String?
    }

    private static func identity(of card: QualityCard) -> CardIdentity {
        CardIdentity(
            id: card.id, repo: card.model.repo, hfPin: card.model.hfPin, verdict: card.verdict,
            hardwareClass: card.config?.hardwareClass)
    }

    public static func resolve(
        repo: String?, revision: String?, hostHardwareClass: String? = nil, in cards: [QualityCard]
    ) -> QualityCardResolution {
        let residentCards = cards.filter { $0.matchesResidentLaunch }
        let repoCards: [QualityCard] =
            repo.map { repoID in residentCards.filter { $0.model.repo == repoID } } ?? []
        let pinCards = pinMatches(revision: revision, in: residentCards)

        let repoIdentities = Set(repoCards.map(identity(of:)))
        let pinIdentities = Set(pinCards.map(identity(of:)))
        if !repoIdentities.isEmpty, !pinIdentities.isEmpty, repoIdentities != pinIdentities {
            let repoIDs = Set(repoCards.map(\.id))
            let pinIDs = Set(pinCards.map(\.id))
            return .ambiguous(repoCardIDs: repoIDs.sorted(), pinCardIDs: pinIDs.sorted())
        }

        let candidates = repoCards.isEmpty ? pinCards : repoCards
        guard !candidates.isEmpty else { return .none }
        guard let selected = select(from: candidates, hostHardwareClass: hostHardwareClass) else {
            return .none
        }
        return .resolved(selected)
    }
}

/// The outcome of `QualityCardStore.resolve(repo:revision:hostHardwareClass:in:)`. Distinct from
/// `QualityCard?` so an ambiguous repo-vs-pin collision (`.ambiguous`) can never be confused with "no
/// card" (`.none`, which `QualityAdmission.decide` treats as `.admitUnmeasured`) — see `resolve`'s
/// doc comment.
public enum QualityCardResolution: Sendable, Equatable {
    /// No repo or pin candidate at all.
    case none
    /// Exactly one card survived the repo/pin lookup and the shared selection rule.
    case resolved(QualityCard)
    /// The repo lookup and the pin lookup both matched, but named DIFFERENT card id sets. Both id
    /// lists are sorted for a stable, reproducible refusal message.
    case ambiguous(repoCardIDs: [String], pinCardIDs: [String])
}

/// Thrown by `QualityCardStore.loadManifest(contentsOf:)`.
public struct QualityCardsManifestUndecodable: Error, CustomStringConvertible, Sendable, Equatable {
    public let path: String

    public init(path: String) {
        self.path = path
    }

    public var description: String {
        "quality-cards manifest at \(path) could not be read or decoded as fast-mlx-quality-card-v1"
    }
}

/// The effective quality-card manifest path this run resolved, or `.inactive` when no manifest
/// applies. Distinct from "the gate ran and admitted": `.inactive` means the gate never consulted a
/// manifest at all (no explicit `--quality-cards` and no conventional default present) — see
/// `QualityCardsManifestResolver.resolve`. Both cases are meant to be ANNOUNCED (e.g.
/// `quality_cards=<path>` / `quality_cards=none` in the startup line), closing the defect where a
/// server launched from an unexpected working directory silently ran with no gate and no way to
/// tell from the outside.
public enum QualityCardsManifestResolution: Sendable, Equatable {
    /// A manifest path this run will read from — ALWAYS ABSOLUTE, resolved once against the
    /// injected working directory. `explicit` distinguishes an operator-supplied `--quality-cards`
    /// value (whose absence/decode-failure must REFUSE startup — see
    /// `QualityCardsManifestPathNotFound` / `QualityCardsManifestUndecodable`) from the
    /// conventional default (whose absence/decode-failure fails open, admitting every model
    /// unmeasured, exactly like today).
    case active(path: URL, explicit: Bool)
    /// No explicit `--quality-cards` was supplied and the conventional default
    /// (`site/quality-guides.json` under the working directory) does not exist. The gate is
    /// inactive this run — every model admits unmeasured.
    case inactive
}

/// `--quality-cards <path>` was supplied but nothing exists at the resolved location. A hard
/// refusal: unlike the conventional default (whose absence is a normal, silent "gate inactive"
/// outcome), an operator who explicitly asked for gating and mistyped the path must find out at
/// startup, not discover months later that every model has been serving ungated.
public struct QualityCardsManifestPathNotFound: Error, CustomStringConvertible, Sendable, Equatable
{
    public let requestedPath: String
    public let resolvedPath: String

    public init(requestedPath: String, resolvedPath: String) {
        self.requestedPath = requestedPath
        self.resolvedPath = resolvedPath
    }

    public var description: String {
        "--quality-cards \(requestedPath) does not exist (resolved to \(resolvedPath))"
    }
}

/// Resolves the effective quality-card manifest path once at startup. Pure and unit-testable: the
/// working directory and the file-existence check are both INJECTED rather than read from
/// `FileManager.default` directly, so a test drives every branch (explicit missing, explicit
/// present, default absent, a relative explicit path) without touching the real filesystem or
/// depending on the test runner's own working directory.
///
/// Closes two defects in the original CWD-relative lookup: (1) the DEFAULT manifest path was
/// resolved relative to the process's current working directory with no announcement, so a server
/// launched from another directory silently ran with no quality gate and no way to tell; (2) an
/// EXPLICIT `--quality-cards` path that did not exist silently fell back to "inactive" (fail open)
/// instead of refusing — indistinguishable from an operator who never asked for gating at all.
public enum QualityCardsManifestResolver {
    /// The conventional default manifest location, relative to the working directory.
    static let conventionalDefaultRelativePath = "site/quality-guides.json"

    public static func resolve(
        explicitPath: String?,
        cwd: URL,
        fileExists: (URL) -> Bool = { FileManager.default.fileExists(atPath: $0.path) }
    ) throws -> QualityCardsManifestResolution {
        if let explicitPath {
            let resolved = absoluteURL(for: explicitPath, relativeTo: cwd)
            guard fileExists(resolved) else {
                throw QualityCardsManifestPathNotFound(
                    requestedPath: explicitPath, resolvedPath: resolved.path)
            }
            return .active(path: resolved, explicit: true)
        }
        let defaultURL = absoluteURL(for: conventionalDefaultRelativePath, relativeTo: cwd)
        return fileExists(defaultURL) ? .active(path: defaultURL, explicit: false) : .inactive
    }

    private static func absoluteURL(for path: String, relativeTo cwd: URL) -> URL {
        if path.hasPrefix("/") {
            return URL(fileURLWithPath: path).standardizedFileURL
        }
        return URL(fileURLWithPath: path, relativeTo: cwd).standardizedFileURL
    }
}

/// Parsed `--accept-quality <id-or-repo>` opt-in flags: the fail-closed CLI parser mirroring
/// `QuantPickPreference.validated`'s idiom, but additive rather than throwing — any string is a
/// valid card id or repo to elect, so there is no invalid VALUE to reject; absence of the flag simply
/// means "not opted in", which is the unchanged default behavior. An operator may pass the flag more
/// than once (electing several cards across a session's model switches); all values are collected.
public struct QualityOptIn: Sendable, Equatable {
    public let acceptedIDs: Set<String>

    public init(acceptedIDs: Set<String> = []) {
        self.acceptedIDs = acceptedIDs
    }

    /// `true` when the operator elected this card's `id`, its `model.repo`, or its `model.hfPin` —
    /// whichever identity is more convenient at the CLI. `nil` for an absent card: nothing to elect.
    /// Mirrors Python `is_opted_in` (`scripts/fastmlx_launch.py:1099-1109`), which is the reference
    /// semantics this Swift path must match — a card whose `model.repo` is `nil` (e.g. a shipped
    /// NO_GO card identified only by `hfPin`) would otherwise only be electable by card id, which is
    /// the escape hatch this overload exists to restore before a later increment teaches card lookup
    /// itself to resolve by pin.
    public func isElected(card: QualityCard?) -> Bool {
        guard let card else { return false }
        return isElected(cardID: card.id, repoID: card.model.repo, hfPin: card.model.hfPin)
    }

    /// `true` when `acceptedIDs` contains any non-nil one of `cardID`, `repoID`, `hfPin`. `hfPin` has
    /// no default value deliberately: the bug class this widening closes is "a field that is decoded
    /// and never read" (a repo-less card's `hfPin` was exactly that until this change) — a defaulted
    /// parameter would let a future call site silently reintroduce the same gap by omitting it, so
    /// every caller is forced to pass an explicit choice (`nil` included) rather than inherit one.
    public func isElected(cardID: String?, repoID: String?, hfPin: String?) -> Bool {
        if let cardID, acceptedIDs.contains(cardID) { return true }
        if let repoID, acceptedIDs.contains(repoID) { return true }
        if let hfPin, acceptedIDs.contains(hfPin) { return true }
        return false
    }

    /// Scan a raw argument list for `--accept-quality <value>` occurrences. Additive/no-op absent
    /// the flag: an argument list with none produces `QualityOptIn()`, the empty/not-opted-in state,
    /// so a serve invocation that never mentions the flag is unchanged.
    public static func parse<S: Sequence>(_ arguments: S) -> QualityOptIn where S.Element == String {
        let args = Array(arguments)
        var ids: Set<String> = []
        var index = 0
        while index < args.count {
            if args[index] == "--accept-quality", index + 1 < args.count {
                ids.insert(args[index + 1])
                index += 2
            } else {
                index += 1
            }
        }
        return QualityOptIn(acceptedIDs: ids)
    }
}
