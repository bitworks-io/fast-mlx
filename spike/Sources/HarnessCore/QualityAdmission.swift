import Foundation

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
/// message quotes. Decoded leniently (`CodingKeys` names only the fields consumed here) so new
/// emitter-side fields (`rawMetrics`, `provenance`, `boundary`, …) never break this decode.
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

    /// The slice of `config` this gate consults: `residency`, the only field that changes whether a
    /// card is even ELIGIBLE to gate a resident Swift launch. Decoded leniently — a missing or `null`
    /// `residency` is `nil` here (→ "resident" via `QualityCard.effectiveResidency`), and any other
    /// emitter-side `config` field (`quant`, `enhancement`, `hardwareClass`, …) is ignored, never
    /// failing this decode.
    public struct Config: Sendable, Decodable, Equatable {
        public let residency: String?

        public init(residency: String? = nil) {
            self.residency = residency
        }
    }

    public let id: String
    public let model: Model
    public let verdict: QualityVerdict
    public let admission: Admission
    public let legible: Legible
    /// `nil` when the card predates the `config.residency` field, or when its `config` object omits
    /// `residency` — both mean "measured resident" per `effectiveResidency` below.
    public let config: Config?

    public init(
        id: String, model: Model, verdict: QualityVerdict, admission: Admission, legible: Legible,
        config: Config? = nil
    ) {
        self.id = id
        self.model = model
        self.verdict = verdict
        self.admission = admission
        self.legible = legible
        self.config = config
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

/// The top-level `site/quality-guides.json` manifest envelope
/// (`docs/quality-card-schema-v1.md`'s "Top-level manifest"). Decoded leniently — `schema` and
/// `generatedAt` are carried but not asserted here; only `cards` is consulted.
struct QualityCardManifest: Decodable {
    let cards: [QualityCard]
}

/// Loads a `QualityCard` for a resolved repo out of a `fast-mlx-quality-card-v1` manifest. Fails
/// closed to "no card" (→ `QualityAdmission.decide` → `.admitUnmeasured`) on any decode failure or
/// missing file, so a serve with no manifest, an unreadable manifest, or a manifest that fails to
/// parse behaves EXACTLY as today — never a refusal caused by a broken loader.
public enum QualityCardStore {
    /// Pure core: decode `manifestData` as a `fast-mlx-quality-card-v1` manifest and return the first
    /// card whose `model.repo` matches `repoID` AND whose `effectiveResidency` is `"resident"`
    /// (`QualityCard.matchesResidentLaunch`). This engine only serves resident weights — it cannot
    /// stream experts from SSD — and streaming a model changes its greedy-decode output relative to a
    /// resident load, so a card measured under `"expert-stream"` (or any unrecognized residency
    /// string) must never match here, however it is ordered in the manifest: it is treated exactly
    /// like "no card for this repo", i.e. `.admitUnmeasured`, never as a resident admission input.
    /// `nil` on an unknown repo, a repo with only non-resident cards, or a decode failure.
    public static func card(forRepo repoID: String, in manifestData: Data) -> QualityCard? {
        guard let manifest = try? JSONDecoder().decode(QualityCardManifest.self, from: manifestData)
        else {
            return nil
        }
        return card(forRepo: repoID, in: manifest.cards)
    }

    /// The single card-selection rule over already-decoded cards: the first card for `repoID` that
    /// was measured resident. Every Swift call site that picks a card by identity goes through here,
    /// so the residency filter cannot be bypassed by a caller that loaded the manifest itself.
    public static func card(forRepo repoID: String, in cards: [QualityCard]) -> QualityCard? {
        cards.first { $0.model.repo == repoID && $0.matchesResidentLaunch }
    }

    /// Convenience: read `manifestURL` and decode it. Returns `nil` (never throws) when the file is
    /// missing or unreadable — the no-manifest-today behavior is unchanged.
    public static func card(forRepo repoID: String, manifestURL: URL) -> QualityCard? {
        guard let data = try? Data(contentsOf: manifestURL) else { return nil }
        return card(forRepo: repoID, in: data)
    }

    /// Strict decode: throws `QualityCardsManifestUndecodable` on an unreadable file or a payload
    /// that fails `fast-mlx-quality-card-v1` decoding, instead of `card(forRepo:manifestURL:)`'s
    /// silent nil. Used only for an EXPLICITLY supplied `--quality-cards` path — see
    /// `QualityCardsManifestResolver`'s doc comment for why the conventional default keeps the
    /// silent fail-open behavior via the convenience method above instead.
    public static func loadManifest(contentsOf url: URL) throws -> [QualityCard] {
        guard let data = try? Data(contentsOf: url),
            let manifest = try? JSONDecoder().decode(QualityCardManifest.self, from: data)
        else {
            throw QualityCardsManifestUndecodable(path: url.path)
        }
        return manifest.cards
    }
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

    /// `true` when the operator elected either this card's `id` or its `model.repo` — whichever
    /// identity is more convenient at the CLI.
    public func isElected(cardID: String?, repoID: String?) -> Bool {
        if let cardID, acceptedIDs.contains(cardID) { return true }
        if let repoID, acceptedIDs.contains(repoID) { return true }
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
