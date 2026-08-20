import Foundation

/// The operator's ranking preference for the quant auto-pick's cross-candidate comparison. `context`
/// (the default) keeps served context primary — the shipped behavior: silently shipping less context
/// to gain quality bits is the worse surprise. `quality` hoists quantization bits to the primary key —
/// the operator explicitly values fidelity over context length, so the highest-bit build that still
/// fits wins even when a lower-bit build would serve more context. Neither changes the fit-check: an
/// ineligible (red) candidate is never selected under either preference. See
/// docs/task-inbox/2026-08-18-quant-auto-pick-policy.md (decision #1).
public enum QuantPickPreference: String, Sendable, CaseIterable {
    case context
    case quality
}

/// Fail-closed error for an unrecognized `--prefer` value, mirroring the `--tier`/`--kv-quant`
/// validation idiom so ServingCore can carry the raw string and validate it in HarnessCore at the
/// serve call site.
public enum QuantPickPreferenceError: Error, CustomStringConvertible, Equatable {
    case unknownPreference(String)
    public var description: String {
        switch self {
        case .unknownPreference(let raw):
            return "Unknown --prefer '\(raw)'. Valid values: "
                + QuantPickPreference.allCases.map(\.rawValue).joined(separator: "|") + "."
        }
    }
}

extension QuantPickPreference {
    /// Parse a raw `--prefer` value, failing closed on an unknown mode (never a silent default).
    public static func validated(_ raw: String) throws -> QuantPickPreference {
        guard let mode = QuantPickPreference(rawValue: raw) else {
            throw QuantPickPreferenceError.unknownPreference(raw)
        }
        return mode
    }
}

/// One quantization variant of a model, ready to be fit-checked. `parsed == nil` marks a candidate
/// the decoder could not model (e.g. an unsupported `model_type`): it is carried through so the
/// picker can *exclude it with a reason* rather than have the whole selection fail — never silently
/// dropped. `repoID` is the fetchable identity (HF repo path) the serve path downloads on a win; it
/// may differ from `parsed?.profile.id`.
public struct QuantServeCandidate: Sendable {
    public let repoID: String
    public let parsed: ParsedModelArch?
    public let exclusionReason: String?

    public init(repoID: String, parsed: ParsedModelArch?, exclusionReason: String? = nil) {
        self.repoID = repoID
        self.parsed = parsed
        self.exclusionReason = exclusionReason
    }
}

/// The fit outcome for a single candidate: its `ServingFitDecision` when the fit-check ran, or an
/// `exclusionReason` when it was excluded before the check (unmodeled arch). `eligible` is the
/// selection predicate — a candidate that fit-checked and did not land red.
public struct QuantCandidateEvaluation: Sendable {
    public let repoID: String
    public let quantBits: Int?
    public let decision: ServingFitDecision?
    /// The KV precision tier the `decision` above was computed at — the tier the picker escalated to
    /// for THIS candidate (highest fidelity that still fits, or the highest-fidelity tier tried when
    /// it lands red). `nil` for an excluded/unmodeled candidate the fit-check never ran on.
    public let chosenKVTier: KVQuantTier?
    public let exclusionReason: String?

    public init(
        repoID: String, quantBits: Int?, decision: ServingFitDecision?,
        chosenKVTier: KVQuantTier? = nil, exclusionReason: String?
    ) {
        self.repoID = repoID
        self.quantBits = quantBits
        self.decision = decision
        self.chosenKVTier = chosenKVTier
        self.exclusionReason = exclusionReason
    }

    /// The candidate can actually be served: it fit-checked and did not fail closed on red.
    public var eligible: Bool { decision?.shouldProceed == true }
}

/// The auto-pick outcome: the winning quant (if any), its decision, and the full per-candidate
/// evaluation set for the operator-facing announce (so *why* each loser lost is surfaced, not just
/// the winner).
public struct QuantPickResult: Sendable {
    public let shouldProceed: Bool
    public let winnerRepoID: String?
    public let winnerDecision: ServingFitDecision?
    public let evaluations: [QuantCandidateEvaluation]
    /// The ranking preference the pick ran under. `.context` (default) is the shipped context-first
    /// behavior; `.quality` hoisted quant bits to primary. Surfaced additively in the announce/machine
    /// line ONLY when non-default, so every existing default-preference line stays byte-identical.
    public let preference: QuantPickPreference

    public init(
        shouldProceed: Bool, winnerRepoID: String?, winnerDecision: ServingFitDecision?,
        evaluations: [QuantCandidateEvaluation], preference: QuantPickPreference = .context
    ) {
        self.shouldProceed = shouldProceed
        self.winnerRepoID = winnerRepoID
        self.winnerDecision = winnerDecision
        self.evaluations = evaluations
        self.preference = preference
    }

    private static func gib(_ b: Int) -> String { String(format: "%.2f GiB", Double(b) / 1_073_741_824.0) }

    /// Operator-facing summary: one line per candidate (bits, color, served context, or the
    /// exclusion reason) then the winner / refusal verdict. Mirrors `ServingFitDecision.summaryLines`
    /// so the announce and a refusal read identically.
    public func summaryLines() -> [String] {
        // Name the ranking axis only when the operator departed from the context-first default, so the
        // default announce header stays byte-identical to the pre-preference behavior.
        let preferNote = preference == .context ? "" : " (prefer=\(preference.rawValue))"
        var lines: [String] = ["quant auto-pick: \(evaluations.count) candidate(s)\(preferNote)"]
        for e in evaluations {
            let bits = e.quantBits.map { "\($0)-bit" } ?? "unquantized"
            if let d = e.decision {
                let cap = d.contextWasCapped ? " (capped from \(d.requestedContext))" : ""
                let mark = e.eligible ? "" : " EXCLUDED"
                // Additive KV-tier note (only when escalated off fp16), so the fp16 line is unchanged.
                let kvNote = e.chosenKVTier.flatMap { $0 == .fp16 ? nil : " kv=\($0.rawValue)" } ?? ""
                lines.append("  \(e.repoID) [\(bits)] \(d.color.rawValue) ctx=\(d.servedContext)\(cap)\(kvNote)\(mark)")
            } else {
                lines.append("  \(e.repoID) [\(bits)] EXCLUDED: \(e.exclusionReason ?? "not fit-checkable")")
            }
        }
        if let winner = winnerRepoID, let d = winnerDecision {
            let winnerEval = evaluations.first { $0.repoID == winner }
            let bits = winnerEval?.quantBits.map { "\($0)-bit" } ?? "unquantized"
            // Name the KV tier only when the picker escalated off the fp16 default — additive, so the
            // fp16 winner line stays byte-identical to the pre-escalation announce.
            let kvNote = (winnerEval?.chosenKVTier).flatMap { $0 == .fp16 ? nil : " kv=\($0.rawValue)" } ?? ""
            lines.append("  WINNER: \(winner) [\(bits)] serving ctx=\(d.servedContext) (\(d.color.rawValue))\(kvNote)")
        } else {
            // No `--force` clause: forcing the best red candidate in candidates mode is not wired
            // (see docs/task-inbox/2026-08-18-quant-auto-pick-policy.md), so advertising it here would
            // mislead — the serve path refuses a red-only set regardless of --force.
            lines.append("  REFUSED: no quant fits this host. Re-run with a lighter model or more RAM.")
        }
        return lines
    }

    /// One machine-readable line naming the winning quant + its fit verdict, for the `--quant-pick-only`
    /// dry-run (a script captures it: `WINNER=$(fastmlx-serve --quant-pick-only …)`). Frozen key
    /// set/order — `nil` when the pick refused (no candidate fits), so the caller fails closed on the
    /// same refusal path as the serve route instead of emitting a bogus winner.
    public func machineReadableWinnerLine() -> String? {
        guard let winner = winnerRepoID, let d = winnerDecision else { return nil }
        let bits = d.quantBits.map(String.init) ?? "none"
        var line = "quant_pick winner=\(winner) quant_bits=\(bits) "
            + "fit_check=\(d.color.rawValue) fit_served_context=\(d.servedContext)"
        // Additive, opt-in field appended AFTER the frozen base contract: present only when the picker
        // escalated the KV tier off the fp16 default (a script that pinned no allowed tiers, or whose
        // winner fit at fp16, sees the byte-identical original line — the frozen contract holds).
        if let tier = (evaluations.first { $0.repoID == winner }?.chosenKVTier), tier != .fp16 {
            line += " kv_tier=\(tier.rawValue)"
        }
        // Additive, opt-in: appended only when the operator picked a non-default ranking axis, so a
        // context-first pick (the default) emits the byte-identical frozen base line.
        if preference != .context {
            line += " prefer=\(preference.rawValue)"
        }
        return line
    }
}

/// Pick the best-fitting quantization of a model for a host (fit-checked-serve, differentiator #2
/// full shape). Pure: each candidate is fit-checked through `ServingFitPlanner.decide`, so the whole
/// selection policy is unit-tested off-box; the metallib-gated live serve only confirms the winner
/// loads. Adds no new capacity math — it composes the existing per-model verdict engine.
///
/// **Selection policy** (resolves the comparison-context ambiguity the sizer left open, see
/// docs/task-inbox/2026-08-18-quant-auto-pick-policy.md): all candidates are fit-checked at the SAME
/// requested context; among the non-red ones, rank by
/// `(servedContext desc, color green>yellow, quant bits desc [nil = unquantized ranks highest],
/// repoID asc)`. Context served is primary because it is the operator's explicit ask (default =
/// native max); silently trading context for quality bits is the worse surprise. `--force` is
/// intentionally NOT applied here — a red candidate is never auto-selected.
public enum QuantAutoPicker {

    public static func pick(
        candidates: [QuantServeCandidate], host: SystemProfile,
        requestedContext: Int? = nil, kvQuant: KVQuantTier = .fp16, concurrency: Int = 1,
        thresholds: CapacityThresholds = .default, allowedKVTiers: [KVQuantTier]? = nil,
        allowContextCapping: Bool = true, preference: QuantPickPreference = .context
    ) -> QuantPickResult {
        // Backward compatible: with no allowed-tier set, the picker runs at the single `kvQuant`
        // exactly as before (default `[.fp16]`). A `ServingPolicy.allowedKVTiers` (from the serve
        // dial) turns on joint weight-quant × KV-tier escalation — highest fidelity first.
        let tiers = allowedKVTiers ?? [kvQuant]

        let evaluations: [QuantCandidateEvaluation] = candidates.map { c in
            guard let parsed = c.parsed else {
                return QuantCandidateEvaluation(
                    repoID: c.repoID, quantBits: nil, decision: nil, chosenKVTier: nil,
                    exclusionReason: c.exclusionReason)
            }
            let (tier, decision) = bestTierDecision(
                parsed: parsed, host: host, requestedContext: requestedContext, tiers: tiers,
                concurrency: concurrency, thresholds: thresholds, allowContextCapping: allowContextCapping)
            return QuantCandidateEvaluation(
                repoID: c.repoID, quantBits: parsed.quantBits, decision: decision,
                chosenKVTier: tier, exclusionReason: nil)
        }

        let winner = evaluations
            .filter { $0.eligible }
            .sorted { Self.isBetter($0, $1, preference: preference) }
            .first

        return QuantPickResult(
            shouldProceed: winner != nil, winnerRepoID: winner?.repoID,
            winnerDecision: winner?.decision, evaluations: evaluations, preference: preference)
    }

    /// Escalate one candidate down the allowed KV tiers (highest fidelity first) and return the tier
    /// the picker commits to for it, with its fit-check. Among tiers where the candidate is eligible,
    /// pick the one that serves the MOST context, breaking ties toward higher KV fidelity — so fp16 is
    /// kept whenever it already serves the full ask, and a lossy tier is taken only to buy more
    /// context. When no tier fits, the highest-fidelity tier (`tiers[0]`) carries the red verdict for
    /// the operator-facing exclusion line ("even at best fidelity, red").
    private static func bestTierDecision(
        parsed: ParsedModelArch, host: SystemProfile, requestedContext: Int?,
        tiers: [KVQuantTier], concurrency: Int, thresholds: CapacityThresholds,
        allowContextCapping: Bool
    ) -> (KVQuantTier, ServingFitDecision) {
        func decide(_ kv: KVQuantTier) -> ServingFitDecision {
            ServingFitPlanner.decide(
                profile: parsed.profile, weightsAreMeasured: parsed.weightsAreMeasured, host: host,
                requestedContext: requestedContext, kvQuant: kv, concurrency: concurrency,
                force: false, thresholds: thresholds, quantBits: parsed.quantBits,
                weightsAreDeclared: parsed.weightsAreDeclared, allowContextCapping: allowContextCapping)
        }
        let evaluated = tiers.map { ($0, decide($0)) }
        let eligible = evaluated.filter { $0.1.shouldProceed }
        if let best = eligible.max(by: { Self.tierIsWorse($0, $1) }) {
            return best
        }
        return evaluated.first! // tiers is non-empty; tiers[0] is the highest-fidelity red verdict.
    }

    /// `true` when `lhs` is the WORSE per-candidate tier choice (so `max` returns the best): fewer
    /// served-context tokens, or — at equal context — the lower-fidelity KV tier (more bytes/element =
    /// higher fidelity). Keeps fp16 whenever it already serves the full requested context.
    private static func tierIsWorse(
        _ lhs: (KVQuantTier, ServingFitDecision), _ rhs: (KVQuantTier, ServingFitDecision)
    ) -> Bool {
        if lhs.1.servedContext != rhs.1.servedContext { return lhs.1.servedContext < rhs.1.servedContext }
        return lhs.0.bytesPerElement < rhs.0.bytesPerElement
    }

    /// `true` when `lhs` should rank ahead of `rhs`. Under the default `.context` preference the axis
    /// order is: served context, then KV fidelity (fp16 over a lossy tier — never silently trade
    /// fidelity for a same-context green-vs-yellow win), then green>yellow, then higher quant bits
    /// (unquantized highest), then deterministic repoID. Under `.quality` the operator asked for
    /// fidelity over length, so quant bits are hoisted to the PRIMARY key (the highest-bit build that
    /// still fits wins even when a lower-bit build serves more context); every remaining tiebreak is
    /// unchanged. Both candidates are eligible, so `decision`/`chosenKVTier` are non-nil.
    private static func isBetter(
        _ lhs: QuantCandidateEvaluation, _ rhs: QuantCandidateEvaluation,
        preference: QuantPickPreference
    ) -> Bool {
        let l = lhs.decision!, r = rhs.decision!
        if preference == .quality {
            let lq = qualityRank(lhs.quantBits), rq = qualityRank(rhs.quantBits)
            if lq != rq { return lq > rq }
        }
        if l.servedContext != r.servedContext { return l.servedContext > r.servedContext }
        let lf = lhs.chosenKVTier?.bytesPerElement ?? 0, rf = rhs.chosenKVTier?.bytesPerElement ?? 0
        if lf != rf { return lf > rf }
        let lc = colorRank(l.color), rc = colorRank(r.color)
        if lc != rc { return lc > rc }
        let lq = qualityRank(lhs.quantBits), rq = qualityRank(rhs.quantBits)
        if lq != rq { return lq > rq }
        // Equal bits + equal fit: a DWQ build recovers accuracy the naive quantization loses (mlx-lm
        // LEARNED_QUANTS.md), so it outranks the plain build of the same width — before falling back to
        // the deterministic repoID order (which would otherwise hand the tie to the plain build, since
        // "…-4bit" sorts before "…-4bit-DWQ").
        let ld = QuantCandidateSourcer.denotesDWQ(repoID: lhs.repoID)
        let rd = QuantCandidateSourcer.denotesDWQ(repoID: rhs.repoID)
        if ld != rd { return ld }
        return lhs.repoID < rhs.repoID
    }

    private static func colorRank(_ c: CapacityColor) -> Int {
        switch c { case .green: return 2; case .yellow: return 1; case .red: return 0 }
    }

    /// Higher = better quality. `nil` (no quantization block) is an unquantized checkpoint — the
    /// highest quality — so it ranks above any finite bit width.
    private static func qualityRank(_ bits: Int?) -> Int { bits ?? Int.max }
}
