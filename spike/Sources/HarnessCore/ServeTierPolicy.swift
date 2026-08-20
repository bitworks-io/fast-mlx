import Foundation

/// The operator-intent dial for the serve route (differentiator #4 slice). Each tier is a stance on
/// the fidelity-vs-fit tradeoff the sizer exposes, resolved into a concrete `ServingPolicy` the
/// quant auto-picker and serve path consult. The tier says *what the operator wants*; the sizer math
/// says *what fits* — this enum never changes a verdict, only which KV tiers/context moves the picker
/// is ALLOWED to make on the operator's behalf.
public enum ServeTier: String, Sendable, CaseIterable {
    /// Fidelity-first: fp16 KV only, and never silently cap context — refuse instead. "What you
    /// asked for is what you get, with no hidden downgrades."
    case transparent
    /// Trade KV precision to fit, but never silently shorten context: try fp16, escalate to the one
    /// measured lossy KV tier (int8) to hold the FULL requested context, and refuse rather than cap
    /// if even int8 can't. "I'll lower KV precision to fit your whole context, but I won't quietly
    /// give you less of it."
    case balanced
    /// Cram it in: the same measured KV tiers as balanced, but capping is ALSO allowed — escalate KV
    /// AND shorten context to the ceiling when that is what makes the model fit. The most aggressive
    /// fit the codebase can HONESTLY plan (see `ServingPolicy.allowedKVTiers` on why the unmeasured
    /// turbo/tq tiers are excluded even here).
    case maxfit
}

/// Fail-closed error for an unrecognized `--tier` value, mirroring the `--kv-quant` validation idiom
/// so ServingCore can carry the raw string and validate it in HarnessCore at the serve call site.
public enum ServeTierError: Error, CustomStringConvertible, Equatable {
    case unknownTier(String)
    public var description: String {
        switch self {
        case .unknownTier(let raw):
            return "Unknown --tier '\(raw)'. Valid values: "
                + ServeTier.allCases.map(\.rawValue).joined(separator: "|") + "."
        }
    }
}

extension ServeTier {
    /// Parse a raw `--tier` value, failing closed on an unknown tier (never a silent default).
    public static func validated(_ raw: String) throws -> ServeTier {
        guard let tier = ServeTier(rawValue: raw) else { throw ServeTierError.unknownTier(raw) }
        return tier
    }
}

/// The resolved policy a serve invocation runs under: the concrete search space + guard rails the
/// auto-picker and fit-check obey. Pure value type — no MLX, no filesystem.
public struct ServingPolicy: Sendable {
    public let tier: ServeTier
    /// KV precision tiers the auto-picker MAY select, ordered highest-fidelity first (the picker
    /// escalates down this list). Only tiers with a real/measured reference implementation appear —
    /// see the honesty note on `ServeTierPolicy`.
    public let allowedKVTiers: [KVQuantTier]
    /// `true` when the tier permits capping the served context to the computed ceiling to make a
    /// model fit; `false` (transparent) means an over-ceiling request is refused, not silently cut.
    public let allowContextCapping: Bool
    /// Capacity thresholds the verdict is classified against. Held at `.default` for every tier for
    /// now: the tiers differ on *which moves are allowed*, not on where red begins — loosening the
    /// red boundary per-tier would weaken the fail-closed guarantee and is deliberately not done here.
    public let thresholds: CapacityThresholds
    /// Non-nil when an explicit `--kv-quant` override departs from the tier's default fidelity
    /// contract (e.g. `int8` under `transparent`): the pin still wins, but the departure is announced
    /// so the operator is never surprised that the tier's promise was overridden.
    public let conflictAnnotation: String?

    public init(
        tier: ServeTier, allowedKVTiers: [KVQuantTier], allowContextCapping: Bool,
        thresholds: CapacityThresholds = .default, conflictAnnotation: String? = nil
    ) {
        self.tier = tier
        self.allowedKVTiers = allowedKVTiers
        self.allowContextCapping = allowContextCapping
        self.thresholds = thresholds
        self.conflictAnnotation = conflictAnnotation
    }
}

/// Resolves a `ServeTier` (plus any explicit operator overrides) into the concrete `ServingPolicy`.
///
/// **Honesty rule (why maxfit is not "anything goes"):** the KV quant enum carries experimental
/// placeholder tiers (`turbo4`, `tq2_5`, `tq3_5`) whose bytes/element are DOCUMENTED ESTIMATES, not
/// measured — no build of them exists in this codebase (see `KVQuantTier`). Auto-selecting one would
/// hand the operator a "fits" plan resting on an unvalidated footprint, exactly the measured-vs-
/// modeled dishonesty the fit-check exists to prevent. So even `maxfit` restricts auto-pick to the
/// two tiers with a real reference (`fp16`, and `int8` ≈ mlx-swift's own 32-wide K/V int8 groups). An
/// operator may still *explicitly* pin an experimental tier via `--kv-quant`; that is their informed
/// choice and is surfaced with a conflict annotation rather than chosen silently.
public enum ServeTierPolicy {

    /// The measured/auto-selectable KV tiers (a real/measured reference footprint exists), highest
    /// fidelity first. Public so the agreement test can freeze the honesty chain `qualityApproved ⊆
    /// runtimeWired ⊆ measured` and every tier's `allowedKVTiers ⊆ measured`.
    public static let measuredKVTiers: [KVQuantTier] = [.fp16, .int8]

    /// The KV tiers the SERVING RUNTIME can actually store today (a CAPABILITY set). The fit-check on
    /// the ENFORCED (model-loading) serve path must not claim a compressed-KV ceiling the runtime cannot
    /// honor: the runtime stores KV in fp16 (int8 runtime KV is metallib-gated and not yet wired). A
    /// future int8-KV runtime wiring flips exactly this member. Contrast `measuredKVTiers`, which is what
    /// the auto-pick *planner* may explore (fine on the `--quant-pick-only` dry-run, which loads
    /// nothing), and `qualityApprovedKVTiers`, which is what the enforced path may actually SERVE.
    public static let runtimeWiredKVTiers: [KVQuantTier] = [.fp16]

    /// The KV tiers whose LONG-SESSION QUALITY has been measured and approved for the auto-picker to
    /// serve on the operator's behalf (a QUALITY set). Frozen to `[.fp16]` until a dated big-box (M5
    /// 128 GB) long-context int8 quality measurement PASSES — see
    /// `docs/task-inbox/2026-08-19-runtime-kv-quant-quality.md`. Deliberately DISTINCT from
    /// `runtimeWiredKVTiers`: the runtime being ABLE to store a KV format (capability) is not license to
    /// auto-SERVE it (quality). This split is what lets int8 runtime wiring land WITHOUT silently
    /// promoting int8 to a served default — wiring flips `runtimeWiredKVTiers` alone; int8 becomes
    /// auto-servable only when a dated quality PASS also adds it here. Invariant frozen by
    /// `ServeTierKVTierAgreementTests`: `qualityApproved ⊆ runtimeWired ⊆ measured`.
    public static let qualityApprovedKVTiers: [KVQuantTier] = [.fp16]

    /// The KV tiers the ENFORCED (model-loading) serve path may actually store AND serve by default: a
    /// tier must be BOTH runtime-wired (the runtime can store it) AND quality-approved (its long-session
    /// quality is measured). Keying the enforced-path gate HERE — not on `runtimeWiredKVTiers` — is what
    /// prevents a future int8 runtime wiring from silently auto-promoting int8 to a served default before
    /// its quality is proven. Equals `qualityApprovedKVTiers` while the invariant `qualityApproved ⊆
    /// runtimeWired` holds (frozen by the agreement test); computed as the intersection so it stays
    /// correct if the sets ever transiently diverge.
    public static var enforcedServableKVTiers: [KVQuantTier] {
        let wired = Set(runtimeWiredKVTiers)
        return qualityApprovedKVTiers.filter { wired.contains($0) }
    }

    /// One-line advisory for the ENFORCED serve path: when the resolved tier's auto-pick set would
    /// escalate KV below fp16 (balanced/maxfit → int8) to hold more context, but the serving runtime
    /// stores KV in fp16 only, the enforced path keeps fp16 and returns this note so the operator knows
    /// the escalation the tier promised is not applied on a LOADED model. Returns `nil` when no `--tier`
    /// was given, or when the tier asks for nothing beyond the wired set (transparent, or an fp16-only
    /// pin) — so the default serve path stays silent and byte-identical. The KV escalation stays fully
    /// available on the `--quant-pick-only` dry-run, which is a plan and loads no model.
    public static func enforcedPathKVAdvisory(for policy: ServingPolicy?) -> String? {
        guard let policy else { return nil }
        // Gate on what the enforced path may actually SERVE (runtime-wired AND quality-approved), not on
        // capability alone — so wiring a KV format into the runtime never auto-promotes it to a served
        // default before its quality is proven. Today `enforcedServableKVTiers == runtimeWiredKVTiers ==
        // [.fp16]`, so this advisory is byte-identical to the capability-keyed version.
        let servable = Set(enforcedServableKVTiers)
        let unhonored = policy.allowedKVTiers.filter { !servable.contains($0) }
        guard !unhonored.isEmpty else { return nil }
        let names = unhonored.map(\.rawValue).joined(separator: ", ")
        return "serve-tier note: \(policy.tier.rawValue) would escalate KV to [\(names)] to hold more "
            + "context, but the serving runtime stores KV in fp16 only (runtime_not_wired); the enforced "
            + "serve keeps fp16. KV escalation stays available on --quant-pick-only."
    }

    private static func defaultAllowedTiers(_ tier: ServeTier) -> [KVQuantTier] {
        switch tier {
        case .transparent: return [.fp16]
        case .balanced, .maxfit: return measuredKVTiers
        }
    }

    private static func defaultAllowsCapping(_ tier: ServeTier) -> Bool {
        switch tier {
        // Only maxfit will silently shorten context. transparent and balanced both refuse a
        // memory-bound cap — they differ on whether a lossy KV tier may be used to AVOID the cap
        // (transparent: no, fp16-only; balanced: yes, escalate to int8 to hold full context).
        case .transparent, .balanced: return false
        case .maxfit: return true
        }
    }

    /// Resolve `tier` into a `ServingPolicy`. `explicitKVQuant` (a validated `--kv-quant`) pins the
    /// auto-pick set to exactly that tier and, when it is outside the tier's default fidelity set,
    /// records a `conflictAnnotation`. `explicitContext` is accepted for symmetry with the serve
    /// call site; context capping is governed by the tier and asserted by the caller, so it does not
    /// change the returned set here.
    public static func resolve(
        tier: ServeTier, explicitKVQuant: KVQuantTier? = nil, explicitContext: Int? = nil
    ) -> ServingPolicy {
        let defaults = defaultAllowedTiers(tier)

        guard let pinned = explicitKVQuant else {
            return ServingPolicy(
                tier: tier, allowedKVTiers: defaults, allowContextCapping: defaultAllowsCapping(tier))
        }

        // Explicit pin wins — but if it leaves the tier's default fidelity set, say so.
        let conflict: String?
        if defaults.contains(pinned) {
            conflict = nil
        } else {
            conflict = "--kv-quant \(pinned.rawValue) overrides the \(tier.rawValue) tier's default "
                + "KV set (\(defaults.map(\.rawValue).joined(separator: ", "))); honoring the explicit pin"
        }
        return ServingPolicy(
            tier: tier, allowedKVTiers: [pinned], allowContextCapping: defaultAllowsCapping(tier),
            conflictAnnotation: conflict)
    }
}
