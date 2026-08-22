import Foundation

/// The first brick of the adaptive concurrency-aware serving mode: a pure, stateless policy that
/// decides whether SOLO SPECULATION is enabled.
///
/// **What this does NOT do:** the continuous-batch scheduler's `makeTick()` already decides
/// solo-vs-batch from the in-flight candidate count (1 → solo single-stream, ≥2 → batch). This type
/// does not re-decide that — a second mode authority would be a bug, and `mode(for:)` freezes that by
/// making the `>= 2` branch unconditional (never overridable by warmup or queue signals).
///
/// **What this DOES do:** given the scheduler's own signals, it decides whether the solo path may
/// additionally draft speculative tokens, with two fail-safes:
///   1. **Hysteresis** — after a 1↔2 membership crossing, speculation stays suppressed for
///      `speculationWarmupTicks` ticks, so a request count flapping across the boundary doesn't thrash
///      speculation on/off tick over tick.
///   2. **Companion suppression** — when a companion request is imminent (queued or prefilling),
///      speculation is suppressed even with zero warmup remaining: drafting tokens now only to have a
///      batch drain immediately discard them wastes compute for no benefit.
///
/// This is also the seam MTP speculative decode will later plug into: MTP wiring is a matter of the
/// caller ANDing its own per-request speculation flag with `mode(for:) == .soloSpeculative`.
///
/// Pure function; all state (the tick counter) is the CALLER's responsibility — this type never stores
/// mutable state, mirroring `ServeTierPolicy`'s pure value-type idiom.
public struct AdaptiveSoloSpeculationPolicy: Sendable, Equatable {

    /// The scheduler-observed inputs the policy decides against. All counts, never state; the caller
    /// (the continuous-batch scheduler) recomputes/passes these fresh every tick.
    public struct Signals: Sendable, Equatable {
        /// Ready/decoding requests in the selected cohort. 0 → idle, 1 → solo, >=2 → batch — this is
        /// the SAME count the scheduler's own solo-vs-batch mode decision is keyed on; this policy
        /// reads it but never redefines the solo/batch boundary itself.
        public let decodeCandidates: Int
        /// Requests currently prefilling: not yet decoding, but about to join the decode cohort. A
        /// nonzero count here means a companion is imminent even though `decodeCandidates` may still
        /// read 1 — speculation is suppressed pre-emptively rather than waiting for the count to flip.
        public let prefillingRequests: Int
        /// Requests admitted but not yet decoding or prefilling. Same imminent-companion logic as
        /// `prefillingRequests`: counted separately because the two arrive from different admission
        /// stages, but combined identically for the suppression check below.
        public let queuedRequests: Int
        /// Ticks elapsed since the last solo/batch membership change (a 1↔2 crossing in either
        /// direction). The CALLER owns this counter — this type is pure and stores no state — and is
        /// expected to reset it to 0 on any membership change, incrementing it otherwise.
        public let ticksSinceMembershipChange: Int

        public init(
            decodeCandidates: Int, prefillingRequests: Int, queuedRequests: Int,
            ticksSinceMembershipChange: Int
        ) {
            self.decodeCandidates = decodeCandidates
            self.prefillingRequests = prefillingRequests
            self.queuedRequests = queuedRequests
            self.ticksSinceMembershipChange = ticksSinceMembershipChange
        }
    }

    /// The resolved decision for this tick.
    public enum Mode: Sendable, Equatable {
        /// No decode candidates at all — nothing to speculate for.
        case idle
        /// >=2 candidates: speculation is structurally impossible (the solo speculative path only
        /// exists for a single in-flight stream) — this branch is NEVER policy-overridable, by
        /// warmup or by any signal, so it can never become a second mode authority.
        case batch
        /// Exactly 1 candidate, but speculation is suppressed this tick (warmup window still open, or
        /// a companion request is queued/prefilling).
        case soloPlain
        /// Exactly 1 candidate and speculation is permitted. The caller still ANDs this with the
        /// request's own speculation-eligibility flag (e.g. MTP support) before actually drafting.
        case soloSpeculative
    }

    /// The number of ticks after a solo/batch membership change during which solo speculation stays
    /// suppressed, even with an empty companion horizon — damps thrash at the 1↔2 boundary. Zero is
    /// valid (no hysteresis window; speculation may re-enable on the very next tick).
    public let speculationWarmupTicks: Int

    /// Fail-closed: throws rather than silently clamping a negative warmup to 0, so a caller bug
    /// (e.g. an unvalidated CLI flag) surfaces immediately instead of quietly disabling hysteresis.
    public init(speculationWarmupTicks: Int) throws {
        guard speculationWarmupTicks >= 0 else {
            throw AdaptiveSoloSpeculationPolicyError.negativeWarmupTicks(speculationWarmupTicks)
        }
        self.speculationWarmupTicks = speculationWarmupTicks
    }

    /// Resolve `signals` into a `Mode`. Fail-closed: throws on any negative signal count rather than
    /// silently treating it as zero, since a negative count can only originate from a scheduler bug and
    /// masking it risks a wrong (and non-obvious) speculation decision.
    public func mode(for signals: Signals) throws -> Mode {
        guard signals.decodeCandidates >= 0 else {
            throw AdaptiveSoloSpeculationPolicyError.negativeSignal(
                "decodeCandidates", signals.decodeCandidates)
        }
        guard signals.prefillingRequests >= 0 else {
            throw AdaptiveSoloSpeculationPolicyError.negativeSignal(
                "prefillingRequests", signals.prefillingRequests)
        }
        guard signals.queuedRequests >= 0 else {
            throw AdaptiveSoloSpeculationPolicyError.negativeSignal(
                "queuedRequests", signals.queuedRequests)
        }
        guard signals.ticksSinceMembershipChange >= 0 else {
            throw AdaptiveSoloSpeculationPolicyError.negativeSignal(
                "ticksSinceMembershipChange", signals.ticksSinceMembershipChange)
        }

        if signals.decodeCandidates == 0 {
            return .idle
        }
        if signals.decodeCandidates >= 2 {
            // Unconditional — never gated by warmup or queue, so this can never become a second
            // solo-vs-batch authority alongside the scheduler's own decision.
            return .batch
        }

        // Exactly 1 candidate: solo. A companion arrival (queued or prefilling) is imminent, so
        // drafting speculative tokens now would only be discarded by the next batch drain.
        if (signals.queuedRequests + signals.prefillingRequests) > 0 {
            return .soloPlain
        }
        // No imminent companion, but still inside the post-membership-change hysteresis window.
        if signals.ticksSinceMembershipChange < speculationWarmupTicks {
            return .soloPlain
        }
        return .soloSpeculative
    }
}

/// Fail-closed errors for `AdaptiveSoloSpeculationPolicy`, mirroring `ServeTierError`'s idiom: every
/// invalid input throws with a message naming the offending field, never a silent clamp/default.
public enum AdaptiveSoloSpeculationPolicyError: Error, CustomStringConvertible, Equatable {
    /// `speculationWarmupTicks` was negative at construction.
    case negativeWarmupTicks(Int)
    /// A `Signals` field named by `field` was negative at `mode(for:)` time.
    case negativeSignal(String, Int)

    public var description: String {
        switch self {
        case .negativeWarmupTicks(let value):
            return "AdaptiveSoloSpeculationPolicy: speculationWarmupTicks must be >= 0, got \(value)."
        case .negativeSignal(let field, let value):
            return "AdaptiveSoloSpeculationPolicy: Signals.\(field) must be >= 0, got \(value)."
        }
    }
}
