import Foundation
import MLXLMCommon

/// Optional capability a `Decoder` may add so `RouteSwitchingDecoder` can reclaim the FAST route's
/// preallocated KV storage before the GENERAL route allocates its own — see
/// `RouteSwitchingDecoder`'s own doc comment for the single-live-KV-storage invariant this exists
/// to uphold. Kept as a separate protocol (mirroring `LogprobDecoding`/`SpeculativeTelemetryProviding`)
/// rather than widening `Decoder` itself, so a decoder with nothing to release (every existing
/// conformer predating this type) needs no change.
public protocol KVCacheReleasingDecoder: Decoder {
    /// Drop any preallocated KV storage (and anything that captures it, e.g. a compiled step
    /// closure) so the next `prefill` starts from the same cold state a freshly constructed
    /// decoder would. Must be safe to call at any point between requests (not mid-generation).
    mutating func releaseKVCaches()
}

/// Composes a FAST greedy-only decoder (e.g. `CompiledMLXDecoder`) with a GENERAL decoder that
/// supports sampling/penalties/response-format constraints/logprobs (e.g. `MLXDecoder`), routing
/// each REQUEST to whichever side it actually needs. Exists because the compiled fast path
/// (`CompiledMLXDecoder`) is greedy-only by construction (its `MLX.compile`d step traces one fixed
/// argmax graph), so a dense/uniform-GQA checkpoint resolving to that route previously refused
/// every sampled, penalized, logprobs, or `response_format: json_object` request outright — see
/// `docs/task-inbox/2026-09-17-DECISION-response-format-json-object-pure-swift.md`'s "1d result".
///
/// ROUTE LATCH: the route is decided ONCE per request, at that request's `prefill`/
/// `prefillWithLogprob` call — never re-evaluated mid-request — and every subsequent `step`/
/// `stepWithLogprob` call for that same request reuses the latched route. This matches
/// `InferenceActor.generateBounded`'s own call shape: `setSampling`/`setPenalties`/
/// `setResponseFormatConstraint` are always called BEFORE the first `decodeStep` of a request (see
/// that method's body), so by the time `prefill`/`prefillWithLogprob` runs, this decoder already
/// knows everything the request will ask of it for its whole lifetime — `logprobTopN` is likewise
/// fixed for the whole request, so `decodeStep` there calls either the plain pair
/// (`prefill`/`step`) or the `*WithLogprob` pair for EVERY step of one request, never a mix.
/// `prefillWithLogprob`/`stepWithLogprob` therefore always latch/read the GENERAL route: a
/// `LogprobDecoding` entry point at request start (`prefillWithLogprob`) always routes general
/// (`CompiledMLXDecoder` doesn't conform to `LogprobDecoding` at all, so there is no "fast logprob"
/// to route to), and by construction `stepWithLogprob` can only be reached on a request whose
/// `prefillWithLogprob` already latched general — reaching it with `fast` latched (or unlatched) is
/// a caller bug in `InferenceActor`, not a state this type can observe in production, and is
/// reported via `fatalError` exactly like `CompiledMLXDecoder.step`'s own "called before prefill"
/// caller-bug guard.
///
/// MEMORY INVARIANT: at most ONE route's KV storage is live at any time, so the capacity fit check
/// (which prices KV once, not once per route) stays honest. `Fast` (`CompiledMLXDecoder`) keeps its
/// preallocated buffers across an ordinary `reset()` (`resetInPlace`, by design — see that type's
/// doc comment on why: it keeps `compiledStep` valid across requests on the fast route). So a
/// request that needs `general` for the first time after `fast` actually allocated must explicitly
/// call `fast.releaseKVCaches()` BEFORE `general` prefills — `fastHasAllocatedKVSinceRelease`
/// tracks exactly that ("did fast allocate since the last release", not "does fast currently hold a
/// closure/struct"), so the release only fires on a genuine fast→general handoff, never on every
/// general request. The reverse direction needs no symmetric call: `General` (`MLXDecoder`)
/// rebuilds its cache array from `cacheFactory()` inside its own `reset()`, which this type's
/// `reset()` always calls — dropping the old (possibly populated) cache array's only reference, so
/// ARC frees its underlying storage synchronously, with no explicit "release" seam required on that
/// side.
public struct RouteSwitchingDecoder<
    Fast: KVCacheReleasingDecoder, General: Decoder & LogprobDecoding
>: Decoder, LogprobDecoding {
    private enum Route {
        case fast
        case general
    }

    private var fast: Fast
    private var general: General

    /// Recorded from `setSampling`/`setPenalties`/`setResponseFormatConstraint` for the NEXT
    /// request; read once at that request's `prefill` to decide the route. Reset to the greedy/
    /// empty/nil defaults by `reset()`, mirroring `InferenceActor.generateBounded`'s own
    /// `defer`-time `setSampling(.greedy)`/`setPenalties(.none)`/`setResponseFormatConstraint(nil)`
    /// calls (which run AFTER `reset()` there and so leave this decoder's own copies consistent
    /// either way).
    private var pendingSampling: DecoderSampling = .greedy
    private var pendingPenalties: DecoderPenalties = .none
    private var pendingConstraint: (any LogitProcessor)?

    /// The route latched by this request's `prefill`/`prefillWithLogprob` call, consulted by every
    /// later `step`/`stepWithLogprob` in the same request. `nil` before the first prefill of a
    /// request (or after `reset()`) — `step`/`stepWithLogprob` reaching this state is a caller bug.
    private var activeRoute: Route?

    /// Tracks whether `fast` currently holds real preallocated KV storage that has not yet been
    /// released — see this type's own doc comment's MEMORY INVARIANT section. Set `true` right
    /// after `fast.prefill` actually runs, cleared by `releaseFastKVCachesIfNeeded()`.
    private var fastHasAllocatedKVSinceRelease = false

    public init(fast: Fast, general: General) {
        self.fast = fast
        self.general = general
    }

    /// This request needs the general route whenever ANY configured capability the fast route
    /// cannot honor was requested: non-greedy sampling, a non-empty penalty set, or a response-
    /// format constraint. Pure function of the recorded pending state, so `prefill` and
    /// `prefillWithLogprob` (which always needs general regardless of this value — see the type's
    /// doc comment) can share it without duplicating the condition.
    private var pendingRequestNeedsGeneral: Bool {
        pendingSampling != .greedy || !pendingPenalties.isEmpty || pendingConstraint != nil
    }

    // MARK: - Capability surface: mirrors `general`'s, since `fast` is by construction a strict
    // subset (greedy, no penalties, no constraint, no logprobs) of what `general` supports, and a
    // request needing any of those capabilities always routes to `general` anyway.

    public var supportsSampling: Bool { general.supportsSampling }
    public var supportsPenalties: Bool { general.supportsPenalties }
    public var supportsResponseFormatConstraint: Bool { general.supportsResponseFormatConstraint }

    public mutating func setSampling(_ sampling: DecoderSampling) {
        pendingSampling = sampling
        general.setSampling(sampling)
    }

    public mutating func setPenalties(_ penalties: DecoderPenalties) {
        pendingPenalties = penalties
        general.setPenalties(penalties)
    }

    public mutating func setResponseFormatConstraint(_ constraint: (any LogitProcessor)?) {
        pendingConstraint = constraint
        general.setResponseFormatConstraint(constraint)
    }

    // MARK: - Decode

    public mutating func prefill(_ promptTokens: [Int]) throws -> Int {
        if pendingRequestNeedsGeneral {
            activeRoute = .general
            releaseFastKVCachesIfNeeded()
            return try general.prefill(promptTokens)
        }
        activeRoute = .fast
        // Set before the call: a prefill that throws after allocating still leaves storage that
        // the next general request must release. Releasing an unallocated decoder is harmless.
        fastHasAllocatedKVSinceRelease = true
        return try fast.prefill(promptTokens)
    }

    public mutating func step(last: Int) throws -> Int {
        switch activeRoute {
        case .general:
            return try general.step(last: last)
        case .fast:
            return try fast.step(last: last)
        case nil:
            fatalError("RouteSwitchingDecoder.step called before prefill")
        }
    }

    public mutating func reset() {
        fast.reset()
        general.reset()
        activeRoute = nil
        pendingSampling = .greedy
        pendingPenalties = .none
        pendingConstraint = nil
    }

    // MARK: - LogprobDecoding: always the general route — see the type's doc comment.

    public mutating func prefillWithLogprob(
        _ promptTokens: [Int], topN: Int
    ) throws -> (token: Int, logprob: DecodedTokenLogprob) {
        activeRoute = .general
        releaseFastKVCachesIfNeeded()
        return try general.prefillWithLogprob(promptTokens, topN: topN)
    }

    public mutating func stepWithLogprob(
        last: Int, topN: Int
    ) throws -> (token: Int, logprob: DecodedTokenLogprob) {
        guard case .general = activeRoute else {
            // By construction unreachable: `logprobTopN != nil` (the only reason
            // `InferenceActor.generateBounded` ever calls this method) always drives that same
            // request's `prefill` through `prefillWithLogprob` first, which always latches
            // `.general` — see the type's doc comment's ROUTE LATCH section.
            fatalError(
                "RouteSwitchingDecoder.stepWithLogprob reached without a general-latched prefillWithLogprob")
        }
        return try general.stepWithLogprob(last: last, topN: topN)
    }

    // MARK: - Memory invariant

    private mutating func releaseFastKVCachesIfNeeded() {
        guard fastHasAllocatedKVSinceRelease else { return }
        fast.releaseKVCaches()
        fastHasAllocatedKVSinceRelease = false
    }
}
