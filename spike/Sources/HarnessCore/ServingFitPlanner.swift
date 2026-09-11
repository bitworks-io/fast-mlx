import Foundation

/// The complete fit-check outcome for one (model, host, requested-context) triple — the single
/// pure decision the serving path consults before loading (fit-checked-serve, differentiator #2).
/// Every number the loader needs (proceed/refuse, MLX limits derived from the sizer, the served
/// context) is here, so the whole policy is unit-tested off-box and the live serve only confirms
/// the wired numbers match (the metallib-gated live path proves nothing about this logic).
public struct ServingFitDecision: Sendable {
    public let modelID: String
    public let color: CapacityColor
    public let bindingConstraint: BindingConstraint
    public let mitigation: String
    /// Quantization bit-width label for the summary header (candidates-mode parity); `nil` when
    /// unknown (single-model path, unless the decoded config carries it).
    public let quantBits: Int?

    /// Context the operator asked for (explicit `--context`, or the effective default when unset).
    public let requestedContext: Int
    /// Context the server will actually run at: `== requestedContext` unless it was capped to the
    /// computed ceiling.
    public let servedContext: Int
    /// Largest context whose predicted peak stays out of the red at this host/quant/concurrency
    /// (`0` ⟹ the model is unservable at any context here — weights don't fit, or KV not derivable).
    public let contextCeiling: Int
    /// `true` when `servedContext < requestedContext` (the request was reduced to the ceiling).
    public let contextWasCapped: Bool
    /// `true` when the operator passed an explicit `--context` (vs. falling back to the default).
    public let explicitContextRequested: Bool
    /// Concurrent decode streams the verdict was COMPUTED for (KV scales ×this). `1` is the shipped
    /// default and renders byte-identically; `>1` is an opt-in stricter verdict (`--plan-concurrency`)
    /// and is labeled in both the summary and the machine line so the tighter basis is attributable.
    public let planningConcurrency: Int

    // MARK: MLX limits — derived from the sizer, NOT flat RAM percentages.
    /// The host policy's effective memory ceiling. Shared hosts use the lowest of exact 75% RAM,
    /// a lower wired observation, and a lower Metal recommended working set.
    public let memoryLimitBytes: Int
    /// `CapacityModel.recommendedCacheLimitBytes(wiredLimit)` — the explicit anti-hoard cache cap,
    /// not a flat `RAM×10%`.
    public let cacheLimitBytes: Int
    /// The sizer's predicted KV footprint for `servedContext × concurrency` — what the
    /// continuous-batch backend should reserve, not a flat `RAM×30%`.
    public let maxReservedKVBytes: Int
    /// The number of concurrent decode slots the advisory line models, or `nil` when no advisory
    /// was requested (single-model default / scalar route). Advisory ONLY — never feeds the verdict.
    public let advisorySlotCount: Int?
    /// The modeled KV footprint at `servedContext × advisorySlotCount`, or `nil` when no advisory
    /// was requested. Surfaces the concurrent-decode headroom the single-slot verdict omits
    /// (…-fit-check-concurrency-kv-undercount.md, option 1); it is deliberately NOT applied as a cap.
    public let kvAtSlotsBytes: Int?

    /// `false` when the verdict is red and `--force` was NOT set: the loader must refuse.
    public let shouldProceed: Bool
    /// `true` when proceeding past a red verdict only because `--force` was set.
    public let proceedingUnderForce: Bool

    // MARK: provenance (spec §5 — never present an estimate as a measured fact)
    public let weightsAreMeasured: Bool
    /// `true` when the weights byte count came from the checkpoint's declared
    /// `model.safetensors.index.json` total (honest but not measured on disk). Distinguishes a
    /// *declared* size from a param-count *estimate* in the summary; the machine-readable line still
    /// reports `weights_measured=false` for both (declared is, correctly, not measured).
    public let weightsAreDeclared: Bool
    public let wiredLimitIsMeasured: Bool
    /// Provenance for the envelope that actually governed this decision. This stays distinct from
    /// the wired-input flag so a synthesized shared cap or advisory Metal bound cannot be folded
    /// into `fit_estimate_measured=true` merely because the unused wired observation was measured.
    public let effectiveMemoryCeilingSource: EffectiveMemoryCeiling.Source
    public let effectiveMemoryCeilingIsMeasured: Bool
    /// The full term-by-term peak prediction the verdict was computed from (weights/KV/transient/
    /// allocator), for surfacing measured-vs-modeled totals.
    public let prediction: CapacityPrediction

    /// Compares `prediction.allocatorHeadroomBytes` (the byte term the SIZER prices the allocator
    /// at — currently `CapacityModel.predictPeakBytes`'s hardcoded default, never overridden by any
    /// call site) against `cacheLimitBytes` (`CapacityModel.recommendedCacheLimitBytes`, the byte
    /// cap the RUNTIME is actually entitled to let `MLX.Memory.cacheLimit` hold). These are two
    /// independently-computed numbers from the SAME `decide()` call; nothing enforces
    /// `allocatorHeadroomBytes >= cacheLimitBytes`, so the model can price the allocator below what
    /// the runtime may legitimately cache — and cache bytes are exactly what MLX's own `peakMemory`
    /// excludes (`FitCheckMeasuredReport`'s `measuredFootprintBytes` is the other half of this
    /// increment, for the same reason).
    ///
    /// ADVISORY ONLY. This state is never read by `classify`/`shouldProceed`/`color` and must never
    /// be allowed to change them — substituting the real cache entitlement into the gating
    /// prediction is a separate, operator-coordinated increment (arming), not this one.
    public let allocatorHeadroomState: ServingFitPlanner.AllocatorHeadroomState

    private static func gib(_ b: Int) -> String { String(format: "%.2f GiB", Double(b) / 1_073_741_824.0) }
    private static func gib(_ b: Double) -> String { String(format: "%.2f GiB", b / 1_073_741_824.0) }

    /// Operator-facing summary, one string per line — used in the startup announce and the refusal
    /// message so both read identically.
    public func summaryLines() -> [String] {
        var lines: [String] = []
        let mark = color.rawValue.uppercased()
        let bitsLabel = quantBits.map { " [\($0)-bit]" } ?? ""
        lines.append("fit-check [\(mark)] model=\(modelID)\(bitsLabel) binding=\(bindingConstraint.rawValue)")
        let weightsProv = weightsAreMeasured ? "measured" : (weightsAreDeclared ? "declared" : "estimated")
        let wiredProv = wiredLimitIsMeasured ? "measured" : "estimated"
        lines.append("  weights=\(Self.gib(prediction.weightsBytes)) (\(weightsProv)) "
            + "kv@\(servedContext)=\(Self.gib(prediction.kvBytes)) "
            + "peak=\(Self.gib(prediction.totalBytes)) wired-limit=\(wiredProv)")
        if contextWasCapped {
            lines.append("  context capped: requested \(requestedContext) → serving \(servedContext) (ceiling \(contextCeiling))")
        } else {
            lines.append("  context=\(servedContext) (ceiling \(contextCeiling))")
        }
        // When the operator opted into a concurrency-aware verdict (--plan-concurrency N), name the
        // basis so the tighter ceiling/verdict is never misread as the single-slot default. Absent at
        // the default concurrency=1, keeping the shipped summary byte-identical.
        if planningConcurrency > 1 {
            lines.append("  planning concurrency: verdict computed for \(planningConcurrency) concurrent"
                + " decode streams (stricter than the single-slot default)")
        }
        // memory + cache are the sizer limits the serving path applies; the KV figure is the
        // estimate for `servedContext` (shown above), advisory only — the reserved-KV byte cap is
        // left at the provided value so multi-slot admission is not tightened to a concurrency-1
        // boundary.
        lines.append("  sizer limits: memory=\(Self.gib(memoryLimitBytes)) cache=\(Self.gib(cacheLimitBytes))"
            + " (reserved-KV estimate \(Self.gib(maxReservedKVBytes)), advisory)")
        // Concurrent-decode headroom the single-slot verdict omits: the backend admits multiple
        // slots, so peak KV can be up to N× the single-slot estimate. Shown as a modeled advisory
        // — NOT applied as a cap (that would reintroduce the tight-cap regression `faf1f35` removed).
        if let slots = advisorySlotCount, let kvAtSlots = kvAtSlotsBytes {
            lines.append("  concurrency advisory: kv@\(servedContext) x\(slots) slots = \(Self.gib(kvAtSlots)) (modeled, advisory — not a cap)")
        }
        if !shouldProceed {
            lines.append("  REFUSED (red): \(mitigation). Re-run with --force to override.")
        } else if proceedingUnderForce {
            lines.append("  PROCEEDING under --force despite red: \(mitigation)")
        } else if color != .green {
            lines.append("  note: \(mitigation)")
        }
        return lines
    }

    /// Machine-readable fit-check fields for the STDOUT startup line (spliced immediately before
    /// `listening=<addr>`, which must remain the last token — gate scripts anchor on it).
    public func machineReadableFields() -> String {
        // The folded value follows the envelope that actually governed this decision. A shared
        // policy cap is synthesized, and Metal documents its recommendation as an approximation;
        // neither is promoted to a fully measured fit even if the separate wired input was read.
        let estimateMeasured = weightsAreMeasured && effectiveMemoryCeilingIsMeasured
        var fields = "fit_check=\(color.rawValue) fit_binding=\(bindingConstraint.rawValue) "
            + "weights_measured=\(weightsAreMeasured) wired_limit_measured=\(wiredLimitIsMeasured) "
            + "fit_estimate_measured=\(estimateMeasured) fit_quant_bits=\(quantBits.map(String.init) ?? "none") "
            + "fit_served_context=\(servedContext) fit_context_ceiling=\(contextCeiling) "
            + "fit_context_capped=\(contextWasCapped)"
        // Additive, opt-in field: appended AFTER the frozen base contract and BEFORE `fit_forced` so
        // both the fixed prefix and the `fit_forced`-last invariant gate scripts rely on stay intact.
        // Absent at the default concurrency=1 (frozen-contract locked).
        if planningConcurrency > 1 {
            fields += " fit_plan_concurrency=\(planningConcurrency)"
        }
        if proceedingUnderForce {
            fields += " fit_forced=true"
        }
        return fields
    }
}

/// Pure planner assembling a `ServingFitDecision` from the sizer primitives (`CapacityModel` +
/// `SystemProfile`). No filesystem, no MLX — the caller supplies a decoded `ModelArchProfile`
/// (typically from `ModelConfigDecoder`) and a `SystemProfile` (typically `detectHost()`).
public enum ServingFitPlanner {

    /// Three-state result of comparing the sizer's modeled allocator-headroom term against the
    /// runtime's actual cache-byte entitlement (`ServingFitDecision.allocatorHeadroomState`'s doc
    /// comment explains what each side means and why the gap matters). Mirrors
    /// `WiredCeilingAssessment`'s shape deliberately: distinguishing "checked and consistent" from
    /// "never checked" matters here for the same reason it matters in
    /// `WiredCeilingOvercommitGuard` — this project has been bitten by advisories that render
    /// identically whether they ran and found nothing, or never ran at all.
    public enum AllocatorHeadroomState: Equatable, Sendable {
        /// `allocatorHeadroomBytes >= cacheLimitBytes` — the modeled allocator term already prices
        /// at or above what the runtime may cache. No gap to report.
        case consistent(headroomBytes: Int, cacheLimitBytes: Int)
        /// `allocatorHeadroomBytes < cacheLimitBytes` — the model prices the allocator below the
        /// runtime's own cache entitlement by `gapBytes`.
        case underEntitled(headroomBytes: Int, cacheLimitBytes: Int, gapBytes: Int)
    }

    private static func gib(_ b: Int) -> String { String(format: "%.2f GiB", Double(b) / 1_073_741_824.0) }

    /// Operator-facing advisory lines for the allocator-headroom-vs-cache-entitlement gap — same
    /// `"NOTE:"`-prefixed, GiB-plus-raw-bytes style as `WiredCeilingOvercommitGuard.advisoryLines`
    /// / `ModelSizer.provenanceNotes`. Emits exactly ONE stable machine-readable
    /// `allocator_headroom_state=<value>` token per call (`under_entitled` / `consistent`) so a log
    /// scraper need not parse prose, and emits a line in BOTH states — the `consistent` line proves
    /// in the logs that this check actually ran, rather than leaving "checked and fine"
    /// indistinguishable from "never checked". ADVISORY ONLY: never gates, never proceeds/refuses.
    public static func allocatorHeadroomAdvisoryLines(for state: AllocatorHeadroomState) -> [String] {
        switch state {
        case .consistent(let headroom, let cacheLimit):
            return [
                "NOTE: allocator_headroom_state=consistent — modeled allocator headroom "
                    + "\(headroom) B (\(gib(headroom))) is at or above the runtime's cache "
                    + "entitlement \(cacheLimit) B (\(gib(cacheLimit))); advisory only.",
            ]
        case .underEntitled(let headroom, let cacheLimit, let gap):
            return [
                "NOTE: allocator_headroom_state=under_entitled — the sizer prices the allocator "
                    + "below what the runtime is entitled to cache.",
                "  modeled_allocator_headroom=\(headroom) B (\(gib(headroom)))",
                "  runtime_cache_limit=\(cacheLimit) B (\(gib(cacheLimit)))",
                "  gap=\(gap) B (\(gib(gap))) of cache the process may legitimately hold is priced "
                    + "below the allocator headroom term AND excluded from MLX's own `peakMemory` "
                    + "(see FitCheckMeasuredReport.measuredFootprintBytes); advisory only — this does "
                    + "not change the verdict, and no verdict field consumes this state.",
            ]
        }
    }

    /// Decide whether/how to serve `profile` on `host` at `requestedContext`.
    ///
    /// Context policy (reconciles "cap+announce at ceiling" with "fail closed on red" AND "existing
    /// serving unaffected"):
    /// - `requestedContext == nil` (default): serve up to the largest context that fits —
    ///   `min(nativeMax, ceiling)`. On a box that can hold the full native context this equals the
    ///   native max, i.e. the SAME cap the backend applies when `maxContextTokens` is unset, so no
    ///   previously-serving request is newly rejected; on a box that can't, it caps to the ceiling
    ///   (protective — the backend would otherwise accept the request and then OOM). It is
    ///   deliberately NOT the 32K "effective default" (spec §4), which is a per-request default, not
    ///   the server's hard cap: enforcing 32K here would hard-reject longer requests the model+box
    ///   can actually serve.
    /// - explicit `--context`: honor the ask — if it lands red, refuse (unless `--force`); when
    ///   forced, serve capped at the ceiling.
    public static func decide(
        profile: ModelArchProfile, weightsAreMeasured: Bool, host: SystemProfile,
        requestedContext: Int? = nil, kvQuant: KVQuantTier = .fp16, concurrency: Int = 1,
        force: Bool = false, thresholds: CapacityThresholds = .default, quantBits: Int? = nil,
        weightsAreDeclared: Bool = false, advisorySlotCount: Int? = nil,
        allowContextCapping: Bool = true
    ) -> ServingFitDecision {
        // Default request = the model's native max; servedContext then caps it to what fits below.
        let resolvedRequest = requestedContext ?? profile.nativeMaxContext
        let explicit = requestedContext != nil
        let ceiling = CapacityModel.contextCeiling(
            model: profile, profile: host, kvQuant: kvQuant, concurrency: concurrency, thresholds: thresholds)

        let exceedsNativeMax = resolvedRequest > profile.nativeMaxContext

        // A serve-tier dial (transparent/balanced) can forbid silently shortening context: rather than
        // capping a memory-bound request to the ceiling, refuse the full ask (default `true` = the
        // shipped cap-and-proceed behavior, byte-identical). Native-max-exceeded and nothing-fits are
        // model/hardware limits capping never fixes, so they are unaffected by this flag.
        let memoryBoundCap = !exceedsNativeMax && ceiling > 0 && resolvedRequest > ceiling
        let refuseInsteadOfCap = memoryBoundCap && !allowContextCapping

        // Pick the served context and whether it was capped.
        let servedContext: Int
        let capped: Bool
        if exceedsNativeMax {
            servedContext = min(profile.nativeMaxContext, ceiling > 0 ? ceiling : profile.nativeMaxContext)
            capped = true
        } else if ceiling == 0 {
            servedContext = resolvedRequest
            capped = false
        } else if resolvedRequest <= ceiling {
            servedContext = resolvedRequest
            capped = false
        } else if refuseInsteadOfCap {
            // The tier forbids a silent cap — surface the full ask (assessed red below → refuse). The
            // ceiling that WOULD have fit is still reported so the operator can re-run with --tier
            // maxfit or an explicit smaller --context.
            servedContext = resolvedRequest
            capped = false
        } else {
            // Memory-bound: the model fits at a smaller context than requested.
            servedContext = ceiling
            capped = true
        }

        // Assess the verdict at the context that governs proceed/refuse:
        //  - explicit ask over the ceiling, native-max exceeded, nothing-fits, or a cap the tier
        //    forbids → assess the ASK (so the red is honest and we fail closed);
        //  - default auto-cap or a request that fits → assess what we SERVE (non-red → proceed).
        let assessContext: Int
        if exceedsNativeMax || ceiling == 0 || (resolvedRequest > ceiling && (explicit || refuseInsteadOfCap)) {
            assessContext = resolvedRequest
        } else {
            assessContext = servedContext
        }

        let prediction = CapacityModel.predictPeakBytes(
            model: profile, context: assessContext, concurrency: concurrency, kvQuant: kvQuant, profile: host)
        let verdict = CapacityModel.classify(
            prediction, profile: host, weightsBytes: Double(profile.weightsBytes4bitEstimate), thresholds: thresholds)

        let isRed = verdict.color == .red
        let shouldProceed = !isRed || force
        let proceedingUnderForce = isRed && force

        // Limits from the sizer, not flat RAM percentages.
        let effectiveMemoryCeiling = host.effectiveMemoryCeiling
        let memoryLimitBytes = effectiveMemoryCeiling.bytes
        let cacheLimitBytes = CapacityModel.recommendedCacheLimitBytes(wiredLimitBytes: memoryLimitBytes)
        let kvServe = CapacityModel.kvBytesForContext(
            profile, context: max(1, servedContext), kvQuant: kvQuant, concurrency: concurrency)
        let maxReservedKVBytes = Int(kvServe.rounded(.up))

        // Advisory only (option 1): model the KV footprint at N concurrent slots so the operator
        // sees the concurrent-decode headroom, WITHOUT feeding it into the verdict/limits above.
        let kvAtSlotsBytes: Int? = advisorySlotCount.map { slots in
            Int(CapacityModel.kvBytesForContext(
                profile, context: max(1, servedContext), kvQuant: kvQuant, concurrency: slots).rounded(.up))
        }

        // Advisory only: name the gap between the modeled allocator-headroom term and the runtime's
        // actual cache entitlement (see `ServingFitDecision.allocatorHeadroomState`'s doc comment).
        // Deliberately computed from `prediction`/`cacheLimitBytes` alone, so it can never influence
        // `verdict`/`shouldProceed` above (both are already assigned by this point).
        let allocatorHeadroomBytes = Int(prediction.allocatorHeadroomBytes.rounded())
        let allocatorHeadroomState: AllocatorHeadroomState
        if allocatorHeadroomBytes < cacheLimitBytes {
            allocatorHeadroomState = .underEntitled(
                headroomBytes: allocatorHeadroomBytes, cacheLimitBytes: cacheLimitBytes,
                gapBytes: cacheLimitBytes - allocatorHeadroomBytes)
        } else {
            allocatorHeadroomState = .consistent(
                headroomBytes: allocatorHeadroomBytes, cacheLimitBytes: cacheLimitBytes)
        }

        return ServingFitDecision(
            modelID: profile.id, color: verdict.color, bindingConstraint: verdict.bindingConstraint,
            mitigation: verdict.suggestedMitigation, quantBits: quantBits, requestedContext: resolvedRequest,
            servedContext: servedContext, contextCeiling: ceiling, contextWasCapped: capped,
            explicitContextRequested: explicit, planningConcurrency: concurrency,
            memoryLimitBytes: memoryLimitBytes,
            cacheLimitBytes: cacheLimitBytes, maxReservedKVBytes: maxReservedKVBytes,
            advisorySlotCount: advisorySlotCount, kvAtSlotsBytes: kvAtSlotsBytes,
            shouldProceed: shouldProceed, proceedingUnderForce: proceedingUnderForce,
            weightsAreMeasured: weightsAreMeasured, weightsAreDeclared: weightsAreDeclared,
            wiredLimitIsMeasured: host.wiredLimitIsMeasured,
            effectiveMemoryCeilingSource: effectiveMemoryCeiling.source,
            effectiveMemoryCeilingIsMeasured: host.effectiveMemoryCeilingIsMeasured,
            prediction: prediction, allocatorHeadroomState: allocatorHeadroomState)
    }
}
