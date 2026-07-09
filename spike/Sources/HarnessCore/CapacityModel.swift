import Foundation

/// KV-cache storage precision tier. `bytesPerElement` is the multiplier the KV formulas use in
/// place of the naive "2 bytes for fp16" constant.
public enum KVQuantTier: String, Sendable {
    case fp16
    case int8
    /// "turbo4" ≈ 4-bit-equivalent packing. The raw 4-bit payload is 0.25 bytes/element, but a
    /// real implementation carries per-block scale/zero-point metadata; 0.5 bytes/element here is
    /// a documented ~2x tax over the raw payload to account for that, not a measured figure —
    /// mlx-swift's own K/V quantization is a reference point (32-wide int8 groups, 33rd byte of
    /// scale/bias per group ≈ 3% tax at int8 group granularity), but no turbo4 build exists yet in
    /// this codebase to measure directly (spec §6, mitigation #1: "shaped engineering", not built).
    case turbo4
    /// TurboQuant 2-bit tier. ⚠️ EXPERIMENTAL / UNMEASURED — spec §6 mitigation #1 calls TurboQuant
    /// 2-bit "deferred"; 0.3125 (2.5/8) is a placeholder bytes/element for a 2.5-bit-equivalent
    /// scheme, not a validated number.
    case tq2_5
    /// ⚠️ EXPERIMENTAL / UNMEASURED, same caveat as `.tq2_5`. 0.4375 = 3.5/8.
    case tq3_5

    public var bytesPerElement: Double {
        switch self {
        case .fp16: return 2.0
        case .int8: return 1.0
        case .turbo4: return 0.5
        case .tq2_5: return 0.3125
        case .tq3_5: return 0.4375
        }
    }
}

/// Which of the model's or the box's constraints is actually binding a red/yellow verdict — the
/// differentiator the advisor exists to name (spec §5).
public enum BindingConstraint: String, Sendable {
    /// Predicted peak fits comfortably; nothing is binding.
    case fits
    /// Total physical RAM is the smaller of {wired limit, RAM} and is exceeded.
    case physicalRAM
    /// The GPU wired-memory ceiling (`iogpu.wired_limit_mb`) is the smaller of the two and is
    /// exceeded — raisable, but requires root (spec §3).
    case wiredLimit
    /// The requested context exceeds the model's own native maximum — no amount of hardware helps.
    case modelNativeMax
    /// The model's native max sits below the 32K tunable default (Phi-4 = 16K) — the effective
    /// default must be the native max, not 32K (spec §4/§8).
    case nativeMaxBelowDefault
    /// The architecture's own KV formula is the wall (DeepSeek-R1 as-implemented: 152.5 GiB @32K)
    /// — no hardware on the roadmap holds it until absorbed-MLA (spec §7 backlog) ships.
    case mlaAsImplemented
    /// The model's KV cost is NOT derivable from confirmed config (unconfirmed attention-layer
    /// count, or an out-of-scope arch) — we cannot honestly say whether it fits. Reported instead
    /// of a misleading under-count (spec §2.1 "do not multiply blind"; §8).
    case kvNotDerivable
}

public enum CapacityColor: String, Sendable { case green, yellow, red }

/// Configurable green/yellow/red thresholds (spec §5: "thresholds are config, not magic numbers").
public struct CapacityThresholds: Sendable {
    public let greenMax: Double
    public let yellowMax: Double
    public let osReserveBytes: Int
    public init(greenMax: Double = 0.70, yellowMax: Double = 0.90, osReserveBytes: Int = 4 * 1024 * 1024 * 1024) {
        self.greenMax = greenMax; self.yellowMax = yellowMax; self.osReserveBytes = osReserveBytes
    }
    public static let `default` = CapacityThresholds()
}

/// Term-by-term breakdown of a peak-footprint prediction (spec §2 formula), so tests and the
/// advisor can inspect each contributor rather than trusting an opaque total.
public struct CapacityPrediction: Sendable {
    public let modelID: String
    public let modelType: ArchClass
    public let nativeMaxContext: Int
    public let context: Int
    public let concurrency: Int
    public let weightsBytes: Double
    /// `concurrency × KV(context)` — the full per-sequence KV (growth + any fixed state), summed
    /// across concurrent streams.
    public let kvBytes: Double
    public let transientPrefillPeakBytes: Double
    public let allocatorHeadroomBytes: Double
    /// `false` when `kvBytes` is not trustworthy (the model's attention-layer count is unconfirmed
    /// or the arch is out of scope — `ModelArchProfile.isKVDerivable`). `classify` checks this
    /// first and refuses to return a fit color built on an under-count.
    public let derivable: Bool
    public var totalBytes: Double { weightsBytes + kvBytes + transientPrefillPeakBytes + allocatorHeadroomBytes }
}

public struct CapacityVerdict: Sendable {
    public let color: CapacityColor
    public let bindingConstraint: BindingConstraint
    /// `(prediction.totalBytes - weightsBytes) / headroomBytes` — ratio of the non-weights peak to
    /// available headroom above weights (spec §5's classification is stated against this, since
    /// `hardwareHolds` already subtracts weights). `.infinity` when headroom is <= 0 (weights alone
    /// don't fit).
    public let ratio: Double
    public let suggestedMitigation: String
}

/// The spine: per-architecture KV memory model as pure logic (spec §2). Dispatches the
/// KV-bytes/token formula by `ArchClass`, exactly as the vendored Swift arch layer dispatches
/// cache TYPE per layer (`MambaCache` vs `KVCacheSimple` vs `RotatingKVCache`).
public enum CapacityModel {

    /// The per-token KV growth RATE for the given model + quant tier, in bytes.
    ///
    /// This is NOT the whole per-sequence KV cost for every class — see `kvBytesForContext` for
    /// the class-specific assembly (fixed state, SWA local cap, etc.). Per spec §2.1:
    /// - uniformGQA: every layer grows every token.
    /// - hybridLinear: only the attention-layer subset grows; the rest is fixed recurrent state
    ///   (added separately via `fixedStateBytes`, not here).
    /// - interleavedSWA: returns the GLOBAL-layer growth rate only; the local layers are
    ///   context-independent (capped at the sliding window) and are folded in by
    ///   `kvBytesForContext`, not this function.
    /// - mlaAsImplemented: the decompressed per-head rate (rope + nope + v dims already sum both
    ///   K and V contributions, so no separate ×2 factor here — unlike the GQA formula's explicit
    ///   K-and-V doubling).
    /// - hybridMamba2MoE: `nAttnLayers` is ⚠️-unconfirmed for every catalog member (sentinel 0) —
    ///   this returns the PER-ATTENTION-LAYER unit only, never multiplied by an unconfirmed layer
    ///   count. Callers must not silently multiply this by `nLayers` and report a total.
    /// - novelCompressedUnsupported: 0 — out of scope, not derived (spec §8); never fabricated.
    public static func kvBytesPerToken(_ m: ModelArchProfile, kvQuant: KVQuantTier) -> Double {
        let bpe = kvQuant.bytesPerElement
        switch m.modelType {
        case .uniformGQA:
            return Double(m.nLayers) * Double(m.nKVHeads) * Double(m.headDim) * 2 * bpe
        case .hybridLinear:
            return Double(m.nAttnLayers) * Double(m.nKVHeads) * Double(m.headDim) * 2 * bpe
        case .interleavedSWA:
            // Global (full-context-growing) layers only; local layers are handled as a fixed cap.
            return Double(m.nAttnLayers) * Double(m.nKVHeads) * Double(m.headDim) * 2 * bpe
        case .mlaAsImplemented:
            let rope = Double(m.mlaRopeDim ?? 0), nope = Double(m.mlaNopeDim ?? 0), v = Double(m.mlaVDim ?? 0)
            let heads = Double(m.mlaHeads ?? 0)
            return Double(m.nAttnLayers) * heads * (rope + nope + v) * bpe
        case .hybridMamba2MoE:
            // Per-attention-layer unit ONLY — nAttnLayers is an unconfirmed sentinel (0) for every
            // catalog member; do not multiply by it and present a fabricated total.
            return Double(m.nKVHeads) * Double(m.headDim) * 2 * bpe
        case .novelCompressedUnsupported:
            return 0
        }
    }

    /// The fixed (context-independent) per-sequence bytes for interleaved-SWA's local/rotating
    /// layers, capped at the sliding window rather than growing with context.
    public static func swaFixedLocalBytes(_ m: ModelArchProfile, kvQuant: KVQuantTier) -> Double {
        guard m.modelType == .interleavedSWA, let window = m.slidingWindow else { return 0 }
        let bpe = kvQuant.bytesPerElement
        return Double(m.nLocalLayers) * Double(m.nKVHeads) * Double(m.headDim) * 2 * bpe * Double(window)
    }

    /// The full per-SEQUENCE KV at a given context (growth + any fixed state/local-cap term),
    /// multiplied by `concurrency` since KV is per-sequence and N streams multiply it (spec §2).
    public static func kvBytesForContext(
        _ m: ModelArchProfile, context: Int, kvQuant: KVQuantTier, concurrency: Int
    ) -> Double {
        let perSequence: Double
        switch m.modelType {
        case .interleavedSWA:
            perSequence = kvBytesPerToken(m, kvQuant: kvQuant) * Double(context) + swaFixedLocalBytes(m, kvQuant: kvQuant)
        default:
            perSequence = kvBytesPerToken(m, kvQuant: kvQuant) * Double(context) + Double(m.fixedStateBytes)
        }
        return perSequence * Double(concurrency)
    }

    /// `transient_prefill_peak` (spec §2/§6): the LEAST-characterized term — dropping it is
    /// exactly the mistake that killed processes at the 7K wall. Modeled here as a documented
    /// UPPER-BOUND coefficient times a prefill chunk's attention-activation footprint (Q/K/V +
    /// attention scores + MLP intermediate, order-of-magnitude via the same kv_heads×head_dim
    /// unit the KV formulas use). This is a defensible placeholder, not a derived constant — the
    /// spec (§7 backlog) gates a real measurement on extending `CtxProbe` to one-shot 64K/128K/
    /// 262K prefills and observing actual peak.
    /// `kvQuant` is accepted for signature symmetry with the other KV functions but intentionally
    /// UNUSED for scaling: prefill activations (Q/K/V projections, attention scores, MLP
    /// intermediate) stay at compute precision (bf16/fp16) regardless of which quant tier the KV
    /// *cache* is stored in — quantizing the cache doesn't quantize the transient activations that
    /// produce it.
    public static func transientPrefillPeakBytes(
        _ m: ModelArchProfile, chunkTokens: Int, kvQuant: KVQuantTier, coefficient: Double
    ) -> Double {
        guard m.modelType != .novelCompressedUnsupported else { return 0 }
        let perTokenUnit = max(kvBytesPerToken(m, kvQuant: .fp16), Double(m.nKVHeads) * Double(m.headDim) * 2 * 2)
        return coefficient * perTokenUnit * Double(chunkTokens)
    }

    /// `weights + N_concurrent × KV(context) + transient_prefill_peak + allocator_headroom` (spec
    /// §2's peak formula), broken into inspectable terms.
    public static func predictPeakBytes(
        model: ModelArchProfile, context: Int, concurrency: Int, kvQuant: KVQuantTier, profile: SystemProfile,
        chunkTokens: Int = 2048, transientPrefillCoefficient: Double = 8.0,
        allocatorHeadroomBytes: Double = 2 * 1024 * 1024 * 1024
    ) -> CapacityPrediction {
        let weights = Double(model.weightsBytes4bitEstimate)
        let kv = kvBytesForContext(model, context: context, kvQuant: kvQuant, concurrency: concurrency)
        let transient = transientPrefillPeakBytes(
            model, chunkTokens: chunkTokens, kvQuant: kvQuant, coefficient: transientPrefillCoefficient)
        return CapacityPrediction(
            modelID: model.id, modelType: model.modelType, nativeMaxContext: model.nativeMaxContext,
            context: context, concurrency: concurrency, weightsBytes: weights, kvBytes: kv,
            transientPrefillPeakBytes: transient, allocatorHeadroomBytes: allocatorHeadroomBytes,
            derivable: model.isKVDerivable
        )
    }

    /// Classify a prediction green/yellow/red against the box's headroom and name the binding
    /// constraint (spec §5). `weightsBytes` is passed explicitly (rather than only reading
    /// `prediction.weightsBytes`) to match the model's own signature contract and because a
    /// caller may want to classify a hypothetical weights figure against an already-computed
    /// prediction.
    public static func classify(
        _ prediction: CapacityPrediction, profile: SystemProfile, weightsBytes: Double,
        thresholds: CapacityThresholds = .default
    ) -> CapacityVerdict {
        // Model-capability checks first — independent of the box:
        // 1. A context beyond the model's native max cannot be served at all, regardless of memory.
        if prediction.context > prediction.nativeMaxContext {
            return CapacityVerdict(
                color: .red, bindingConstraint: .modelNativeMax, ratio: .infinity,
                suggestedMitigation: mitigation(for: .modelNativeMax))
        }
        // 2. If the KV cost isn't derivable (unconfirmed arch), refuse to build a fit color on an
        //    under-count — say so honestly instead (spec §2.1/§8).
        guard prediction.derivable else {
            return CapacityVerdict(
                color: .red, bindingConstraint: .kvNotDerivable, ratio: .nan,
                suggestedMitigation: mitigation(for: .kvNotDerivable))
        }

        let headroom = profile.hardwareHoldsBytes(
            weightsBytes: Int(weightsBytes), osReserveBytes: thresholds.osReserveBytes)

        // Weights alone don't fit inside {wired limit, RAM} minus reserve: definitely red,
        // regardless of KV/transient/allocator terms.
        guard headroom > 0 else {
            let binding: BindingConstraint = profile.wiredLimitBytes < profile.totalRAMBytes ? .wiredLimit : .physicalRAM
            return CapacityVerdict(
                color: .red, bindingConstraint: binding, ratio: .infinity,
                suggestedMitigation: mitigation(for: binding))
        }

        let nonWeightsPeak = prediction.totalBytes - weightsBytes
        let ratio = nonWeightsPeak / Double(headroom)

        let color: CapacityColor
        if ratio <= thresholds.greenMax { color = .green }
        else if ratio <= thresholds.yellowMax { color = .yellow }
        else { color = .red }

        let binding: BindingConstraint
        if color == .green {
            binding = .fits
        } else if prediction.modelType == .mlaAsImplemented {
            // The architecture's own KV cost is the wall (DeepSeek-R1: no hardware on the roadmap
            // holds this until absorbed-MLA ships) — name it specifically rather than generic RAM.
            binding = .mlaAsImplemented
        } else {
            binding = profile.wiredLimitBytes < profile.totalRAMBytes ? .wiredLimit : .physicalRAM
        }

        return CapacityVerdict(color: color, bindingConstraint: binding, ratio: ratio, suggestedMitigation: mitigation(for: binding))
    }

    /// Ranked mitigation naming (spec §6): drop KV to a lossy tier -> reduce concurrency -> pick a
    /// lighter-footprint model, in that leverage order; MLA and native-max cases get a specific note.
    private static func mitigation(for binding: BindingConstraint) -> String {
        switch binding {
        case .fits:
            return "fits: no mitigation needed"
        case .mlaAsImplemented:
            return "KV itself is the wall (MLA-as-implemented, no lossy tier changes this) — needs absorbed-MLA caching (backlog) or a smaller context"
        case .modelNativeMax:
            return "requested context exceeds the model's native max — reduce context; no hardware change helps"
        case .nativeMaxBelowDefault:
            return "model's native max is below the 32K tunable default — effective default is the native max"
        case .kvNotDerivable:
            return "KV cost not derivable (unconfirmed attention-layer count or out-of-scope arch) — confirm the model config before trusting any fit estimate"
        case .wiredLimit, .physicalRAM:
            return "drop KV to a lossy quant tier, then reduce concurrency, then pick a lighter-footprint model"
        }
    }

    /// `min(32768, model.nativeMaxContext)` — the effective default context (spec §4). Phi-4
    /// (16,384 native) therefore defaults to 16K, not 32K.
    public static func effectiveDefaultContext(_ m: ModelArchProfile) -> Int {
        min(32768, m.nativeMaxContext)
    }

    /// The advisory that pairs with `effectiveDefaultContext`: `.nativeMaxBelowDefault` when the
    /// model tops out below the 32K default (Phi-4 = 16K), else `nil`. This is surfaced by the
    /// default-selection surface, not `classify` (which assesses memory headroom, not the
    /// default-vs-native-max relationship) — spec §4/§8.
    public static func defaultContextAdvisory(_ m: ModelArchProfile) -> BindingConstraint? {
        m.nativeMaxContext < 32768 ? .nativeMaxBelowDefault : nil
    }

    /// `min(model.nativeMax, largest context that still fits under headroom)` — the tunable's true
    /// ceiling (spec §4). Binary-searches the largest context whose predicted peak stays at or
    /// under 100% of headroom (i.e., not red) at the given concurrency/kv_quant.
    public static func contextCeiling(
        model: ModelArchProfile, profile: SystemProfile, kvQuant: KVQuantTier, concurrency: Int,
        thresholds: CapacityThresholds = .default
    ) -> Int {
        func fits(_ context: Int) -> Bool {
            let prediction = predictPeakBytes(
                model: model, context: context, concurrency: concurrency, kvQuant: kvQuant, profile: profile)
            let verdict = classify(prediction, profile: profile, weightsBytes: Double(model.weightsBytes4bitEstimate), thresholds: thresholds)
            return verdict.color != .red
        }

        guard fits(1) else { return 0 } // doesn't even hold at context=1: nothing fits.
        var lo = 1, hi = model.nativeMaxContext
        if fits(hi) { return hi }
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2
            if fits(mid) { lo = mid } else { hi = mid }
        }
        return lo
    }
}
