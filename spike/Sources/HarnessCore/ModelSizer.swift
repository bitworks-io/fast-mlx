import Foundation

/// "Model sizer v1" — the llmfit-like moat pillar: given a Mac's RAM (a `SystemProfile`, real or
/// preset), tell the operator which cataloged quantized-model builds fit and at what context.
///
/// This file deliberately contains NO capacity math of its own. Every byte figure below is
/// produced by `CapacityModel.predictPeakBytes` / `.classify` / `.contextCeiling` — the only new
/// logic here is (a) scaling the catalog's 4-bit weights estimate to another weight-bit assumption
/// and (b) shaping the per-model, per-bit-width results into `ModelFit` rows for a report.

/// One (model, weight-bit-assumption) row of a `ModelSizer.report` result.
public struct ModelFit: Equatable, Sendable {
    public let modelID: String
    /// The quant assumption used for WEIGHTS in this row (e.g. 4, 8). Independent of `kvQuant`,
    /// which governs the KV-cache storage precision, not the weights.
    public let weightBits: Int
    public let weightsBytes: Int
    public let kvBytesAtContext: Int
    public let transientPrefillBytes: Int
    public let totalPeakBytes: Int
    public let fits: Bool
    public let maxContextThatFits: Int
    public let requestedContext: Int
    public let classification: CapacityColor
    /// `false` whenever this row's numbers rest on an estimate rather than a measurement: the
    /// the box's effective memory ceiling is synthesized/advisory, OR `kvQuant` is one of the
    /// ⚠️ EXPERIMENTAL/UNMEASURED placeholder tiers (`tq2_5`/`tq3_5`). The honest measured-vs-
    /// modeled flag — never present an estimate as a guarantee.
    public let estimateIsMeasured: Bool

    public init(
        modelID: String, weightBits: Int, weightsBytes: Int, kvBytesAtContext: Int,
        transientPrefillBytes: Int, totalPeakBytes: Int, fits: Bool, maxContextThatFits: Int,
        requestedContext: Int, classification: CapacityColor, estimateIsMeasured: Bool
    ) {
        self.modelID = modelID; self.weightBits = weightBits; self.weightsBytes = weightsBytes
        self.kvBytesAtContext = kvBytesAtContext; self.transientPrefillBytes = transientPrefillBytes
        self.totalPeakBytes = totalPeakBytes; self.fits = fits
        self.maxContextThatFits = maxContextThatFits; self.requestedContext = requestedContext
        self.classification = classification; self.estimateIsMeasured = estimateIsMeasured
    }
}

public enum ModelSizer {

    /// Human-facing provenance notes for the sizer report. Derive them from the effective binding
    /// source and the requested KV tier directly; `estimateIsMeasured == false` is intentionally
    /// insufficient because a synthesized/advisory host ceiling is not an experimental KV format.
    public static func provenanceNotes(box: SystemProfile, kvQuant: KVQuantTier) -> [String] {
        var notes: [String] = []

        switch box.effectiveMemoryCeiling.source {
        case .sharedPolicy:
            notes.append(
                "NOTE: the effective memory ceiling is synthesized by shared policy (not measured) — headroom numbers are approximate.")
        case .recommendedWorkingSet:
            notes.append(
                "NOTE: the effective memory ceiling is Metal's advisory recommended working set (not a measured hard limit) — headroom numbers are approximate.")
        case .wiredLimit where !box.wiredLimitIsMeasured:
            notes.append(
                "NOTE: this box's wired-memory limit is ESTIMATED (not read from hardware) — headroom numbers are approximate.")
        case .physicalRAM:
            notes.append(
                "NOTE: physical-RAM measurement provenance is unavailable — headroom numbers are approximate.")
        case .wiredLimit:
            break
        }

        if kvQuant == .tq2_5 || kvQuant == .tq3_5 {
            notes.append(
                "NOTE: \(kvQuant.rawValue) is an ⚠️ EXPERIMENTAL/UNMEASURED placeholder KV tier — treat fit/ceiling numbers as more speculative than usual.")
        }
        return notes
    }

    /// Report every cataloged model at the given weight-bit assumptions (default 4-bit and 8-bit)
    /// against a box, at a single context (or each model's own effective default when `context` is
    /// `nil` — mirrors `fastmlx-capacity`'s per-model default resolution).
    public static func report(
        box: SystemProfile, context: Int?, weightBitOptions: [Int] = [4, 8],
        kvQuant: KVQuantTier = .fp16, concurrency: Int = 1
    ) -> [ModelFit] {
        var rows: [ModelFit] = []
        rows.reserveCapacity(ModelArchProfile.catalog.count * weightBitOptions.count)

        let placeholderKVQuant = kvQuant == .tq2_5 || kvQuant == .tq3_5
        let estimateIsMeasured = box.effectiveMemoryCeilingIsMeasured && !placeholderKVQuant

        for model in ModelArchProfile.catalog {
            let resolvedContext = context ?? CapacityModel.effectiveDefaultContext(model)

            for bits in weightBitOptions {
                let scaled = scaledModel(model, weightBits: bits)

                let prediction = CapacityModel.predictPeakBytes(
                    model: scaled, context: resolvedContext, concurrency: concurrency,
                    kvQuant: kvQuant, profile: box)
                let verdict = CapacityModel.classify(
                    prediction, profile: box, weightsBytes: Double(scaled.weightsBytes4bitEstimate))
                let ceiling = CapacityModel.contextCeiling(
                    model: scaled, profile: box, kvQuant: kvQuant, concurrency: concurrency)

                rows.append(ModelFit(
                    modelID: model.id,
                    weightBits: bits,
                    weightsBytes: scaled.weightsBytes4bitEstimate,
                    kvBytesAtContext: Int(prediction.kvBytes),
                    transientPrefillBytes: Int(prediction.transientPrefillPeakBytes),
                    totalPeakBytes: Int(prediction.totalBytes),
                    fits: verdict.color != .red,
                    maxContextThatFits: ceiling,
                    requestedContext: resolvedContext,
                    classification: verdict.color,
                    estimateIsMeasured: estimateIsMeasured
                ))
            }
        }
        return rows
    }

    /// Scale the catalog's 4-bit weights estimate to another bit width: `bytes4bit × bits/4`. A
    /// documented first-order estimate (linear in bits/param), NOT a re-derivation of the "0.5
    /// GiB/B-param at 4-bit" rule the catalog itself is built on — reusing `CapacityModel`'s
    /// existing formulas against a model whose `weightsBytes4bitEstimate` field has been swapped
    /// for the scaled figure is what keeps this a wrapper rather than a second capacity model.
    private static func scaledModel(_ model: ModelArchProfile, weightBits: Int) -> ModelArchProfile {
        // Copy-and-modify, NOT a field-by-field rebuild: a field list here is exactly what
        // silently dropped `swaKVHeads`/`swaHeadDim`/`vHeadDim`/`swaVHeadDim`/`auxPerLayerKeyDim`
        // from every sizer-report row when they were added to `ModelArchProfile` after this
        // function was last written. `weightsBytes4bitEstimate` is `internal(set) var`
        // specifically so this can mutate just the one field that actually changes and let every
        // other field — present or future — pass through untouched.
        var scaled = model
        scaled.weightsBytes4bitEstimate = Int(Double(model.weightsBytes4bitEstimate) * Double(weightBits) / 4.0)
        return scaled
    }
}
