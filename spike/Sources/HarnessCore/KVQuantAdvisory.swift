import Foundation

/// Pre-load validation + a sizing-only preview for an operator-requested KV-cache quantization tier
/// (`--kv-quant`, fit-checked-serve follow-on).
///
/// The serving RUNTIME stores KV in fp16 today; a requested non-fp16 tier is NOT applied. This type
/// therefore does two honest things and nothing else, so the flag can never over-promise:
///
///  1. `validateTier(_:)` — fail closed on an unknown tier string, so the serve exits before loading
///     rather than silently serving fp16 while the operator believes a quantized tier is in effect.
///  2. `previewLines(...)` — a what-if `ServingFitPlanner.decide` at the requested tier, rendered as
///     STDERR advisory lines each labeled `sizing_only runtime_not_wired`. It previews what the tier
///     WOULD buy (a higher context ceiling — the mitigation to name when the enforced fp16 verdict is
///     red), never a claim that the served KV is quantized.
///
/// The ENFORCED fit decision (the one that gates proceed/refuse and sets the MLX limits) is computed
/// separately, always at fp16, and is not touched here — so the advisory cannot manufacture a phantom
/// GREEN. All logic lives in this pure type; the `fastmlx-serve` call site is a thin ~4-line wiring
/// kept eyeball-verifiable because the executable target is not unit-testable off-box.
public enum KVQuantAdvisory {
    public enum Error: Swift.Error, Equatable, CustomStringConvertible, Sendable {
        case unknownTier(String)

        public var description: String {
            switch self {
            case .unknownTier(let raw):
                let shown = raw.isEmpty ? "(empty)" : raw
                return "--kv-quant \(shown) is not a known KV-cache quant tier "
                    + "(expected one of: \(KVQuantAdvisory.knownTierList))"
            }
        }
    }

    /// The tiers `KVQuantTier` models, comma-joined, for the error message and `--help` text.
    public static let knownTierList = "fp16, int8, turbo4, tq2_5, tq3_5"

    /// Fail closed: an unknown (typo'd or out-of-scope) tier string throws so the serve exits before
    /// loading a model, instead of silently serving fp16 under a tier the operator asked for.
    public static func validateTier(_ raw: String) throws -> KVQuantTier {
        guard let tier = KVQuantTier(rawValue: raw) else {
            throw Error.unknownTier(raw)
        }
        return tier
    }

    /// Advisory sizing preview for a requested tier. Returns `[]` for `.fp16` (identical to the
    /// enforced runtime — nothing to preview). Every returned line is STDERR advisory text labeled so
    /// no reader mistakes it for the enforced, applied verdict.
    ///
    /// The what-if `decide` matches the enforced call on every parameter EXCEPT `kvQuant`, so the
    /// ceiling delta is attributable to the tier alone. `weightsAreMeasured` does not feed the
    /// verdict, so a preview does not need a measured weights count to be honest.
    public static func previewLines(
        tier: KVQuantTier,
        profile: ModelArchProfile,
        host: SystemProfile,
        requestedContext: Int? = nil,
        concurrency: Int = 1,
        quantBits: Int? = nil
    ) -> [String] {
        guard tier != .fp16 else { return [] }

        let whatIf = ServingFitPlanner.decide(
            profile: profile, weightsAreMeasured: false, host: host,
            requestedContext: requestedContext, kvQuant: tier, concurrency: concurrency,
            quantBits: quantBits)

        func gib(_ bytes: Int) -> String { String(format: "%.2f GiB", Double(bytes) / 1_073_741_824.0) }
        let peakBytes = Int(whatIf.prediction.totalBytes.rounded(.up))

        let human = "fit-check advisory kv_quant=\(tier.rawValue) sizing_only runtime_not_wired: "
            + "kv@\(whatIf.servedContext)=\(gib(whatIf.maxReservedKVBytes)) "
            + "peak=\(gib(peakBytes)) ceiling=\(whatIf.contextCeiling) verdict=\(whatIf.color.rawValue) "
            + "— runtime stores KV in fp16; this tier is NOT applied (sizing preview only)"

        // Distinct `fit_advisory_` namespace: never reuses the frozen `fit_*` keys of
        // `machineReadableFields()`, which gate scripts anchor on. STDERR-only, its own line.
        let machine = "fit_advisory_kv_quant=\(tier.rawValue) "
            + "fit_advisory_ceiling=\(whatIf.contextCeiling) "
            + "fit_advisory_kv_bytes=\(whatIf.maxReservedKVBytes) "
            + "fit_advisory_peak_bytes=\(peakBytes) "
            + "fit_advisory_verdict=\(whatIf.color.rawValue) runtime_not_wired=true"

        return [human, machine]
    }
}
