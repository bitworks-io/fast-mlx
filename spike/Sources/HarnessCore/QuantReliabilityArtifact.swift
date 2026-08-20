import Foundation

/// Decoder for a per-quant tool-call reliability artifact (build #2 of the per-quant reliability
/// rows work). This is a SEPARATE JSON file, deliberately NOT a widening of the bench CSV
/// (`BenchRow`): a new artifact is additive and reversible, whereas new CSV columns break the header
/// and force a versioning decision (see
/// docs/task-inbox/2026-08-18-per-quant-tool-call-reliability-rows.md). Build #3 (M5-gated) will have
/// the Python probe emit this shape after serving each quant; this decoder gives that collection a
/// stable schema to target and lets the whole parse/render path be unit-tested off-box.
///
/// The per-quant `reliability` block reuses the field names `scripts/bench-tool-calling.py` already
/// emits (`cases`, `trigger_rate`, `name_accuracy`, `arg_validity_rate`), so a multi-quant collector
/// is just N of the existing single-model probe outputs wrapped with `repo_id`/`quant_bits`.
///
/// Honesty invariant: the current probe measures NO repair data, so `repair_rate` is optional. When
/// absent, the produced `QuantReliabilityRow` carries `repairRateIsMeasured == false` and the report
/// renders `repair=n/a` — an unmeasured field is never presented as a measured `0.00`.
public struct QuantReliabilityArtifact: Decodable, Sendable {
    /// Optional schema tag (e.g. `"quant-reliability/v1"`) for forward-compat; not required to decode.
    public let schema: String?
    /// The base model these quants are variants of (for the report header / provenance).
    public let model: String?
    public let quants: [Entry]

    /// One quant variant's measured reliability, keyed by its fetchable repo id and quant bit-width.
    public struct Entry: Decodable, Sendable {
        public let repoID: String
        /// `nil` marks an unquantized checkpoint (mirrors `QuantReliabilityRow.quantBits`).
        public let quantBits: Int?
        public let reliability: ReliabilityBlock

        enum CodingKeys: String, CodingKey {
            case repoID = "repo_id"
            case quantBits = "quant_bits"
            case reliability
        }
    }

    /// The probe's reliability summary. `repairRate` is optional: absent ⟹ unmeasured (render n/a).
    public struct ReliabilityBlock: Decodable, Sendable {
        public let cases: Int
        public let triggerRate: Double
        public let nameAccuracy: Double
        public let argValidityRate: Double
        public let repairRate: Double?

        enum CodingKeys: String, CodingKey {
            case cases
            case triggerRate = "trigger_rate"
            case nameAccuracy = "name_accuracy"
            case argValidityRate = "arg_validity_rate"
            case repairRate = "repair_rate"
        }
    }

    /// Decode an artifact from raw JSON bytes.
    public static func decode(from data: Data) throws -> QuantReliabilityArtifact {
        try JSONDecoder().decode(QuantReliabilityArtifact.self, from: data)
    }

    /// Project the decoded entries into renderer rows. A missing `repair_rate` yields a score with a
    /// `0.0` placeholder (the type's field is non-optional) but `repairRateIsMeasured == false`, so
    /// `QuantReliabilityReport` renders `repair=n/a` for it rather than the placeholder.
    public func rows() -> [QuantReliabilityRow] {
        quants.map { entry in
            let r = entry.reliability
            let score = ToolCallReliabilityScore(
                caseCount: r.cases,
                triggerRate: r.triggerRate,
                nameAccuracy: r.nameAccuracy,
                argumentValidityRate: r.argValidityRate,
                repairRate: r.repairRate ?? 0.0)
            return QuantReliabilityRow(
                repoID: entry.repoID, quantBits: entry.quantBits, score: score,
                repairRateIsMeasured: r.repairRate != nil)
        }
    }
}
