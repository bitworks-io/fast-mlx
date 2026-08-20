import Foundation

/// The off-box CLI seam for roadmap #4's per-quant reliability rows: turn a `quant-reliability/v1`
/// artifact's raw BYTES into display lines an operator reads (`fastmlx-capacity --quant-reliability
/// <file>`). It composes the two frozen pieces — `QuantReliabilityArtifact.decode(_:)` +
/// `QuantReliabilityReport.summaryLines()` — and adds the one guard the permissive decoder
/// deliberately omits: it FAILS CLOSED on a foreign `schema` tag so a future `quant-reliability/v2`
/// file is never silently rendered as if it were v1. An absent schema stays allowed (hand-built or
/// pre-schema artifacts). All wrong-input paths surface as this type's own `RenderError`, so the CLI
/// can print one clean message and exit non-zero rather than leaking a raw `DecodingError`.
public enum QuantReliabilityArtifactRenderer {
    /// The schema tag this renderer understands. `nil` (absent) is also accepted; anything else fails.
    public static let supportedSchema = "quant-reliability/v1"

    public enum RenderError: Error, CustomStringConvertible, Equatable {
        /// The artifact declared a `schema` this renderer does not understand (associated: the tag).
        case unsupportedSchema(String)
        /// The bytes were not decodable as a `quant-reliability` artifact (associated: a short reason).
        case malformedJSON(String)

        public var description: String {
            switch self {
            case .unsupportedSchema(let tag):
                return "unsupported quant-reliability schema '\(tag)' "
                    + "(this build understands '\(QuantReliabilityArtifactRenderer.supportedSchema)' or none)"
            case .malformedJSON(let reason):
                return "malformed quant-reliability artifact: \(reason)"
            }
        }
    }

    /// Decode `data` and validate its schema — the shared gate for every artifact consumer (the
    /// `fastmlx-capacity` renderer and the `fastmlx-serve` pick overlay). This is the one guard the
    /// permissive `QuantReliabilityArtifact.decode` deliberately omits.
    /// - Throws: `RenderError.malformedJSON` on undecodable bytes, `RenderError.unsupportedSchema` on a
    ///   foreign `schema` tag.
    public static func decodeValidated(from data: Data) throws -> QuantReliabilityArtifact {
        let artifact: QuantReliabilityArtifact
        do {
            artifact = try QuantReliabilityArtifact.decode(from: data)
        } catch {
            throw RenderError.malformedJSON(String(describing: error))
        }
        if let schema = artifact.schema, schema != supportedSchema {
            throw RenderError.unsupportedSchema(schema)
        }
        return artifact
    }

    /// Decode `data`, validate its schema, and render one provenance header line plus the report rows.
    /// - Throws: `RenderError.malformedJSON` on undecodable bytes, `RenderError.unsupportedSchema` on a
    ///   foreign `schema` tag.
    public static func renderLines(from data: Data) throws -> [String] {
        let artifact = try decodeValidated(from: data)
        let model = artifact.model ?? "(unspecified)"
        let schemaLabel = artifact.schema ?? "unversioned"
        var lines = ["quant reliability — model: \(model) (schema \(schemaLabel))"]
        lines.append(contentsOf: QuantReliabilityReport(rows: artifact.rows()).summaryLines())
        return lines
    }
}
