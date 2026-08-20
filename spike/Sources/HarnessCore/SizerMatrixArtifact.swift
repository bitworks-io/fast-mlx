import Foundation

/// A schema-tagged, fully `Codable` snapshot of a `ModelSizer.report(...)` run — the data source for
/// the public "which model fits which Mac" sizer page. Mirrors `QuantReliabilityArtifact` /
/// `QuantReliabilityRendering`'s precedent: an optional `schema` tag on decode, a fail-closed
/// `decodeValidated(from:)` gate for consumers (the CLI, and eventually the site's ingest step), and
/// a `build(...)` factory that projects `HarnessCore`'s own `ModelFit` rows into wire shape.
///
/// `ModelFit` itself stays `Equatable, Sendable` only (per spec) — this artifact's `Row` duplicates
/// its fields rather than making `ModelFit` conform to `Codable`.
///
/// Wire format: CAMELCASE keys (`modelID`, `weightBits`, ...), matching `sizerReportJSON`'s existing
/// `--sizer --json` output in `fastmlx-capacity/main.swift` — chosen so the two JSON surfaces stay
/// consistent for anything that already parses the `--sizer --json` shape. (This differs from
/// `QuantReliabilityArtifact`'s snake_case, which mirrors an external Python emitter's field names;
/// there is no such external producer here.)
public struct SizerMatrixArtifact: Codable, Sendable {
    /// The schema tag this artifact is pinned to (`"sizer-matrix/v1"`).
    public static let schemaTag = "sizer-matrix/v1"

    /// Optional schema tag for forward-compat; not required to decode (mirrors
    /// `QuantReliabilityArtifact.schema`). `decodeValidated` is the gate that rejects a foreign tag.
    public let schema: String?
    public let host: Host
    /// The KV-cache quant tier's `rawValue` the matrix was computed with (provenance).
    public let kvQuant: String
    /// The concurrency the matrix was computed with (provenance).
    public let concurrency: Int
    public let rows: [Row]

    /// The box this matrix was computed against, projected from `SystemProfile`.
    public struct Host: Codable, Sendable {
        /// The box label (a `resolveBox`-style preset name, e.g. `"m5Max128"`, or `"host"`/`"auto"`).
        public let label: String
        public let ramBytes: Int
        public let wiredLimitBytes: Int
        public let wiredLimitIsMeasured: Bool

        public init(label: String, ramBytes: Int, wiredLimitBytes: Int, wiredLimitIsMeasured: Bool) {
            self.label = label
            self.ramBytes = ramBytes
            self.wiredLimitBytes = wiredLimitBytes
            self.wiredLimitIsMeasured = wiredLimitIsMeasured
        }
    }

    /// One `ModelFit` row, projected into wire shape (field-for-field copy of `ModelFit`).
    public struct Row: Codable, Sendable {
        public let modelID: String
        public let weightBits: Int
        public let weightsBytes: Int
        public let kvBytesAtContext: Int
        public let transientPrefillBytes: Int
        public let totalPeakBytes: Int
        public let fits: Bool
        public let maxContextThatFits: Int
        public let requestedContext: Int
        /// The `CapacityColor` rawValue (`"green"`/`"yellow"`/`"red"`).
        public let classification: String
        public let estimateIsMeasured: Bool

        public init(
            modelID: String, weightBits: Int, weightsBytes: Int, kvBytesAtContext: Int,
            transientPrefillBytes: Int, totalPeakBytes: Int, fits: Bool, maxContextThatFits: Int,
            requestedContext: Int, classification: String, estimateIsMeasured: Bool
        ) {
            self.modelID = modelID
            self.weightBits = weightBits
            self.weightsBytes = weightsBytes
            self.kvBytesAtContext = kvBytesAtContext
            self.transientPrefillBytes = transientPrefillBytes
            self.totalPeakBytes = totalPeakBytes
            self.fits = fits
            self.maxContextThatFits = maxContextThatFits
            self.requestedContext = requestedContext
            self.classification = classification
            self.estimateIsMeasured = estimateIsMeasured
        }

        init(_ fit: ModelFit) {
            self.init(
                modelID: fit.modelID, weightBits: fit.weightBits, weightsBytes: fit.weightsBytes,
                kvBytesAtContext: fit.kvBytesAtContext, transientPrefillBytes: fit.transientPrefillBytes,
                totalPeakBytes: fit.totalPeakBytes, fits: fit.fits,
                maxContextThatFits: fit.maxContextThatFits, requestedContext: fit.requestedContext,
                classification: fit.classification.rawValue, estimateIsMeasured: fit.estimateIsMeasured)
        }
    }

    public init(schema: String?, host: Host, kvQuant: String, concurrency: Int, rows: [Row]) {
        self.schema = schema
        self.host = host
        self.kvQuant = kvQuant
        self.concurrency = concurrency
        self.rows = rows
    }

    /// Run `ModelSizer.report(...)` against `box` and project the result into an artifact tagged
    /// `schemaTag`. `boxLabel` carries the `resolveBox`-style preset name into `host.label`.
    public static func build(
        box: SystemProfile, boxLabel: String, context: Int?, kvQuant: KVQuantTier, concurrency: Int
    ) -> SizerMatrixArtifact {
        let fits = ModelSizer.report(box: box, context: context, kvQuant: kvQuant, concurrency: concurrency)
        let host = Host(
            label: boxLabel, ramBytes: box.totalRAMBytes, wiredLimitBytes: box.wiredLimitBytes,
            wiredLimitIsMeasured: box.wiredLimitIsMeasured)
        return SizerMatrixArtifact(
            schema: schemaTag, host: host, kvQuant: kvQuant.rawValue, concurrency: concurrency,
            rows: fits.map(Row.init))
    }

    /// Encode this artifact as deterministic (sorted-key) JSON — stable output for fixtures/diffs.
    public func encodedJSON() -> String {
        let encoder = JSONEncoder()
        if #available(macOS 10.15, *) {
            encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        } else {
            encoder.outputFormatting = [.sortedKeys]
        }
        // swiftlint:disable:next force_try — encoding this pure-value type cannot throw.
        let data = try! encoder.encode(self)
        return String(decoding: data, as: UTF8.self)
    }

    /// Decode an artifact from raw JSON bytes. Permissive: does NOT validate `schema` (mirrors
    /// `QuantReliabilityArtifact.decode`) — `decodeValidated` below is the fail-closed gate.
    public static func decode(from data: Data) throws -> SizerMatrixArtifact {
        try JSONDecoder().decode(SizerMatrixArtifact.self, from: data)
    }

    /// The schema-unaware-versus-foreign error this artifact's fail-closed gate throws.
    public enum SizerMatrixError: Error, CustomStringConvertible, Equatable {
        /// The artifact declared a `schema` this build does not understand (associated: the tag).
        case unsupportedSchema(String)

        public var description: String {
            switch self {
            case .unsupportedSchema(let tag):
                return "unsupported sizer-matrix schema '\(tag)' "
                    + "(this build understands '\(SizerMatrixArtifact.schemaTag)' or none)"
            }
        }
    }

    /// Decode `data` and validate its schema — fails closed on a foreign `schema` tag (mirrors
    /// `QuantReliabilityArtifactRenderer.decodeValidated`). An absent `schema` is still accepted
    /// (hand-built or pre-schema artifacts).
    public static func decodeValidated(from data: Data) throws -> SizerMatrixArtifact {
        let artifact = try decode(from: data)
        if let schema = artifact.schema, schema != schemaTag {
            throw SizerMatrixError.unsupportedSchema(schema)
        }
        return artifact
    }
}
