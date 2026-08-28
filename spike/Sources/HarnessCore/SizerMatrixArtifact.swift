import Foundation

/// A schema-tagged, fully `Codable` snapshot of a `ModelSizer.report(...)` run — the data source for
/// the public "which model fits which Mac" sizer page. Mirrors `QuantReliabilityArtifact` /
/// `QuantReliabilityRendering`'s precedent: an optional `schema` tag on decode, a fail-closed
/// `decodeValidated(from:)` gate for consumers (the CLI, and eventually the site's ingest step), and
/// a `build(...)` factory that projects `HarnessCore`'s own `ModelFit` rows plus host memory
/// provenance into wire shape.
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
    /// The current schema tag this artifact emits.
    public static let schemaTag = "sizer-matrix/v2"
    /// The first published schema tag, still accepted by `decodeValidated`.
    public static let legacySchemaTag = "sizer-matrix/v1"

    /// Optional schema tag for forward-compat; not required to decode (mirrors
    /// `QuantReliabilityArtifact.schema`). `decodeValidated` is the gate that rejects a foreign tag.
    public let schema: String?
    public let host: Host
    /// The KV-cache quant tier's `rawValue` the matrix was computed with (provenance).
    public let kvQuant: String
    /// The concurrency the matrix was computed with (provenance).
    public let concurrency: Int
    public let rows: [Row]

    public enum LiveHostLabel: String, Sendable, Equatable {
        case host
        case auto
    }

    public enum BuildObservation: Sendable, Equatable {
        case live(label: LiveHostLabel, currentMetalAllocatedBytes: Int?)
        case modeledPreset(label: String)

        var label: String {
            switch self {
            case .live(let label, _):
                return label.rawValue
            case .modeledPreset(let label):
                return label
            }
        }

        var source: String {
            switch self {
            case .live:
                return HostObservationSource.live.rawValue
            case .modeledPreset:
                return HostObservationSource.modeledPreset.rawValue
            }
        }

        var currentMetalAllocatedBytes: Int? {
            switch self {
            case .live(_, let currentMetalAllocatedBytes):
                return currentMetalAllocatedBytes
            case .modeledPreset:
                return nil
            }
        }
    }

    /// The box this matrix was computed against, projected from `SystemProfile`.
    public struct Host: Codable, Sendable {
        /// The box label (a `resolveBox`-style preset name, e.g. `"m5Max128"`, or `"host"`/`"auto"`).
        public let label: String
        public let observationSource: String?
        public let hostUse: String?
        public let hostUseSource: String?
        public let hostUsePolicyVersion: String?
        public let physicalRAMBytes: Int
        public let wiredLimitBytes: Int
        public let wiredLimitProvenance: String?
        public let metalRecommendedWorkingSetBytes: Int?
        public let metalCurrentAllocatedBytes: Int?
        public let effectiveMemoryCeilingBytes: Int?
        public let effectiveMemoryCeilingSource: String?
        public let osServiceReserveBytes: Int?
        public let mlxMemoryLimitBytes: Int?
        public let mlxCacheLimitBytes: Int?

        public var ramBytes: Int { physicalRAMBytes }
        public var wiredLimitIsMeasured: Bool { wiredLimitProvenance == "measured" }
        let decodedLegacyRamBytesPresent: Bool
        let decodedLegacyWiredLimitIsMeasuredPresent: Bool

        private enum CodingKeys: String, CodingKey {
            case label
            case observationSource
            case hostUse
            case hostUseSource
            case hostUsePolicyVersion
            case physicalRAMBytes
            case ramBytes
            case wiredLimitBytes
            case wiredLimitIsMeasured
            case wiredLimitProvenance
            case metalRecommendedWorkingSetBytes
            case metalCurrentAllocatedBytes
            case effectiveMemoryCeilingBytes
            case effectiveMemoryCeilingSource
            case osServiceReserveBytes
            case mlxMemoryLimitBytes
            case mlxCacheLimitBytes
        }

        public init(
            label: String,
            observationSource: String?,
            hostUse: String?,
            hostUseSource: String?,
            hostUsePolicyVersion: String?,
            physicalRAMBytes: Int,
            wiredLimitBytes: Int,
            wiredLimitProvenance: String?,
            metalRecommendedWorkingSetBytes: Int?,
            metalCurrentAllocatedBytes: Int?,
            effectiveMemoryCeilingBytes: Int?,
            effectiveMemoryCeilingSource: String?,
            osServiceReserveBytes: Int?,
            mlxMemoryLimitBytes: Int?,
            mlxCacheLimitBytes: Int?,
            decodedLegacyRamBytesPresent: Bool = false,
            decodedLegacyWiredLimitIsMeasuredPresent: Bool = false
        ) {
            self.label = label
            self.observationSource = observationSource
            self.hostUse = hostUse
            self.hostUseSource = hostUseSource
            self.hostUsePolicyVersion = hostUsePolicyVersion
            self.physicalRAMBytes = physicalRAMBytes
            self.wiredLimitBytes = wiredLimitBytes
            self.wiredLimitProvenance = wiredLimitProvenance
            self.metalRecommendedWorkingSetBytes = metalRecommendedWorkingSetBytes
            self.metalCurrentAllocatedBytes = metalCurrentAllocatedBytes
            self.effectiveMemoryCeilingBytes = effectiveMemoryCeilingBytes
            self.effectiveMemoryCeilingSource = effectiveMemoryCeilingSource
            self.osServiceReserveBytes = osServiceReserveBytes
            self.mlxMemoryLimitBytes = mlxMemoryLimitBytes
            self.mlxCacheLimitBytes = mlxCacheLimitBytes
            self.decodedLegacyRamBytesPresent = decodedLegacyRamBytesPresent
            self.decodedLegacyWiredLimitIsMeasuredPresent = decodedLegacyWiredLimitIsMeasuredPresent
        }

        public init(label: String, ramBytes: Int, wiredLimitBytes: Int, wiredLimitIsMeasured: Bool) {
            self.init(
                label: label,
                observationSource: nil,
                hostUse: nil,
                hostUseSource: nil,
                hostUsePolicyVersion: nil,
                physicalRAMBytes: ramBytes,
                wiredLimitBytes: wiredLimitBytes,
                wiredLimitProvenance: wiredLimitIsMeasured ? "measured" : "synthesized",
                metalRecommendedWorkingSetBytes: nil,
                metalCurrentAllocatedBytes: nil,
                effectiveMemoryCeilingBytes: nil,
                effectiveMemoryCeilingSource: nil,
                osServiceReserveBytes: nil,
                mlxMemoryLimitBytes: nil,
                mlxCacheLimitBytes: nil,
                decodedLegacyRamBytesPresent: true,
                decodedLegacyWiredLimitIsMeasuredPresent: true)
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let label = try container.decode(String.self, forKey: .label)
            let legacyRamBytes = try container.decodeIfPresent(Int.self, forKey: .ramBytes)
            let physicalRAMBytes = try container.decodeIfPresent(Int.self, forKey: .physicalRAMBytes)
                ?? legacyRamBytes
                ?? (try SizerMatrixArtifact.decodingError(
                    forKey: CodingKeys.ramBytes, in: container, "missing physicalRAMBytes/ramBytes")
                )
            let wiredLimitBytes = try container.decode(Int.self, forKey: .wiredLimitBytes)
            let legacyWiredLimitIsMeasured = try container.decodeIfPresent(Bool.self, forKey: .wiredLimitIsMeasured)
            let wiredLimitProvenance = try container.decodeIfPresent(String.self, forKey: .wiredLimitProvenance)
                ?? legacyWiredLimitIsMeasured.map {
                    $0 ? "measured" : "synthesized"
                }
                ?? (try SizerMatrixArtifact.decodingError(
                    forKey: CodingKeys.wiredLimitProvenance, in: container,
                    "missing wiredLimitProvenance/wiredLimitIsMeasured")
                )
            self.init(
                label: label,
                observationSource: try container.decodeIfPresent(String.self, forKey: .observationSource),
                hostUse: try container.decodeIfPresent(String.self, forKey: .hostUse),
                hostUseSource: try container.decodeIfPresent(String.self, forKey: .hostUseSource),
                hostUsePolicyVersion: try container.decodeIfPresent(String.self, forKey: .hostUsePolicyVersion),
                physicalRAMBytes: physicalRAMBytes,
                wiredLimitBytes: wiredLimitBytes,
                wiredLimitProvenance: wiredLimitProvenance,
                metalRecommendedWorkingSetBytes: try container.decodeIfPresent(
                    Int.self, forKey: .metalRecommendedWorkingSetBytes),
                metalCurrentAllocatedBytes: try container.decodeIfPresent(
                    Int.self, forKey: .metalCurrentAllocatedBytes),
                effectiveMemoryCeilingBytes: try container.decodeIfPresent(
                    Int.self, forKey: .effectiveMemoryCeilingBytes),
                effectiveMemoryCeilingSource: try container.decodeIfPresent(
                    String.self, forKey: .effectiveMemoryCeilingSource),
                osServiceReserveBytes: try container.decodeIfPresent(Int.self, forKey: .osServiceReserveBytes),
                mlxMemoryLimitBytes: try container.decodeIfPresent(Int.self, forKey: .mlxMemoryLimitBytes),
                mlxCacheLimitBytes: try container.decodeIfPresent(Int.self, forKey: .mlxCacheLimitBytes),
                decodedLegacyRamBytesPresent: legacyRamBytes != nil,
                decodedLegacyWiredLimitIsMeasuredPresent: legacyWiredLimitIsMeasured != nil)
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(label, forKey: .label)
            try encodeNullable(observationSource, forKey: .observationSource, into: &container)
            try encodeNullable(hostUse, forKey: .hostUse, into: &container)
            try encodeNullable(hostUseSource, forKey: .hostUseSource, into: &container)
            try encodeNullable(hostUsePolicyVersion, forKey: .hostUsePolicyVersion, into: &container)
            try container.encode(physicalRAMBytes, forKey: .physicalRAMBytes)
            try container.encode(wiredLimitBytes, forKey: .wiredLimitBytes)
            try encodeNullable(wiredLimitProvenance, forKey: .wiredLimitProvenance, into: &container)
            try encodeNullable(metalRecommendedWorkingSetBytes, forKey: .metalRecommendedWorkingSetBytes, into: &container)
            try encodeNullable(metalCurrentAllocatedBytes, forKey: .metalCurrentAllocatedBytes, into: &container)
            try encodeNullable(effectiveMemoryCeilingBytes, forKey: .effectiveMemoryCeilingBytes, into: &container)
            try encodeNullable(effectiveMemoryCeilingSource, forKey: .effectiveMemoryCeilingSource, into: &container)
            try encodeNullable(osServiceReserveBytes, forKey: .osServiceReserveBytes, into: &container)
            try encodeNullable(mlxMemoryLimitBytes, forKey: .mlxMemoryLimitBytes, into: &container)
            try encodeNullable(mlxCacheLimitBytes, forKey: .mlxCacheLimitBytes, into: &container)
        }

        private func encodeNullable<T: Encodable>(
            _ value: T?,
            forKey key: CodingKeys,
            into container: inout KeyedEncodingContainer<CodingKeys>
        ) throws {
            if let value {
                try container.encode(value, forKey: key)
            } else {
                try container.encodeNil(forKey: key)
            }
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
    /// `schemaTag`. `observation` carries both the trusted live-versus-modeled source and its label,
    /// making a live observation with a preset label unrepresentable through the builder API.
    public static func build(
        box: SystemProfile, context: Int?, kvQuant: KVQuantTier, concurrency: Int,
        observation: BuildObservation
    ) -> SizerMatrixArtifact {
        precondition(context.map { $0 >= 0 } ?? true, "context must be nonnegative")
        precondition(concurrency > 0, "concurrency must be positive")
        if let currentMetalAllocatedBytes = observation.currentMetalAllocatedBytes {
            precondition(currentMetalAllocatedBytes >= 0, "currentMetalAllocatedBytes must be nonnegative")
        }
        let fits = ModelSizer.report(box: box, context: context, kvQuant: kvQuant, concurrency: concurrency)
        let effectiveCeiling = box.effectiveMemoryCeiling
        let memoryLimitBytes = effectiveCeiling.bytes
        let cacheLimitBytes = CapacityModel.recommendedCacheLimitBytes(wiredLimitBytes: memoryLimitBytes)
        let recommendedWorkingSetBytes = box.recommendedWorkingSetBytes.flatMap { $0 > 0 ? $0 : nil }
        let host = Host(
            label: observation.label,
            observationSource: observation.source,
            hostUse: box.hostUse.rawValue,
            hostUseSource: box.hostUse.source.rawValue,
            hostUsePolicyVersion: box.hostUse.policyVersion,
            physicalRAMBytes: box.totalRAMBytes,
            wiredLimitBytes: box.wiredLimitBytes,
            wiredLimitProvenance: box.wiredLimitIsMeasured ? "measured" : "synthesized",
            metalRecommendedWorkingSetBytes: recommendedWorkingSetBytes,
            metalCurrentAllocatedBytes: observation.currentMetalAllocatedBytes,
            effectiveMemoryCeilingBytes: effectiveCeiling.bytes,
            effectiveMemoryCeilingSource: effectiveCeiling.source.rawValue,
            osServiceReserveBytes: CapacityThresholds.default.osReserveBytes,
            mlxMemoryLimitBytes: memoryLimitBytes,
            mlxCacheLimitBytes: cacheLimitBytes)
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
        case invalidLegacy(String)
        case invalidV2(String)

        public var description: String {
            switch self {
            case .unsupportedSchema(let tag):
                return "unsupported sizer-matrix schema '\(tag)' "
                    + "(this build understands '\(SizerMatrixArtifact.schemaTag)', "
                    + "'\(SizerMatrixArtifact.legacySchemaTag)' or none)"
            case .invalidLegacy(let reason):
                return "invalid legacy sizer-matrix artifact: \(reason)"
            case .invalidV2(let reason):
                return "invalid sizer-matrix/v2 artifact: \(reason)"
            }
        }
    }

    /// Decode `data` and validate its schema — fails closed on a foreign `schema` tag (mirrors
    /// `QuantReliabilityArtifactRenderer.decodeValidated`). An absent `schema` is still accepted
    /// (hand-built or pre-schema artifacts).
    public static func decodeValidated(from data: Data) throws -> SizerMatrixArtifact {
        struct SchemaProbe: Decodable { let schema: String? }
        let schema = try JSONDecoder().decode(SchemaProbe.self, from: data).schema
        if let schema, schema != schemaTag && schema != legacySchemaTag {
            throw SizerMatrixError.unsupportedSchema(schema)
        }
        let artifact = try decode(from: data)
        if artifact.schema == schemaTag {
            try artifact.validateV2()
        } else {
            try artifact.validateLegacy()
        }
        return artifact
    }

    private static func decodingError<K: CodingKey, T>(
        forKey key: K,
        in container: KeyedDecodingContainer<K>,
        _ message: String
    ) throws -> T {
        throw DecodingError.dataCorruptedError(forKey: key, in: container, debugDescription: message)
    }

    private enum WiredLimitProvenance: String {
        case measured
        case synthesized
    }

    private enum HostObservationSource: String {
        case live
        case modeledPreset = "modeled-preset"
    }

    private func validateLegacy() throws {
        guard !host.label.isEmpty else { throw SizerMatrixError.invalidLegacy("missing label") }
        guard host.decodedLegacyRamBytesPresent else { throw SizerMatrixError.invalidLegacy("missing ramBytes") }
        guard host.decodedLegacyWiredLimitIsMeasuredPresent else {
            throw SizerMatrixError.invalidLegacy("missing wiredLimitIsMeasured")
        }
        guard host.physicalRAMBytes > 0 else { throw SizerMatrixError.invalidLegacy("nonpositive ramBytes") }
        guard host.wiredLimitBytes > 0 else { throw SizerMatrixError.invalidLegacy("nonpositive wiredLimitBytes") }
    }

    private func validateV2() throws {
        guard concurrency > 0 else { throw SizerMatrixError.invalidV2("nonpositive concurrency") }
        guard let decodedKVQuant = KVQuantTier(rawValue: kvQuant) else {
            throw SizerMatrixError.invalidV2("unknown kvQuant")
        }
        guard !host.label.isEmpty else { throw SizerMatrixError.invalidV2("missing label") }
        guard let observationSourceRaw = host.observationSource,
            let observationSource = HostObservationSource(rawValue: observationSourceRaw)
        else { throw SizerMatrixError.invalidV2("missing or unknown observationSource") }
        switch observationSource {
        case .live:
            guard LiveHostLabel(rawValue: host.label) != nil else {
                throw SizerMatrixError.invalidV2("live observation requires host/auto label")
            }
        case .modeledPreset:
            guard LiveHostLabel(rawValue: host.label) == nil else {
                throw SizerMatrixError.invalidV2("modeled preset cannot use host/auto label")
            }
            guard host.metalCurrentAllocatedBytes == nil else {
                throw SizerMatrixError.invalidV2("modeled preset cannot carry current Metal allocation")
            }
        }
        guard let hostUseRaw = host.hostUse,
            let hostUse = HostUseClassification.Use(rawValue: hostUseRaw)
        else { throw SizerMatrixError.invalidV2("missing or unknown hostUse") }
        guard let hostUseSourceRaw = host.hostUseSource,
            let hostUseSource = HostUseClassification.Source(rawValue: hostUseSourceRaw)
        else { throw SizerMatrixError.invalidV2("missing or unknown hostUseSource") }
        guard host.hostUsePolicyVersion == HostUseClassification.currentPolicyVersion else {
            throw SizerMatrixError.invalidV2("missing hostUsePolicyVersion")
        }
        let hostUseClassification: HostUseClassification
        switch (hostUse, hostUseSource) {
        case (.shared, .default):
            hostUseClassification = .defaultShared
        case (.shared, .automatic):
            hostUseClassification = .automaticShared
        case (.shared, .operatorAssertion):
            hostUseClassification = .operatorAssertedShared()
        case (.dedicatedServing, .operatorAssertion):
            hostUseClassification = .operatorAssertedDedicatedServing()
        case (.dedicatedServing, .default), (.dedicatedServing, .automatic):
            throw SizerMatrixError.invalidV2("dedicated-serving host use requires operator assertion")
        }
        guard host.physicalRAMBytes > 0 else { throw SizerMatrixError.invalidV2("missing physicalRAMBytes") }
        guard host.wiredLimitBytes > 0 else { throw SizerMatrixError.invalidV2("missing wiredLimitBytes") }
        guard host.wiredLimitBytes <= host.physicalRAMBytes else {
            throw SizerMatrixError.invalidV2("wiredLimitBytes must not exceed physicalRAMBytes")
        }
        guard let wiredLimitProvenanceRaw = host.wiredLimitProvenance,
            let wiredLimitProvenance = WiredLimitProvenance(rawValue: wiredLimitProvenanceRaw)
        else {
            throw SizerMatrixError.invalidV2("missing or unknown wiredLimitProvenance")
        }
        if let recommended = host.metalRecommendedWorkingSetBytes {
            guard recommended > 0, recommended <= host.physicalRAMBytes else {
                throw SizerMatrixError.invalidV2("invalid metalRecommendedWorkingSetBytes")
            }
        }
        if let current = host.metalCurrentAllocatedBytes {
            guard current >= 0, current <= host.physicalRAMBytes else {
                throw SizerMatrixError.invalidV2("invalid metalCurrentAllocatedBytes")
            }
        }
        guard let effective = host.effectiveMemoryCeilingBytes, effective > 0 else {
            throw SizerMatrixError.invalidV2("missing effectiveMemoryCeilingBytes")
        }
        guard effective <= host.physicalRAMBytes else {
            throw SizerMatrixError.invalidV2("effectiveMemoryCeilingBytes must not exceed physicalRAMBytes")
        }
        guard let effectiveSourceRaw = host.effectiveMemoryCeilingSource,
            let effectiveSource = EffectiveMemoryCeiling.Source(rawValue: effectiveSourceRaw)
        else {
            throw SizerMatrixError.invalidV2("missing or unknown effectiveMemoryCeilingSource")
        }
        let reconstructedProfile = SystemProfile(
            chip: host.label,
            totalRAMBytes: host.physicalRAMBytes,
            wiredLimitBytes: host.wiredLimitBytes,
            wiredLimitIsMeasured: wiredLimitProvenance == .measured,
            recommendedWorkingSetBytes: host.metalRecommendedWorkingSetBytes,
            hostUse: hostUseClassification)
        let reconstructedEffective = reconstructedProfile.effectiveMemoryCeiling
        guard effective == reconstructedEffective.bytes, effectiveSource == reconstructedEffective.source else {
            throw SizerMatrixError.invalidV2("effective memory ceiling does not match host policy")
        }
        guard let reserve = host.osServiceReserveBytes, reserve > 0 else {
            throw SizerMatrixError.invalidV2("missing nonzero osServiceReserveBytes")
        }
        guard reserve == CapacityThresholds.default.osReserveBytes else {
            throw SizerMatrixError.invalidV2("osServiceReserveBytes must match matrix policy")
        }
        guard let memory = host.mlxMemoryLimitBytes, memory > 0 else {
            throw SizerMatrixError.invalidV2("missing mlxMemoryLimitBytes")
        }
        guard let cache = host.mlxCacheLimitBytes, cache > 0 else {
            throw SizerMatrixError.invalidV2("missing mlxCacheLimitBytes")
        }
        guard memory == effective else {
            throw SizerMatrixError.invalidV2("mlxMemoryLimitBytes must equal effectiveMemoryCeilingBytes")
        }
        guard cache == CapacityModel.recommendedCacheLimitBytes(wiredLimitBytes: memory) else {
            throw SizerMatrixError.invalidV2("mlxCacheLimitBytes must match recommended cache limit")
        }
        guard cache < memory else { throw SizerMatrixError.invalidV2("mlxCacheLimitBytes must be below memory") }
        let placeholderKVQuant = decodedKVQuant == .tq2_5 || decodedKVQuant == .tq3_5
        let expectedEstimateIsMeasured = reconstructedProfile.effectiveMemoryCeilingIsMeasured
            && !placeholderKVQuant
        let expectedKeys = Set(ModelArchProfile.catalog.flatMap { model in
            [4, 8].map { weightBits in RowKey(modelID: model.id, weightBits: weightBits) }
        })
        guard rows.count == expectedKeys.count else {
            throw SizerMatrixError.invalidV2("rows must cover the complete catalog matrix")
        }
        var seenKeys = Set<RowKey>()
        try rows.forEach { row in
            try validateV2Row(
                row,
                profile: reconstructedProfile,
                kvQuant: decodedKVQuant,
                concurrency: concurrency,
                expectedKeys: expectedKeys,
                seenKeys: &seenKeys,
                expectedEstimateIsMeasured: expectedEstimateIsMeasured)
        }
        guard seenKeys == expectedKeys else {
            throw SizerMatrixError.invalidV2("rows do not match the complete catalog matrix")
        }
    }

    private struct RowKey: Hashable {
        let modelID: String
        let weightBits: Int
    }

    private func validateV2Row(
        _ row: Row,
        profile: SystemProfile,
        kvQuant: KVQuantTier,
        concurrency: Int,
        expectedKeys: Set<RowKey>,
        seenKeys: inout Set<RowKey>,
        expectedEstimateIsMeasured: Bool
    ) throws {
        guard !row.modelID.isEmpty else { throw SizerMatrixError.invalidV2("missing modelID") }
        guard row.weightBits > 0 else { throw SizerMatrixError.invalidV2("nonpositive weightBits") }
        guard row.weightsBytes >= 0 else { throw SizerMatrixError.invalidV2("negative weightsBytes") }
        guard row.kvBytesAtContext >= 0 else { throw SizerMatrixError.invalidV2("negative kvBytesAtContext") }
        guard row.transientPrefillBytes >= 0 else {
            throw SizerMatrixError.invalidV2("negative transientPrefillBytes")
        }
        guard row.totalPeakBytes >= 0 else { throw SizerMatrixError.invalidV2("negative totalPeakBytes") }
        guard row.maxContextThatFits >= 0 else { throw SizerMatrixError.invalidV2("negative maxContextThatFits") }
        guard row.requestedContext >= 0 else { throw SizerMatrixError.invalidV2("negative requestedContext") }
        guard let classification = CapacityColor(rawValue: row.classification) else {
            throw SizerMatrixError.invalidV2("unknown classification")
        }
        guard row.fits == (classification != .red) else {
            throw SizerMatrixError.invalidV2("fits disagrees with classification")
        }
        guard row.estimateIsMeasured == expectedEstimateIsMeasured else {
            throw SizerMatrixError.invalidV2("estimateIsMeasured disagrees with host/KV provenance")
        }
        let sum = row.weightsBytes.addingReportingOverflow(row.kvBytesAtContext)
        guard !sum.overflow else { throw SizerMatrixError.invalidV2("row byte sum overflow") }
        let requiredPeak = sum.partialValue.addingReportingOverflow(row.transientPrefillBytes)
        guard !requiredPeak.overflow else { throw SizerMatrixError.invalidV2("row byte sum overflow") }
        guard row.totalPeakBytes >= requiredPeak.partialValue else {
            throw SizerMatrixError.invalidV2("totalPeakBytes below required modeled terms")
        }
        let key = RowKey(modelID: row.modelID, weightBits: row.weightBits)
        guard expectedKeys.contains(key) else {
            throw SizerMatrixError.invalidV2("row is not a supported catalog/weight-bit identity")
        }
        guard seenKeys.insert(key).inserted else {
            throw SizerMatrixError.invalidV2("duplicate catalog/weight-bit row")
        }
        let expected = ModelSizer.report(
            box: profile,
            context: row.requestedContext,
            weightBitOptions: [row.weightBits],
            kvQuant: kvQuant,
            concurrency: concurrency)
            .first { $0.modelID == row.modelID }
        guard let expected else {
            throw SizerMatrixError.invalidV2("row cannot be recomputed from the catalog")
        }
        guard row.weightsBytes == expected.weightsBytes,
            row.kvBytesAtContext == expected.kvBytesAtContext,
            row.transientPrefillBytes == expected.transientPrefillBytes,
            row.totalPeakBytes == expected.totalPeakBytes,
            row.fits == expected.fits,
            row.maxContextThatFits == expected.maxContextThatFits,
            row.classification == expected.classification.rawValue,
            row.estimateIsMeasured == expected.estimateIsMeasured
        else {
            throw SizerMatrixError.invalidV2("row does not match the recomputed catalog fit")
        }
    }
}
