import Foundation
import HarnessCore
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import SpikeCore
import Tokenizers

/// Unsupported `RunConfig` features fail loudly — a "measurement" must never silently
/// measure something other than what was asked for.
enum SwiftEngineDriverError: Error, Equatable, CustomStringConvertible {
    case unsupportedConfig(String)
    var description: String {
        switch self {
        case .unsupportedConfig(let what): return "SwiftEngineDriver: unsupported config: \(what)"
        }
    }
}

/// Fully resolved cache/attention route admitted before actor execution. Keeping the attention
/// mode separate from `KVCacheKind` preserves the existing storage-tier identity while making
/// decoder reuse and evidence distinguish materialized from packed attention.
struct SwiftEngineCacheSelection: Equatable, Sendable {
    let kind: KVCacheKind
    let affineAttentionMode: AffineKVAttentionMode
    let kvarnAttentionMode: KVarNKVAttentionMode
    let kvarnStorageDType: KVarNKVScalarDType?
}

enum ExactPrefixRuntimeError:
    Error, Equatable, CustomStringConvertible
{
    case missingRuntimeIdentity
    case unsupportedRoute
    case unsupportedNativeDType(String)
    case invalidPrompt
    case snapshotByteCountOverflow
    case snapshotByteCountMismatch(expected: Int, actual: Int)
    case snapshotDTypeMismatch(
        expected: String,
        key: String,
        value: String)
    case snapshotGeometryMismatch(
        expected: String,
        actual: String)
    case logicalTokenCountMismatch(expected: Int, actual: Int)

    var description: String {
        switch self {
        case .missingRuntimeIdentity:
            "exact prefix cache is enabled without a loaded runtime identity"
        case .unsupportedRoute:
            "exact prefix cache requires scalar compiled dense-half full attention"
        case .unsupportedNativeDType(let dtype):
            "exact prefix cache requires float16 or bfloat16 dense state, got \(dtype)"
        case .invalidPrompt:
            "exact prefix cache requires positive Int32 prompt tokens"
        case .snapshotByteCountOverflow:
            "exact prefix snapshot byte count overflow"
        case .snapshotByteCountMismatch(let expected, let actual):
            "exact prefix snapshot bytes \(actual) != expected \(expected)"
        case .snapshotDTypeMismatch(
            let expected,
            let key,
            let value):
            "exact prefix snapshot dtype key=\(key) value=\(value) != expected \(expected)"
        case .snapshotGeometryMismatch(
            let expected,
            let actual):
            "exact prefix snapshot geometry \(actual) != expected \(expected)"
        case .logicalTokenCountMismatch(let expected, let actual):
            "exact prefix logical tokens \(actual) != expected \(expected)"
        }
    }
}

/// Stable, path-free source identity retained before the actor observes any live KV arrays.
///
/// Config dtype remains authenticated provenance, but it is not allowed to select the cache route:
/// some source-locked models declare bfloat16 while their loaded MLX attention path emits float16.
struct ExactPrefixRuntimeIdentitySource: Sendable {
    let admission: CompressedKVAttentionRuntimeAdmission
    let modelInstanceID: String

    init(
        admission: CompressedKVAttentionRuntimeAdmission,
        modelInstanceID: String = UUID().uuidString.lowercased()
    ) throws {
        let admission = try admission.validatedForEvidence()
        // Reuse the semantic key's strict path-free validation for the ephemeral instance and
        // authenticated source identities before unresolved state can enter the actor.
        _ = try ExactPrefixSemanticKey(
            isolationNamespaceSHA256: String(
                repeating: "0", count: 64),
            modelInstanceID: modelInstanceID,
            modelRevisionSHA256:
                admission.checkpointContentSHA256,
            tokenizerSHA256: admission.tokenizerSHA256,
            promptTemplateSHA256: String(
                repeating: "1", count: 64),
            toolsSHA256: String(repeating: "2", count: 64),
            kvRouteSHA256: String(repeating: "3", count: 64),
            positionSemanticsSHA256: String(
                repeating: "4", count: 64),
            architectureStateSHA256: String(
                repeating: "5", count: 64),
            drafterStateSHA256: String(repeating: "6", count: 64))
        self.admission = admission
        self.modelInstanceID = modelInstanceID
    }

    func resolve(
        snapshot: CompiledMLXDecoderSnapshot
    ) throws -> ExactPrefixRuntimeIdentity {
        let observed = try observedDenseDType(snapshot: snapshot)
        let identity = try ExactPrefixRuntimeIdentity(
            source: self, observedNativeDType: observed)
        let expectedBytes = try identity.snapshotBytes(
            tokenCount: snapshot.logicalTokenCount)
        guard snapshot.totalNBytes == expectedBytes else {
            throw ExactPrefixRuntimeError
                .snapshotByteCountMismatch(
                    expected: expectedBytes,
                    actual: snapshot.totalNBytes)
        }
        return identity
    }

    func observedDenseDType(
        snapshot: CompiledMLXDecoderSnapshot
    ) throws -> CompressedKVModelNativeDType {
        let profile = try snapshot.validatedDenseHalfProfile()
        let expectedGeometry =
            "rank=4,batch=1,layers=\(admission.layerCount),kvHeads=\(admission.kvHeadCount),headDim=\(admission.headDimension)"
        let actualGeometry =
            "rank=\(profile.rank),batch=\(profile.batchSize),layers=\(profile.layerCount),kvHeads=\(profile.kvHeadCount),headDim=\(profile.headDimension)"
        guard profile.rank == 4,
            profile.batchSize == 1,
            profile.layerCount == admission.layerCount,
            profile.kvHeadCount == admission.kvHeadCount,
            profile.headDimension == admission.headDimension
        else {
            throw ExactPrefixRuntimeError.snapshotGeometryMismatch(
                expected: expectedGeometry,
                actual: actualGeometry)
        }
        switch profile.dtype {
        case .float16:
            return .float16
        case .bfloat16:
            return .bfloat16
        default:
            throw ExactPrefixRuntimeError.unsupportedNativeDType(
                String(describing: profile.dtype))
        }
    }

    /// Exact detached snapshot bytes are independent of which supported half dtype is observed:
    /// both f16 and bf16 occupy two bytes per scalar. This lets the actor reject an undersized
    /// policy before allocating even the one-token live identity probe.
    func snapshotBytes(tokenCount: Int) throws -> Int {
        guard tokenCount > 0 else {
            throw ExactPrefixRuntimeError.invalidPrompt
        }
        func multiply(_ lhs: Int, _ rhs: Int) throws -> Int {
            let (value, overflow) =
                lhs.multipliedReportingOverflow(by: rhs)
            guard !overflow else {
                throw ExactPrefixRuntimeError
                    .snapshotByteCountOverflow
            }
            return value
        }
        func add(_ lhs: Int, _ rhs: Int) throws -> Int {
            let (value, overflow) =
                lhs.addingReportingOverflow(rhs)
            guard !overflow else {
                throw ExactPrefixRuntimeError
                    .snapshotByteCountOverflow
            }
            return value
        }

        var arrayBytes = try multiply(
            admission.layerCount, admission.kvHeadCount)
        arrayBytes = try multiply(arrayBytes, tokenCount)
        arrayBytes = try multiply(
            arrayBytes, admission.headDimension)
        arrayBytes = try multiply(
            arrayBytes, MemoryLayout<UInt16>.stride)
        arrayBytes = try multiply(arrayBytes, 2)
        let layerControlBytes = try multiply(
            admission.layerCount,
            MemoryLayout<CompiledKVCacheSnapshot>.stride)
        let controlBytes = try add(
            layerControlBytes, MemoryLayout<Int32>.stride)
        return try add(arrayBytes, controlBytes)
    }
}

/// Fully resolved actor-owned identity for one observed scalar dense-half route.
///
/// The semantic key binds both the independently authenticated source/config and the homogeneous
/// dtype emitted by every live K/V layer. Only this resolved form may touch the cache policy plane.
struct ExactPrefixRuntimeIdentity: Sendable {
    let source: ExactPrefixRuntimeIdentitySource
    let evidence: ExactPrefixDenseRuntimeIdentityEvidence
    let kvRouteSHA256: String
    let positionSemanticsSHA256: String
    let architectureStateSHA256: String
    let drafterStateSHA256: String
    let nativeDType: CompressedKVModelNativeDType

    var admission: CompressedKVAttentionRuntimeAdmission {
        source.admission
    }

    var modelInstanceID: String {
        source.modelInstanceID
    }

    init(
        source: ExactPrefixRuntimeIdentitySource,
        observedNativeDType nativeDType:
            CompressedKVModelNativeDType
    ) throws {
        guard nativeDType == .float16 || nativeDType == .bfloat16
        else {
            throw ExactPrefixRuntimeError.unsupportedNativeDType(
                nativeDType.rawValue)
        }

        let admission = source.admission
        let evidence =
            try ExactPrefixDenseRuntimeIdentityEvidence(
                observedDenseHalfDType: nativeDType)
        let kvRouteSHA256 = evidence.kvRouteSHA256
        let positionSemanticsSHA256 = sha256Hex(Data(
            """
            fast-mlx-exact-prefix-position-v1
            family=\(admission.family.rawValue)
            config=\(admission.modelConfigSHA256)
            max=\(admission.maxPositionEmbeddings)
            """.utf8))
        let architectureStateSHA256 = sha256Hex(Data(
            """
            fast-mlx-exact-prefix-architecture-v2
            model_type=\(admission.modelType)
            architecture=\(admission.architecture)
            layers=\(admission.layerCount)
            query_heads=\(admission.queryHeadCount)
            kv_heads=\(admission.kvHeadCount)
            head_dim=\(admission.headDimension)
            configured_dtype=\(admission.modelNativeDType?.rawValue ?? "unknown")
            observed_dense_half_dtype=\(nativeDType.rawValue)
            """.utf8))
        let drafterStateSHA256 = sha256Hex(Data(
            "fast-mlx-exact-prefix-drafter-v1\nnone\n".utf8))
        _ = try ExactPrefixSemanticKey(
            isolationNamespaceSHA256: String(
                repeating: "0", count: 64),
            modelInstanceID: source.modelInstanceID,
            modelRevisionSHA256:
                admission.checkpointContentSHA256,
            tokenizerSHA256: admission.tokenizerSHA256,
            promptTemplateSHA256: String(
                repeating: "1", count: 64),
            toolsSHA256: String(repeating: "2", count: 64),
            kvRouteSHA256: kvRouteSHA256,
            positionSemanticsSHA256: positionSemanticsSHA256,
            architectureStateSHA256: architectureStateSHA256,
            drafterStateSHA256: drafterStateSHA256)
        self.source = source
        self.evidence = evidence
        self.kvRouteSHA256 = kvRouteSHA256
        self.positionSemanticsSHA256 = positionSemanticsSHA256
        self.architectureStateSHA256 = architectureStateSHA256
        self.drafterStateSHA256 = drafterStateSHA256
        self.nativeDType = nativeDType
    }

    func semanticKey(
        request: ExactPrefixRequestContext
    ) throws -> ExactPrefixSemanticKey {
        try ExactPrefixSemanticKey(
            isolationNamespaceSHA256:
                request.isolationNamespaceSHA256,
            modelInstanceID: modelInstanceID,
            modelRevisionSHA256:
                admission.checkpointContentSHA256,
            tokenizerSHA256: admission.tokenizerSHA256,
            promptTemplateSHA256:
                request.promptTemplateSHA256,
            toolsSHA256: request.toolsSHA256,
            kvRouteSHA256: kvRouteSHA256,
            positionSemanticsSHA256: positionSemanticsSHA256,
            architectureStateSHA256: architectureStateSHA256,
            drafterStateSHA256: drafterStateSHA256)
    }

    func snapshotBytes(tokenCount: Int) throws -> Int {
        try source.snapshotBytes(tokenCount: tokenCount)
    }

    func validate(
        snapshot: CompiledMLXDecoderSnapshot
    ) throws {
        let actual = try source.observedDenseDType(
            snapshot: snapshot)
        guard actual == nativeDType else {
            throw ExactPrefixRuntimeError.snapshotDTypeMismatch(
                expected: nativeDType.rawValue,
                key: actual.rawValue,
                value: actual.rawValue)
        }
    }
}

/// Resolve and authenticate a runtime request without allocating MLX cache state. A frozen
/// KVTuner selection always requires a live checkpoint admission, even when its attention route
/// is the legacy default, because layer-count compatibility alone is not model identity.
func resolveSwiftEngineCacheSelection(
    config: RunConfig,
    compressedKVAttentionAdmission admission:
        CompressedKVAttentionRuntimeAdmission?
) throws -> SwiftEngineCacheSelection {
    if config.compressedKVAttention != nil {
        guard let admission,
            let expectedCheckpointContentSHA256 = config
                .compressedKVAttentionExpectedCheckpointContentSHA256
        else {
            throw CompressedKVAttentionRuntimeAdmissionError
                .invalidSourceIdentity
        }
        try admission.validateCheckpointContentIdentity(
            expectedCheckpointContentSHA256)
    } else if config
        .compressedKVAttentionExpectedCheckpointContentSHA256 != nil
    {
        throw CompressedKVAttentionRuntimeAdmissionError
            .invalidSourceIdentity
    }
    let affineMode: AffineKVAttentionMode
    let kvarnMode: KVarNKVAttentionMode
    switch config.compressedKVAttention {
    case nil, .materialize:
        affineMode = .materialize
        kvarnMode = .materialize
    case .splitAffineQuantizedMM:
        affineMode = .splitQuantizedMM
        kvarnMode = .materialize
    case .splitKVarNQuantizedMM:
        affineMode = .materialize
        kvarnMode = .splitQuantizedMM
    }

    if let selection = config.kvtunerSelection {
        if config.compressedKVAttention == .splitKVarNQuantizedMM {
            throw SwiftEngineDriverError.unsupportedConfig(
                "split KVarN attention requires a KVarN KV tier")
        }
        guard config.kvQuant == selection.cellID else {
            throw SwiftEngineDriverError.unsupportedConfig(
                "KVTuner selection cell \(selection.cellID) != kvQuant=\(config.kvQuant ?? "nil")")
        }
        guard let admission else {
            throw CompressedKVAttentionRuntimeAdmissionError
                .invalidSourceIdentity
        }
        try admission.validateScheduleIdentity(
            modelConfigHash: selection.modelConfigHash,
            modelConfigSHA256: selection.modelConfigSHA256,
            checkpointManifestHash: selection.checkpointManifestHash,
            checkpointContentSHA256:
                selection.checkpointContentSHA256,
            tokenizerSHA256: selection.tokenizerSHA256,
            layerCount: selection.layers.count,
            groupSize: selection.groupSize)
        return SwiftEngineCacheSelection(
            kind: .kvtuner(selection),
            affineAttentionMode: affineMode,
            kvarnAttentionMode: .materialize,
            kvarnStorageDType: nil)
    }
    if config.kvQuant?.hasPrefix("kvtuner-") == true {
        throw SwiftEngineDriverError.unsupportedConfig(
            "kvQuant=\(config.kvQuant!) requires an authenticated KVTuner selection")
    }
    guard let kind = KVCacheKind(kvQuant: config.kvQuant) else {
        throw SwiftEngineDriverError.unsupportedConfig(
            "kvQuant=\(config.kvQuant ?? "fp16") (unknown tier)")
    }

    let kvarnStorageDType: KVarNKVScalarDType?
    if case .kvarn = kind, let admission {
        switch admission.modelNativeDType {
        case .float16:
            kvarnStorageDType = .float16
        case .bfloat16:
            kvarnStorageDType = .bfloat16
        case .float32, nil:
            throw SwiftEngineDriverError.unsupportedConfig(
                "KVarN requires an authenticated fp16 or bfloat16 model-native storage dtype")
        }
    } else {
        kvarnStorageDType = nil
    }

    if let request = config.compressedKVAttention {
        guard let admission else {
            throw CompressedKVAttentionRuntimeAdmissionError
                .invalidSourceIdentity
        }
        switch request {
        case .materialize:
            switch kind {
            case .affine(let tier):
                try admission.validateAffineGeometry(
                    keyGroupSize: tier.configuration.keyGroupSize,
                    valueGroupSize: tier.configuration.valueGroupSize)
            case .kvarn(let cell):
                _ = try KVarNMLXConfiguration(
                    headDimension: admission.headDimension,
                    groupSize: cell.tier.groupSize,
                    keyBits: cell.tier.keyBits,
                    valueBits: cell.tier.valueBits,
                    iterations: cell.iterations)
            case .fp16, .turboQuant, .kvtuner, .kvtunerCandidate:
                throw SwiftEngineDriverError.unsupportedConfig(
                    "compressedKVAttention requires an affine- or KVarN-backed KV tier")
            }
        case .splitAffineQuantizedMM:
            guard case .affine(let tier) = kind else {
                throw SwiftEngineDriverError.unsupportedConfig(
                    "split affine attention requires an affine-backed KV tier")
            }
            try admission.validateAffineGeometry(
                keyGroupSize: tier.configuration.keyGroupSize,
                valueGroupSize: tier.configuration.valueGroupSize)
        case .splitKVarNQuantizedMM:
            guard case .kvarn(let cell) = kind else {
                throw SwiftEngineDriverError.unsupportedConfig(
                    "split KVarN attention requires a KVarN KV tier")
            }
            _ = try KVarNMLXConfiguration(
                headDimension: admission.headDimension,
                groupSize: cell.tier.groupSize,
                keyBits: cell.tier.keyBits,
                valueBits: cell.tier.valueBits,
                iterations: cell.iterations)
        }
    }
    return SwiftEngineCacheSelection(
        kind: kind,
        affineAttentionMode: affineMode,
        kvarnAttentionMode: kvarnMode,
        kvarnStorageDType: kvarnStorageDType)
}

/// Exact result of one restricted-choice scoring forward. Only actor-safe CPU values cross the
/// engine boundary; the MLX arrays and concrete cache objects remain actor-confined.
struct TaskChoiceLogitsResult: Sendable {
    let logits: [Float]
    let engagement: EngagementCounters
}

/// Actor-safe output of the private KVTuner candidate generation path. The generated token IDs
/// are paired with telemetry captured from the exact decoder/cache instance before its MLX arrays
/// leave actor isolation.
struct KVTunerCandidateRunResult: Sendable {
    let promptOrdinal: Int
    let promptTokenIDsSHA256: String
    let tokens: [Int]
    let finishReason: KVTunerCandidateFinishReason
    let telemetry: KVTunerCandidateKVCacheTelemetry
}

/// Integer-only dtype receipt carried through existing engagement artifacts. Source markers are
/// presence bits because legitimate model layers can enter the same authenticated storage path at
/// different dtypes; storage remains one-hot. Missing/partial evidence stays distinguishable from
/// every valid receipt without changing historical CSV columns.
private func kvarnIngressEngagement(
    _ telemetry: KVarNKVCacheTelemetry,
    prefix: String = "kvarn"
) -> [String: Int] {
    let dtypes: [KVarNKVScalarDType] = [
        .float16, .bfloat16, .float32,
    ]
    var counts: [String: Int] = [:]
    func record(
        _ role: String,
        _ selected: Set<KVarNKVScalarDType>
    ) {
        for dtype in dtypes {
            counts["\(prefix)_\(role)_\(dtype.rawValue)"] =
                selected.contains(dtype) ? 1 : 0
        }
    }
    record("source_key", telemetry.sourceKeyDTypes)
    record("source_value", telemetry.sourceValueDTypes)
    record("storage_key", [telemetry.storageKeyDType])
    record("storage_value", [telemetry.storageValueDType])
    counts["\(prefix)_ingress_normalized"] =
        telemetry.ingressNormalizationApplied ? 1 : 0
    counts["\(prefix)_normalization_workspace_bytes"] =
        telemetry.normalizationWorkspaceBytes
    return counts
}

private func exactPrefixEngagement(
    _ metrics: RequestStartMetrics
) -> [String: Int] {
    func nanoseconds(_ seconds: Double) -> Int {
        let scaled = seconds * 1_000_000_000
        guard scaled < Double(Int.max) else {
            return Int.max
        }
        return max(0, Int(scaled.rounded()))
    }
    var counts = [
        "prefix_cache_request_start": 1,
        "prefix_cache_read_tokens":
            metrics.cacheReadTokenCount,
        "prefix_cache_physical_prefill_tokens":
            metrics.physicalPrefillTokenCount,
        "prefix_cache_retained_bytes":
            metrics.retainedBytes,
        "prefix_cache_entries": metrics.entryCount,
        "prefix_cache_evictions": metrics.evictionCount,
        "prefix_cache_template_hit":
            metrics.templateTokenCacheHit ? 1 : 0,
        "prefix_cache_template_ns":
            nanoseconds(metrics.templateSeconds),
        "prefix_cache_tokenize_ns":
            nanoseconds(metrics.tokenizeSeconds),
        "prefix_cache_lookup_ns":
            nanoseconds(metrics.lookupSeconds),
        "prefix_cache_restore_ns":
            nanoseconds(metrics.restoreSeconds),
        "prefix_cache_prefill_ns":
            nanoseconds(metrics.prefillSeconds),
    ]
    for outcome in [
        PrefixCacheRequestOutcome.disabled,
        .miss, .exactHit, .partialHit, .rejected,
    ] {
        counts[
            "prefix_cache_outcome_\(outcome.rawValue)"
        ] = metrics.prefixCacheOutcome == outcome ? 1 : 0
    }
    for reason in ExactPrefixCommitSkipReason.allCases {
        let suffix = reason.rawValue.replacingOccurrences(
            of: "-", with: "_")
        counts[
            "prefix_cache_rejection_\(suffix)"
        ] = metrics.prefixCacheRejectionReason == reason ? 1 : 0
    }
    if let eagerWarmupSeconds = metrics.eagerWarmupSeconds {
        counts["prefix_cache_eager_warmup_ns"] =
            nanoseconds(eagerWarmupSeconds)
    }
    return counts
}

/// Single-owner actor over the spike's compiled decode core. Owns BOTH the model and the
/// `CompiledMLXDecoder` (one isolation region) because `logprobs` needs direct forward access
/// to full-vocab logits, which the token-only `Decoder` protocol deliberately doesn't expose.
///
/// The generate path preserves the spike's compiled-step + no-sync-readback design by
/// construction: it only calls `CompiledMLXDecoder.prefill`/`.step`, where that pattern lives
/// (the next step's compiled forward is asyncEval'd before the current token's `.item()`).
actor HarnessEngineActor {
    private struct DecoderKey: Hashable {
        let kind: KVCacheKind
        let affineAttentionMode: AffineKVAttentionMode
        let kvarnAttentionMode: KVarNKVAttentionMode
        let kvarnStorageDType: KVarNKVScalarDType?
    }

    private struct KVarNScoringTelemetryKey: Hashable {
        let cell: KVarNKVRuntimeCell
        let attentionMode: KVarNKVAttentionMode
        let storageDType: KVarNKVScalarDType?
    }

    private let model: any LanguageModel
    /// Captured from the same model directory and live tokenizer used to construct `model`.
    /// Candidate policies must reconcile against this independent identity inside the actor.
    private let kvtunerRuntimeIdentity: KVTunerCandidateRuntimeIdentity?
    private let exactPrefixCacheConfiguration:
        ExactPrefixCacheConfiguration
    private let exactPrefixRuntimeIdentitySource:
        ExactPrefixRuntimeIdentitySource?
    private var exactPrefixRuntimeIdentity:
        ExactPrefixRuntimeIdentity?
    private var exactPrefixRuntimeIdentityRejectionReason:
        ExactPrefixCommitSkipReason?
    private var exactPrefixRuntimeIdentityRejectionDetail:
        String?
    /// Both the pure index and its MLX-backed snapshot payloads stay inside this actor.
    private var exactPrefixCache:
        ExactPrefixCache<CompiledMLXDecoderSnapshot>
    private var exactPrefixWarmupSeconds: Double?
    /// One decoder per KV-cache kind, built lazily and kept alive so each kind's compiled
    /// step function survives across runs (an fp16 baseline and a TurboQuant candidate can
    /// alternate within one verify invocation without retracing either).
    private var decoders: [DecoderKey: CompiledMLXDecoder] = [:]
    /// Largest affine scoring allocation observed in this actor. Scoring caches are ephemeral,
    /// so their scalar geometry/byte evidence must be retained before the arrays are released.
    private var maximumAffineScoringTelemetry: AffineKVCacheTelemetry?
    /// Heterogeneous affine policies are keyed by the immutable exact-byte selection, so two
    /// same-cell artifacts cannot share retained scoring evidence.
    private var maximumKVTunerScoringTelemetry: [
        KVTunerRuntimeSelection: KVTunerKVCacheTelemetry
    ] = [:]
    /// KVarN cells share packed geometry but not codec work or attention workspace. Key retained
    /// scalar evidence by both runtime cell and route so a larger materialized pass cannot be
    /// returned as evidence for a later direct pass (or vice versa).
    private var maximumKVarNScoringTelemetry: [
        KVarNScoringTelemetryKey: KVarNKVCacheTelemetry
    ] = [:]
    /// Exact token widths submitted during the latest scoring-prefill operation. This contains
    /// only CPU scalars and is retained so qualification tests can prove every scoring entry
    /// point uses the bounded split-attention route without exposing actor-owned MLX state.
    private var latestScoringPrefillTokenCounts: [Int] = []

    init(
        model: sending any LanguageModel,
        kvtunerRuntimeIdentity: KVTunerCandidateRuntimeIdentity? = nil,
        exactPrefixCacheConfiguration:
            ExactPrefixCacheConfiguration = .disabled,
        exactPrefixRuntimeIdentitySource:
            ExactPrefixRuntimeIdentitySource? = nil
    ) {
        self.model = model
        self.kvtunerRuntimeIdentity = kvtunerRuntimeIdentity
        self.exactPrefixCacheConfiguration =
            exactPrefixCacheConfiguration
        self.exactPrefixRuntimeIdentitySource =
            exactPrefixRuntimeIdentitySource
        exactPrefixRuntimeIdentity = nil
        exactPrefixRuntimeIdentityRejectionReason = nil
        exactPrefixRuntimeIdentityRejectionDetail = nil
        exactPrefixCache = ExactPrefixCache(
            policy: exactPrefixCacheConfiguration.policy)
    }

    func exactPrefixCacheSnapshot() -> ExactPrefixCacheSnapshot {
        exactPrefixCache.snapshot
    }

    func exactPrefixRuntimeIdentityEvidence()
        -> ExactPrefixDenseRuntimeIdentityEvidence?
    {
        exactPrefixRuntimeIdentity?.evidence
    }

    /// Resolve the live scalar dense-half route with a one-token detached snapshot.
    ///
    /// The probe is admitted against the cache byte policy before allocation, never touches the
    /// prefix index, and records a terminal actor-local rejection so unsupported models serve
    /// cold without retrying or mutating the cache plane.
    func prepareExactPrefixRuntimeIdentity() {
        guard exactPrefixCacheConfiguration.policy.isEnabled,
            exactPrefixRuntimeIdentitySource != nil,
            exactPrefixRuntimeIdentity == nil,
            exactPrefixRuntimeIdentityRejectionReason == nil
        else {
            return
        }
        let key = DecoderKey(
            kind: .fp16,
            affineAttentionMode: .materialize,
            kvarnAttentionMode: .materialize,
            kvarnStorageDType: nil)
        if decoders[key] == nil {
            decoders[key] = CompiledMLXDecoder(
                model: model, kvCache: .fp16)
        }
        var decoder = decoders[key]!
        resolveExactPrefixRuntimeIdentityIfNeeded(
            decoder: &decoder)
        decoder.reset()
        decoders[key] = decoder
    }

    private func resolveExactPrefixRuntimeIdentityIfNeeded(
        decoder: inout CompiledMLXDecoder
    ) {
        guard exactPrefixRuntimeIdentity == nil,
            exactPrefixRuntimeIdentityRejectionReason == nil,
            let exactPrefixRuntimeIdentitySource
        else {
            return
        }
        do {
            let probeBytes =
                try exactPrefixRuntimeIdentitySource
                    .snapshotBytes(tokenCount: 1)
            guard probeBytes <= exactPrefixCacheConfiguration
                .policy.maxRetainedBytes
            else {
                exactPrefixRuntimeIdentityRejectionReason =
                    .snapshotExceedsBudget
                return
            }
            decoder.reset()
            let staged =
                try decoder.prefillCapturingPromptSnapshot([1])
            exactPrefixRuntimeIdentity =
                try exactPrefixRuntimeIdentitySource.resolve(
                    snapshot: staged.snapshot)
        } catch {
            switch error {
            case ExactPrefixRuntimeError
                .snapshotByteCountMismatch(_, _),
                ExactPrefixRuntimeError
                    .snapshotByteCountOverflow,
                ExactPrefixRuntimeError
                    .snapshotDTypeMismatch(_, _, _),
                ExactPrefixRuntimeError
                    .snapshotGeometryMismatch(_, _):
                exactPrefixRuntimeIdentityRejectionReason =
                    .snapshotEvidenceMismatch
                exactPrefixRuntimeIdentityRejectionDetail =
                    String(describing: error)
            default:
                exactPrefixRuntimeIdentityRejectionReason =
                    .snapshotCaptureFailed
            }
        }
        decoder.reset()
    }

    /// Optional load-time warmup for the exact scalar dense-half route. It is idempotent, never
    /// touches the prefix index, and resets the decoder after exercising one- and eight-token
    /// prefill plus compiled single-token decode shapes.
    func performExactPrefixWarmup() throws -> Double {
        guard exactPrefixCacheConfiguration.policy.isEnabled,
            exactPrefixRuntimeIdentitySource != nil
        else {
            throw ExactPrefixRuntimeError.missingRuntimeIdentity
        }
        if let exactPrefixWarmupSeconds {
            return exactPrefixWarmupSeconds
        }
        let key = DecoderKey(
            kind: .fp16,
            affineAttentionMode: .materialize,
            kvarnAttentionMode: .materialize,
            kvarnStorageDType: nil)
        if decoders[key] == nil {
            decoders[key] = CompiledMLXDecoder(
                model: model, kvCache: .fp16)
        }
        var decoder = decoders[key]!
        resolveExactPrefixRuntimeIdentityIfNeeded(
            decoder: &decoder)
        let startedAt = ProcessInfo.processInfo.systemUptime
        for prompt in [[1], Array(repeating: 1, count: 8)] {
            decoder.reset()
            let first = decoder.prefill(prompt)
            _ = decoder.step(last: first)
        }
        decoder.reset()
        let duration =
            ProcessInfo.processInfo.systemUptime - startedAt
        decoders[key] = decoder
        exactPrefixWarmupSeconds = duration
        return duration
    }

    /// Greedy decode via the compiled core. The returned ids INCLUDE a terminal eos if one is
    /// produced (mirroring scripts/harness_reference.py exactly, so token streams diff cleanly).
    /// Quantized-cache engagement is read AFTER timing so its synchronization cannot skew the
    /// benchmark. Affine and KVarN return Sendable scalar snapshots; TurboQuant retains its
    /// legacy token marker until that format's evidence schema is generalized.
    func generate(
        prompt: [Int], maxTokens: Int, eos: Int,
        kvCache kind: KVCacheKind,
        affineAttentionMode: AffineKVAttentionMode,
        kvarnAttentionMode: KVarNKVAttentionMode = .materialize,
        kvarnStorageDType: KVarNKVScalarDType? = nil,
        exactPrefixRequest: ExactPrefixRequestContext? = nil
    )
        throws -> (
            tokens: [Int], submitTime: Double, tokenTimes: [Double],
            prefillDurationSeconds: Double?,
            turboQuantTokens: Int?, affineTelemetry: AffineKVCacheTelemetry?,
            kvtunerTelemetry: KVTunerKVCacheTelemetry?,
            kvarnTelemetry: KVarNKVCacheTelemetry?,
            requestStartMetrics: RequestStartMetrics?
        )
    {
        if case .kvtunerCandidate = kind {
            preconditionFailure(
                "KVTuner candidates require the private qualification path")
        }
        let decoderKey = DecoderKey(
            kind: kind,
            affineAttentionMode: affineAttentionMode,
            kvarnAttentionMode: kvarnAttentionMode,
            kvarnStorageDType: kvarnStorageDType)
        if decoders[decoderKey] == nil {
            decoders[decoderKey] = CompiledMLXDecoder(
                model: model,
                kvCache: kind,
                affineAttentionMode: affineAttentionMode,
                kvarnAttentionMode: kvarnAttentionMode,
                kvarnStorageDType: kvarnStorageDType)
        }
        var decoder = decoders[decoderKey]!
        defer { decoders[decoderKey] = decoder }
        decoder.reset() // in-place KV reset: compiled graph stays valid across runs
        let submitTime = Date().timeIntervalSinceReferenceDate
        guard maxTokens > 0 else {
            return (
                [], submitTime, [], nil, nil, nil, nil, nil,
                nil)
        }
        guard !prompt.isEmpty,
            prompt.allSatisfy({
                $0 >= 0 && Int32(exactly: $0) != nil
            })
        else {
            throw ExactPrefixRuntimeError.invalidPrompt
        }
        if exactPrefixRequest != nil {
            guard kind == .fp16,
                affineAttentionMode == .materialize,
                kvarnAttentionMode == .materialize,
                kvarnStorageDType == nil
            else {
                throw ExactPrefixRuntimeError.unsupportedRoute
            }
        }

        var promptReservation: ExactPrefixReservation?
        var stagedPromptSnapshot:
            CompiledMLXDecoderSnapshot?
        do {
            try Task.checkCancellation()

            let cacheEnabled =
                exactPrefixRequest != nil
                && exactPrefixCacheConfiguration.policy.isEnabled
            var prefixCacheRejectionReason:
                ExactPrefixCommitSkipReason?
            var prefixCacheRejectionDetail: String?
            var semanticKey: ExactPrefixSemanticKey?
            if cacheEnabled {
                guard exactPrefixRuntimeIdentitySource != nil,
                    let exactPrefixRequest
                else {
                    throw ExactPrefixRuntimeError
                        .missingRuntimeIdentity
                }
                resolveExactPrefixRuntimeIdentityIfNeeded(
                    decoder: &decoder)
                prefixCacheRejectionReason =
                    exactPrefixRuntimeIdentityRejectionReason
                prefixCacheRejectionDetail =
                    exactPrefixRuntimeIdentityRejectionDetail
                if prefixCacheRejectionReason == nil {
                    semanticKey = try exactPrefixRuntimeIdentity?
                        .semanticKey(
                            request: exactPrefixRequest)
                } else {
                    semanticKey = nil
                }
            } else {
                semanticKey = nil
            }

            let lookupStartedAt =
                ProcessInfo.processInfo.systemUptime
            var lookupPerformed = false
            let hit: ExactPrefixCacheHit<
                CompiledMLXDecoderSnapshot>?
            if let semanticKey {
                lookupPerformed = true
                hit = try exactPrefixCache.lookup(
                    key: semanticKey,
                    promptTokens: prompt)
            } else {
                hit = nil
            }
            var primaryEntryID = hit?.entryID
            let lookupSeconds =
                ProcessInfo.processInfo.systemUptime
                    - lookupStartedAt

            if hit == nil, let semanticKey,
                let exactPrefixRuntimeIdentity
            {
                do {
                    let expectedBytes =
                        try exactPrefixRuntimeIdentity
                            .snapshotBytes(
                                tokenCount: prompt.count)
                    let decision = try exactPrefixCache.reserve(
                        key: semanticKey,
                        tokens: prompt,
                        snapshotBytes: expectedBytes)
                    promptReservation = decision.reservation
                    if decision.reservation == nil,
                        decision.skipReason != .disabled
                    {
                        prefixCacheRejectionReason =
                            decision.skipReason
                    }
                } catch {
                    prefixCacheRejectionReason =
                        .snapshotEvidenceMismatch
                    if let runtimeError =
                        error as? ExactPrefixRuntimeError
                    {
                        prefixCacheRejectionDetail =
                            runtimeError.description
                    }
                }
            }

            var restoreSeconds = 0.0
            var physicalPrefillDuration =
                Double.leastNonzeroMagnitude
            let physicalPrefillTokenCount: Int
            var prefixCacheOutcome: PrefixCacheRequestOutcome
            let prefillStartedAt =
                ProcessInfo.processInfo.systemUptime
            var tok: Int
            if let hit {
                let tail = Array(
                    prompt.dropFirst(hit.prefixTokenCount))
                let restoreAttemptStartedAt =
                    ProcessInfo.processInfo.systemUptime
                do {
                    guard let exactPrefixRuntimeIdentity else {
                        throw ExactPrefixRuntimeError
                            .missingRuntimeIdentity
                    }
                    try exactPrefixRuntimeIdentity.validate(
                        snapshot: hit.state)
                    let measured =
                        try decoder
                            .prefillRestoredPrefixMeasuring(
                                hit.state, tailTokens: tail)
                    tok = measured.firstToken
                    restoreSeconds =
                        measured.restoreDurationSeconds
                    physicalPrefillDuration = max(
                        measured.tailPrefillDurationSeconds,
                        Double.leastNonzeroMagnitude)
                    physicalPrefillTokenCount = tail.count
                    prefixCacheOutcome =
                        tail.isEmpty ? .exactHit : .partialHit
                } catch {
                    _ = exactPrefixCache.invalidate(
                        entryID: hit.entryID)
                    restoreSeconds =
                        ProcessInfo.processInfo.systemUptime
                            - restoreAttemptStartedAt
                    decoder.reset()
                    let coldPrefillStartedAt =
                        ProcessInfo.processInfo.systemUptime
                    tok = decoder.prefill(prompt)
                    physicalPrefillDuration = max(
                        ProcessInfo.processInfo.systemUptime
                            - coldPrefillStartedAt,
                        Double.leastNonzeroMagnitude)
                    physicalPrefillTokenCount = prompt.count
                    prefixCacheOutcome = .rejected
                    prefixCacheRejectionReason =
                        .snapshotRestoreFailed
                }
            } else if promptReservation != nil {
                let coldPrefillStartedAt =
                    ProcessInfo.processInfo.systemUptime
                do {
                    let staged =
                        try decoder
                            .prefillCapturingPromptSnapshot(
                                prompt)
                    guard let exactPrefixRuntimeIdentity else {
                        throw ExactPrefixRuntimeError
                            .missingRuntimeIdentity
                    }
                    try exactPrefixRuntimeIdentity.validate(
                        snapshot: staged.snapshot)
                    let expectedBytes =
                        try exactPrefixRuntimeIdentity
                            .snapshotBytes(
                                tokenCount: prompt.count)
                    guard staged.snapshot.totalNBytes
                        == expectedBytes
                    else {
                        throw ExactPrefixRuntimeError
                            .snapshotByteCountMismatch(
                                expected: expectedBytes,
                                actual:
                                    staged.snapshot.totalNBytes)
                    }
                    tok = staged.firstToken
                    physicalPrefillDuration = max(
                        ProcessInfo.processInfo.systemUptime
                            - coldPrefillStartedAt,
                        Double.leastNonzeroMagnitude)
                    stagedPromptSnapshot = staged.snapshot
                    physicalPrefillTokenCount = prompt.count
                    prefixCacheOutcome = .miss
                } catch {
                    if let promptReservation {
                        try? exactPrefixCache.rollback(
                            promptReservation)
                    }
                    promptReservation = nil
                    stagedPromptSnapshot = nil
                    decoder.reset()
                    let fallbackPrefillStartedAt =
                        ProcessInfo.processInfo.systemUptime
                    tok = decoder.prefill(prompt)
                    physicalPrefillDuration = max(
                        ProcessInfo.processInfo.systemUptime
                            - fallbackPrefillStartedAt,
                        Double.leastNonzeroMagnitude)
                    physicalPrefillTokenCount = prompt.count
                    prefixCacheOutcome = .rejected
                    switch error {
                    case ExactPrefixRuntimeError
                        .snapshotByteCountMismatch(_, _),
                        ExactPrefixRuntimeError
                            .snapshotByteCountOverflow,
                        ExactPrefixRuntimeError
                            .snapshotDTypeMismatch(_, _, _),
                        ExactPrefixRuntimeError
                            .snapshotGeometryMismatch(_, _):
                        prefixCacheRejectionReason =
                            .snapshotEvidenceMismatch
                        prefixCacheRejectionDetail =
                            String(describing: error)
                    default:
                        prefixCacheRejectionReason =
                            .snapshotCaptureFailed
                    }
                }
            } else {
                let coldPrefillStartedAt =
                    ProcessInfo.processInfo.systemUptime
                tok = decoder.prefill(prompt)
                physicalPrefillDuration = max(
                    ProcessInfo.processInfo.systemUptime
                        - coldPrefillStartedAt,
                    Double.leastNonzeroMagnitude)
                physicalPrefillTokenCount = prompt.count
                if exactPrefixRequest != nil,
                    !exactPrefixCacheConfiguration.policy
                        .isEnabled
                {
                    prefixCacheOutcome = .disabled
                } else if prefixCacheRejectionReason != nil {
                    prefixCacheOutcome = .rejected
                } else {
                    prefixCacheOutcome = .miss
                }
            }
            let fullPrefillDuration =
                ProcessInfo.processInfo.systemUptime
                    - prefillStartedAt

            var tokens: [Int] = [tok]
            var tokenTimes: [Double] = [
                Date().timeIntervalSinceReferenceDate,
            ]
            while tokens.count < maxTokens && tok != eos {
                tok = decoder.step(last: tok)
                tokens.append(tok)
                tokenTimes.append(
                    Date().timeIntervalSinceReferenceDate)
            }
            try Task.checkCancellation()

            let visibleTokenCount = tokens.reduce(into: 0) {
                if $1 != eos { $0 += 1 }
            }
            let disposition = ExactPrefixCommitDisposition
                .successfulText(
                    generatedTokenCount: tokens.count,
                    visibleTokenCount: visibleTokenCount)
            if let reservation = promptReservation,
                let stagedPromptSnapshot
            {
                do {
                    let decision = try exactPrefixCache.commit(
                        reservation,
                        state: stagedPromptSnapshot,
                        actualSnapshotBytes:
                            stagedPromptSnapshot.totalNBytes,
                        disposition: disposition)
                    if let skipReason = decision.skipReason {
                        prefixCacheOutcome = .rejected
                        prefixCacheRejectionReason = skipReason
                    } else {
                        primaryEntryID = decision.entryID
                    }
                } catch {
                    try? exactPrefixCache.rollback(
                        reservation)
                    prefixCacheRejectionReason =
                        .snapshotEvidenceMismatch
                    prefixCacheOutcome = .rejected
                }
                promptReservation = nil
            }

            if prefixCacheOutcome != .rejected,
                visibleTokenCount > 0, let semanticKey,
                let exactPrefixRuntimeIdentity
            {
                var finalReservation:
                    ExactPrefixReservation?
                do {
                    let finalTokens = prompt + tokens
                    let expectedBytes =
                        try exactPrefixRuntimeIdentity
                            .snapshotBytes(
                                tokenCount: finalTokens.count)
                        let finalDecision = try exactPrefixCache
                            .reserve(
                                key: semanticKey,
                                tokens: finalTokens,
                                snapshotBytes: expectedBytes,
                                protectingEntryIDs:
                                    primaryEntryID.map {
                                        Set([$0])
                                    } ?? [])
                    if let reserved =
                        finalDecision.reservation
                    {
                        finalReservation =
                            reserved
                        let finalSnapshot =
                            try decoder
                                .captureContinuationSnapshot()
                        try exactPrefixRuntimeIdentity.validate(
                            snapshot: finalSnapshot)
                        guard finalSnapshot.logicalTokenCount
                            == finalTokens.count
                        else {
                            throw ExactPrefixRuntimeError
                                .logicalTokenCountMismatch(
                                    expected: finalTokens.count,
                                    actual: finalSnapshot
                                        .logicalTokenCount)
                        }
                        guard finalSnapshot.totalNBytes
                            == expectedBytes
                        else {
                            throw ExactPrefixRuntimeError
                                .snapshotByteCountMismatch(
                                    expected: expectedBytes,
                                    actual:
                                        finalSnapshot.totalNBytes)
                        }
                        _ = try exactPrefixCache.commit(
                            reserved,
                            state: finalSnapshot,
                            actualSnapshotBytes:
                                finalSnapshot.totalNBytes,
                            disposition: disposition)
                        finalReservation = nil
                    }
                } catch {
                    if let finalReservation {
                        try? exactPrefixCache.rollback(
                            finalReservation)
                    }
                    // The visible request has already succeeded. Final-context caching is a
                    // best-effort sidecar; an incomplete snapshot must not retroactively fail
                    // generation or invalidate the independently committed prompt snapshot.
                }
            }

            let cacheSnapshot = exactPrefixCache.snapshot
            let requestStartMetrics:
                RequestStartMetrics?
            if exactPrefixRequest != nil {
                let cacheReadTokenCount =
                    prompt.count - physicalPrefillTokenCount
                requestStartMetrics = try RequestStartMetrics(
                    promptTokenCount: prompt.count,
                    cacheReadTokenCount:
                        cacheReadTokenCount,
                    physicalPrefillTokenCount:
                        physicalPrefillTokenCount,
                    prefixCacheOutcome: prefixCacheOutcome,
                    prefixCacheRejectionReason:
                        prefixCacheRejectionReason,
                    prefixCacheRejectionDetail:
                        prefixCacheRejectionDetail,
                    templateTokenCacheHit: false,
                    templateSeconds: 0,
                    tokenizeSeconds: 0,
                    lookupSeconds:
                        lookupPerformed ? lookupSeconds : 0,
                    restoreSeconds: restoreSeconds,
                    prefillSeconds:
                        physicalPrefillDuration,
                    retainedBytes:
                        cacheSnapshot.retainedBytes,
                    entryCount: cacheSnapshot.entryCount,
                    evictionCount:
                        cacheSnapshot.evictionCount,
                    runtimeIdentity:
                        exactPrefixRuntimeIdentity?.evidence,
                    eagerWarmupSeconds:
                        exactPrefixWarmupSeconds)
            } else {
                requestStartMetrics = nil
            }

            let turboQuantTokens =
                decoder.turboQuantCachedTokens()
            let affineTelemetry = decoder.affineKVTelemetry()
            let kvtunerTelemetry = decoder.kvtunerKVTelemetry()
            let kvarnTelemetry = decoder.kvarnKVTelemetry()
            if case .turboQuant = kind {
                precondition(
                    turboQuantTokens != nil,
                    "TurboQuant tier requested but the quantized cache did not engage")
            }
            if case .affine = kind {
                precondition(
                    affineTelemetry != nil,
                    "affine tier requested but the affine cache did not engage")
            }
            if case .kvtuner(let selection) = kind {
                precondition(
                    kvtunerTelemetry?.artifactSHA256
                        == selection.artifactSHA256,
                    "KVTuner tier requested but matching schedule telemetry did not engage")
            }
            if case .kvarn(let cell) = kind {
                let expectedExecutionMode = kind.executionMode(
                    requestingCompilation: true,
                    kvarnAttentionMode: kvarnAttentionMode)
                precondition(
                    kvarnTelemetry?.tier == cell.tier
                        && kvarnTelemetry?.iterations
                            == cell.iterations
                        && kvarnTelemetry?.executionMode
                            == expectedExecutionMode
                        && kvarnTelemetry?.attentionOperation
                            == expectedKVarNAttentionOperation(
                                kvarnAttentionMode)
                        && (kvarnStorageDType == nil
                            || kvarnTelemetry?.storageKeyDType
                                == kvarnStorageDType)
                        && (kvarnStorageDType == nil
                            || kvarnTelemetry?.storageValueDType
                                == kvarnStorageDType),
                    "KVarN tier requested but matching KVarN telemetry did not engage")
            }
            return (
                tokens, submitTime, tokenTimes,
                fullPrefillDuration,
                turboQuantTokens, affineTelemetry,
                kvtunerTelemetry, kvarnTelemetry,
                requestStartMetrics)
        } catch {
            if let promptReservation {
                try? exactPrefixCache.rollback(
                    promptReservation)
            }
            decoder.reset()
            throw error
        }
    }

    /// Greedy candidate generation for exhaustive KVTuner qualification only. This has no
    /// `RunConfig` or tier-string entry point, cannot be combined with speculation, and returns
    /// the exact policy/cache receipt needed by schema-2 candidate evidence.
    func evaluateKVTunerCandidateCohort(
        prompts: [[Int]],
        maxTokens: Int,
        policy: KVTunerCandidateRuntimePolicy
    ) throws -> [KVTunerCandidateRunResult] {
        precondition(!prompts.isEmpty, "KVTuner candidate cohort must be nonempty")
        precondition(
            prompts.allSatisfy { !$0.isEmpty },
            "KVTuner candidate prompts must be nonempty")
        precondition(maxTokens > 0, "KVTuner candidate maxTokens must be positive")
        guard let kvtunerRuntimeIdentity else {
            throw KVTunerCandidateRuntimeIdentityError.missingRuntimeIdentity
        }
        _ = try kvtunerRuntimeIdentity.validate(runtimePolicy: policy)
        let kind = KVCacheKind.kvtunerCandidate(policy)
        // The decoder is local to exactly one complete prompt cohort. Prompts reuse its compiled
        // graph and monotonically grown allocation, but retries and later candidates start from a
        // fresh cache so first-row capacity evidence is reproducible and memory cannot accumulate
        // with the number of candidates visited.
        var decoder = CompiledMLXDecoder(model: model, kvCache: kind)
        var results: [KVTunerCandidateRunResult] = []
        results.reserveCapacity(prompts.count)
        for (promptOrdinal, prompt) in prompts.enumerated() {
            decoder.reset()
            var tokens: [Int] = []
            var token = decoder.prefill(prompt)
            tokens.append(token)
            while tokens.count < maxTokens
                && token != kvtunerRuntimeIdentity.eosTokenID
            {
                token = decoder.step(last: token)
                tokens.append(token)
            }
            guard let telemetry = decoder.kvtunerCandidateKVTelemetry() else {
                preconditionFailure(
                    "KVTuner candidate decoder did not return candidate telemetry")
            }
            precondition(
                telemetry.runtimePolicySHA256
                    == policy.runtimePolicySHA256,
                "KVTuner candidate telemetry does not match its runtime policy")
            let finishReason: KVTunerCandidateFinishReason
            if tokens.last == kvtunerRuntimeIdentity.eosTokenID {
                finishReason = .endOfSequence
            } else {
                precondition(
                    tokens.count == maxTokens,
                    "KVTuner candidate stopped without EOS or exhausting its generation budget")
                finishReason = .generationBudgetExhausted
            }
            results.append(KVTunerCandidateRunResult(
                promptOrdinal: promptOrdinal,
                promptTokenIDsSHA256: taskTokenIDsSHA256(prompt),
                tokens: tokens,
                finishReason: finishReason,
                telemetry: telemetry))
        }
        return results
    }

    /// Captures KVTuner's offline per-layer metrics inside the model's isolation region. The
    /// exact config comes from the source snapshot sampled around this live model load; callers
    /// provide only Sendable token IDs and receive only scalar samples.
    func captureKVTunerSensitivity(
        promptTokenIDs: [[Int]],
        groupSize: Int,
        expectedRuntimeIdentity: KVTunerCandidateRuntimeIdentity
    ) throws -> [KVTunerSensitivitySample] {
        guard let kvtunerRuntimeIdentity else {
            throw KVTunerCandidateRuntimeIdentityError.missingRuntimeIdentity
        }
        guard kvtunerRuntimeIdentity == expectedRuntimeIdentity else {
            throw KVTunerCandidateRuntimeIdentityError
                .sourceIdentityChangedDuringModelLoad
        }
        return try KVTunerSensitivityCapture.capture(
            model: model,
            exactModelConfigData:
                kvtunerRuntimeIdentity.exactModelConfigData,
            promptTokenIDs: promptTokenIDs,
            groupSize: groupSize,
            precisionPairs:
                KVTunerSensitivityArtifact.canonicalPrecisionPairs)
    }

    /// Speculative-decoding generate (PLD first): routes to `CompiledMLXDecoder.generateSpec`,
    /// which drafts from the context, batch-verifies, accept-walks, and rolls the KV back —
    /// byte-identical to the plain greedy loop at temp 0 by construction. Same decoder-per-kind
    /// reuse as `generate` (compiled step + compiled verify survive across runs; in-place reset).
    func generateSpec(prompt: [Int], maxTokens: Int, eos: Int, kvCache kind: KVCacheKind, spec: SpecDecodeConfig)
        -> (
            tokens: [Int], submitTime: Double, tokenTimes: [Double],
            prefillDurationSeconds: Double?, stats: SpecDecodeStats
        )
    {
        let decoderKey = DecoderKey(
            kind: kind,
            affineAttentionMode: .materialize,
            kvarnAttentionMode: .materialize,
            kvarnStorageDType: nil)
        if decoders[decoderKey] == nil {
            decoders[decoderKey] = CompiledMLXDecoder(model: model, kvCache: kind)
        }
        var decoder = decoders[decoderKey]!
        defer { decoders[decoderKey] = decoder }
        decoder.reset() // in-place KV reset: compiled graph stays valid across runs
        return decoder.generateSpec(prompt: prompt, maxTokens: maxTokens, eos: eos, spec: spec)
    }

    /// Full-vocab RAW LOGITS per generated position at temp=0 — the `EngineDriver.logprobs`
    /// contract: index == token id, length == vocab, NOT top-k, NOT softmaxed. Runs the plain
    /// (uncompiled) forward on a fresh cache: this is the measurement path, not the perf path,
    /// and the per-position full-vocab readback is inherently synchronous anyway.
    /// fp16 -> float32 conversion is exact, so argmax over a returned row reproduces the
    /// greedy token chosen at that position.
    func logprobs(
        prompt: [Int], maxTokens: Int, eos: Int,
        kvCache kind: KVCacheKind,
        affineAttentionMode: AffineKVAttentionMode,
        kvarnAttentionMode: KVarNKVAttentionMode = .materialize,
        kvarnStorageDType: KVarNKVScalarDType? = nil
    ) -> [[Float]] {
        let cache = makeScoringCache(
            kind: kind,
            capacity: prompt.count + max(maxTokens, 0),
            affineAttentionMode: affineAttentionMode,
            kvarnAttentionMode: kvarnAttentionMode,
            kvarnStorageDType: kvarnStorageDType)
        var rows: [[Float]] = []
        let generationCount = max(maxTokens, 0)
        guard generationCount > 0 else { return rows }
        var logits = prefillScoringLogits(
            prompt: prompt,
            cache: cache,
            affineAttentionMode: affineAttentionMode,
            kvarnAttentionMode: kvarnAttentionMode)
        for index in 0..<generationCount {
            let last = logits[0..., -1, 0...] // [1, vocab] raw logits
            rows.append(last.asType(.float32).asArray(Float.self))
            let tok = argMax(last, axis: -1).item(Int.self)
            if tok == eos || index == generationCount - 1 { break }
            logits = model(
                MLXArray([tok]).reshaped([1, 1]),
                cache: cache)
        }
        captureQuantizedScoringTelemetry(
            kind, cache: cache, minTokens: prompt.count,
            affineAttentionMode: affineAttentionMode,
            kvarnAttentionMode: kvarnAttentionMode,
            kvarnStorageDType: kvarnStorageDType)
        return rows
    }

    /// Scores the four-choice task head with one fresh full-prompt cache and returns engagement
    /// from that exact local cache. The ordinary scoring accessors retain maximum telemetry for
    /// KL accounting; they are intentionally not used here because a prior larger run could make
    /// a silently unengaged task row appear valid.
    func taskChoiceLogits(
        prompt: [Int], kvCache kind: KVCacheKind,
        affineAttentionMode: AffineKVAttentionMode,
        kvarnAttentionMode: KVarNKVAttentionMode = .materialize,
        kvarnStorageDType: KVarNKVScalarDType? = nil
    ) -> TaskChoiceLogitsResult {
        precondition(!prompt.isEmpty, "task scoring requires a nonempty prompt")
        let cache = makeScoringCache(
            kind: kind, capacity: prompt.count,
            affineAttentionMode: affineAttentionMode,
            kvarnAttentionMode: kvarnAttentionMode,
            kvarnStorageDType: kvarnStorageDType)
        let logits = prefillScoringLogits(
            prompt: prompt,
            cache: cache,
            affineAttentionMode: affineAttentionMode,
            kvarnAttentionMode: kvarnAttentionMode)
        let last = logits[0..., -1, 0...]
        let row = last.asType(.float32).asArray(Float.self)
        var counts: [String: Int] = [:]

        switch kind {
        case .fp16:
            break
        case .affine(let tier):
            let affineCaches = cache.compactMap { $0 as? AffineKVCache }
            precondition(
                affineCaches.count == cache.count,
                "affine task-scoring cache contains a different cache type")
            let telemetry = AffineKVCacheTelemetry.capture(
                tier: tier, caches: affineCaches)
            precondition(
                telemetry.cachedTokens == prompt.count,
                "affine task-scoring cache did not consume the full prompt")
            counts["scoring_cached_tokens"] = telemetry.cachedTokens
            counts["scoring_attention_split"] =
                telemetry.attentionOperation == .splitQuantizedMM ? 1 : 0
            counts["scoring_attention_materialized"] =
                telemetry.attentionOperation == .materializedKV ? 1 : 0
        case .kvtuner(let selection):
            let affineCaches = cache.compactMap { $0 as? AffineKVCache }
            precondition(
                affineCaches.count == cache.count,
                "KVTuner task-scoring cache contains a different cache type")
            let telemetry: KVTunerKVCacheTelemetry
            do {
                telemetry = try KVTunerKVCacheTelemetry.capture(
                    selection: selection, caches: affineCaches)
            } catch {
                preconditionFailure(
                    "KVTuner task-scoring telemetry mismatch: \(error)")
            }
            precondition(
                telemetry.cachedTokens == prompt.count,
                "KVTuner task-scoring cache did not consume the full prompt")
            counts["scoring_cached_tokens"] = telemetry.cachedTokens
            counts["scoring_kvtuner_layers"] = telemetry.layerCount
            counts["scoring_attention_split"] =
                telemetry.attentionOperation == .splitQuantizedMM ? 1 : 0
            counts["scoring_attention_materialized"] =
                telemetry.attentionOperation == .materializedKV ? 1 : 0
        case .kvtunerCandidate:
            preconditionFailure(
                "KVTuner candidates are generation-only qualification inputs")
        case .kvarn(let cell):
            let kvarnCaches = cache.compactMap { $0 as? KVarNKVCache }
            precondition(
                kvarnCaches.count == cache.count,
                "KVarN task-scoring cache contains a different cache type")
            let telemetry = KVarNKVCacheTelemetry.capture(caches: kvarnCaches)
            precondition(
                telemetry.tier == cell.tier
                    && telemetry.iterations == cell.iterations
                    && telemetry.executionMode == .uncompiledCorrectness
                    && telemetry.attentionOperation
                        == expectedKVarNAttentionOperation(kvarnAttentionMode)
                    && (kvarnStorageDType == nil
                        || telemetry.storageKeyDType == kvarnStorageDType)
                    && (kvarnStorageDType == nil
                        || telemetry.storageValueDType == kvarnStorageDType)
                    && telemetry.cachedTokens == prompt.count,
                "KVarN task-scoring telemetry does not match its runtime cell")
            counts["scoring_cached_tokens"] = telemetry.cachedTokens
            counts["scoring_kvarn_completed_tiles"] =
                telemetry.completedTileCount
            counts["scoring_kvarn_compressed_tokens"] =
                telemetry.compressedTokens
            counts["scoring_kvarn_attention_split"] =
                telemetry.attentionOperation == .splitQuantizedMM ? 1 : 0
            counts["scoring_kvarn_attention_materialized"] =
                telemetry.attentionOperation == .materializedKV ? 1 : 0
            counts.merge(
                kvarnIngressEngagement(
                    telemetry, prefix: "scoring_kvarn"),
                uniquingKeysWith: { _, new in new })
        case .turboQuant:
            preconditionFailure(
                "TurboQuant is outside the authenticated task-coherence tier map")
        }
        return TaskChoiceLogitsResult(
            logits: row, engagement: EngagementCounters(counts))
    }

    /// Bounded prefill shared by every free-running/restricted-choice scoring route. The split
    /// attention path owns a score and softmax tensor proportional to queryTokens * cacheTokens;
    /// chunking therefore prevents one long prompt from constructing the full quadratic pair.
    /// Materialized and short prompts retain their historical single-forward behavior.
    private func prefillScoringLogits(
        prompt: [Int],
        cache: [any KVCache],
        affineAttentionMode: AffineKVAttentionMode,
        kvarnAttentionMode: KVarNKVAttentionMode
    ) -> MLXArray {
        precondition(!prompt.isEmpty, "scoring prefill requires a nonempty prompt")
        latestScoringPrefillTokenCounts.removeAll(keepingCapacity: true)
        if affineAttentionMode != .splitQuantizedMM
            && kvarnAttentionMode != .splitQuantizedMM
            || prompt.count <= Self.scoringChunkSize
        {
            latestScoringPrefillTokenCounts.append(prompt.count)
            return model(
                MLXArray(prompt).reshaped([1, prompt.count]),
                cache: cache)
        }

        var latest: MLXArray?
        for start in stride(
            from: 0, to: prompt.count, by: Self.scoringChunkSize)
        {
            let end = min(start + Self.scoringChunkSize, prompt.count)
            let width = end - start
            let ids = MLXArray(Array(prompt[start..<end]))
                .reshaped([1, width])
            let logits = model(ids, cache: cache)
            eval(logits)
            latestScoringPrefillTokenCounts.append(width)
            latest = logits
        }
        return latest!
    }

    /// Qualification receipt for the last scoring-prefill call. Only scalar widths cross actor
    /// isolation; production evidence continues to use cache engagement and allocator receipts.
    func scoringPrefillTokenCounts() -> [Int] {
        latestScoringPrefillTokenCounts
    }

    /// TEACHER-FORCED `logprobs`: row i is the next-token distribution given
    /// context = prompt + forced[0..<i]; forced[i] is fed as the next input regardless of
    /// argmax, and eos does NOT stop the loop (the forced continuation already encodes where
    /// its producer stopped). Exactly forced.count rows. Same measurement path as the
    /// free-running variant (plain forward, fresh cache) — the perf path's compiled-step +
    /// no-sync-readback design lives untouched in `generate`.
    func teacherForcedLogprobs(
        prompt: [Int], forced: [Int], kvCache kind: KVCacheKind,
        affineAttentionMode: AffineKVAttentionMode,
        kvarnAttentionMode: KVarNKVAttentionMode = .materialize,
        kvarnStorageDType: KVarNKVScalarDType? = nil
    ) -> [[Float]] {
        scoreForced(
            prompt: prompt, forced: forced, wanted: nil, kind: kind,
            affineAttentionMode: affineAttentionMode,
            kvarnAttentionMode: kvarnAttentionMode,
            kvarnStorageDType: kvarnStorageDType)
    }

    /// Sampled variant: same chunked forward over the full forced continuation (causal decoding
    /// requires every intermediate token as context regardless), but only converts+keeps a
    /// full-vocab row at `positions` — a long-context entry can be thousands of positions, and
    /// materializing every row would be ~0.6MB/row x thousands x 2 drivers. `positions` must be
    /// ascending (evenlySpacedPositions's contract); rows are returned in that order.
    func teacherForcedLogprobsAtPositions(
        prompt: [Int], forced: [Int], positions: [Int], kvCache kind: KVCacheKind,
        affineAttentionMode: AffineKVAttentionMode,
        kvarnAttentionMode: KVarNKVAttentionMode = .materialize,
        kvarnStorageDType: KVarNKVScalarDType? = nil
    ) -> [[Float]] {
        scoreForced(
            prompt: prompt, forced: forced, wanted: positions, kind: kind,
            affineAttentionMode: affineAttentionMode,
            kvarnAttentionMode: kvarnAttentionMode,
            kvarnStorageDType: kvarnStorageDType)
    }

    /// CHUNKED teacher-forced scoring (`forcedScoringPlan` in HarnessCore holds the pure
    /// bookkeeping): multi-token chunk forwards instead of one forward per forced token.
    ///
    /// WHY (the ~7K-context SIGKILL root cause): single-token stepping makes every step's
    /// transient buffers slightly LARGER than the last step's (the stock cache returns growing
    /// K/V slices), so MLX's buffer cache can never reuse a freed buffer and grows as
    /// O(context²) — measured 43GB of dead cache (active flat at 17GB) by position 6750 on
    /// Qwen3-32B-4bit, which, with the Python reference process ballooning identically, is
    /// exactly the ~6.7–7.1K jetsam SIGKILL ceiling the harness hit.
    ///
    /// The fix is BOTH layers, each necessary: chunking cuts the number of growing-transient
    /// events by the chunk factor and scores at prefill speed instead of decode speed (24K
    /// tokens: ~1 min/side instead of ~10), but the materialized K/V slices still grow chunk to
    /// chunk, so unbounded the cache still reaches ~62GB by 16K context (measured, ctxprobe
    /// `score` mode) — it is the allocator-cache bound in `loadSwiftDriver` that guarantees
    /// flat memory by evicting those unreusable buffers (32K tokens: 33.8GB peak footprint,
    /// cache pinned at 8GB). The per-chunk `eval` bounds the lazy graph so pending work cannot
    /// pile up across chunks.
    private func scoreForced(
        prompt: [Int], forced: [Int], wanted: [Int]?,
        kind: KVCacheKind,
        affineAttentionMode: AffineKVAttentionMode,
        kvarnAttentionMode: KVarNKVAttentionMode,
        kvarnStorageDType: KVarNKVScalarDType?
    ) -> [[Float]] {
        let input = prompt + forced.dropLast()
        let cache = makeScoringCache(
            kind: kind, capacity: input.count,
            affineAttentionMode: affineAttentionMode,
            kvarnAttentionMode: kvarnAttentionMode,
            kvarnStorageDType: kvarnStorageDType)
        let plan = forcedScoringPlan(
            promptCount: prompt.count, forcedCount: forced.count,
            wantedPositions: wanted, chunkSize: Self.scoringChunkSize)
        var rows: [[Float]] = []
        for chunk in plan.chunks {
            let ids = MLXArray(Array(input[chunk.inputRange])).reshaped([1, chunk.inputRange.count])
            let logits = model(ids, cache: cache)
            eval(logits)
            for sel in chunk.rows {
                rows.append(logits[0..., sel.localIndex, 0...].asType(.float32).asArray(Float.self))
            }
        }
        captureQuantizedScoringTelemetry(
            kind, cache: cache, minTokens: input.count,
            affineAttentionMode: affineAttentionMode,
            kvarnAttentionMode: kvarnAttentionMode,
            kvarnStorageDType: kvarnStorageDType)
        return rows
    }

    /// Scoring-path cache selection (Task 7): the MEASUREMENT forwards must run the same KV
    /// tier the config asked for, not silently fp16 — Phase 3's KL/ppl numbers come through
    /// here, not through the compiled decode path. fp16 keeps the stock model cache; affine,
    /// KVarN, and TurboQuant tiers get their requested concrete cache per layer, sized for the
    /// whole pass up front (scoring knows its total length; no chunked growth needed).
    private func makeScoringCache(
        kind: KVCacheKind, capacity: Int,
        affineAttentionMode: AffineKVAttentionMode,
        kvarnAttentionMode: KVarNKVAttentionMode,
        kvarnStorageDType: KVarNKVScalarDType?
    ) -> [any KVCache] {
        switch kind {
        case .fp16:
            return model.newCache(parameters: nil)
        case .affine, .kvtuner, .turboQuant, .kvarn:
            let layerCount = model.newCache(parameters: nil).count
            do {
                return try kind.makeCaches(
                    layerCount: layerCount,
                    capacity: Self.scoringCacheCapacity(
                        requested: capacity,
                        kind: kind,
                        kvarnAttentionMode: kvarnAttentionMode),
                    affineAttentionMode: affineAttentionMode,
                    kvarnAttentionMode: kvarnAttentionMode,
                    kvarnStorageDType: kvarnStorageDType)
            } catch {
                preconditionFailure(
                    "scoring KV-cache policy does not match the loaded model: \(error)")
            }
        case .kvtunerCandidate:
            preconditionFailure(
                "KVTuner candidates are unavailable to generic scoring paths")
        }
    }

    static func scoringCacheCapacity(
        requested: Int,
        kind: KVCacheKind,
        kvarnAttentionMode: KVarNKVAttentionMode
    ) -> Int {
        let requested = max(requested, 1)
        guard case .kvarn(let cell) = kind,
            kvarnAttentionMode == .splitQuantizedMM
        else { return requested }
        // Direct KVarN requires at least one position beyond the exact fp16 sink so its packed
        // route has a valid post-sink geometry even for very short scoring prompts.
        return max(requested, cell.tier.sinkTokens + 1)
    }

    /// Engagement backstop for the scoring paths: a requested lossy cache must have the
    /// matching concrete type and must have cached every scored position. A silent fp16
    /// fallback here would make the quality evidence measure the wrong thing.
    private func captureQuantizedScoringTelemetry(
        _ kind: KVCacheKind, cache: [any KVCache], minTokens: Int,
        affineAttentionMode: AffineKVAttentionMode,
        kvarnAttentionMode: KVarNKVAttentionMode,
        kvarnStorageDType: KVarNKVScalarDType?
    ) {
        switch kind {
        case .fp16:
            return
        case .affine(let tier):
            guard let affine = cache.first as? AffineKVCache, affine.offset >= minTokens else {
                preconditionFailure("affine tier requested but the affine scoring cache did not engage")
            }
            let affineCaches = cache.compactMap { $0 as? AffineKVCache }
            precondition(
                affineCaches.count == cache.count,
                "affine scoring cache contains a different cache type")
            let telemetry = AffineKVCacheTelemetry.capture(
                tier: tier, caches: affineCaches)
            precondition(
                telemetry.attentionOperation
                    == expectedAttentionOperation(affineAttentionMode),
                "affine scoring cache executed a different attention operation")
            if maximumAffineScoringTelemetry.map({
                telemetry.capacityTokens > $0.capacityTokens
            }) ?? true {
                maximumAffineScoringTelemetry = telemetry
            }
        case .kvtuner(let selection):
            guard let affine = cache.first as? AffineKVCache,
                affine.offset >= minTokens
            else {
                preconditionFailure(
                    "KVTuner requested but its scoring cache did not engage")
            }
            let affineCaches = cache.compactMap { $0 as? AffineKVCache }
            precondition(
                affineCaches.count == cache.count,
                "KVTuner scoring cache contains a different cache type")
            let telemetry: KVTunerKVCacheTelemetry
            do {
                telemetry = try KVTunerKVCacheTelemetry.capture(
                    selection: selection, caches: affineCaches)
            } catch {
                preconditionFailure(
                    "KVTuner scoring telemetry mismatch: \(error)")
            }
            precondition(
                telemetry.attentionOperation
                    == expectedAttentionOperation(affineAttentionMode),
                "KVTuner scoring cache executed a different attention operation")
            if maximumKVTunerScoringTelemetry[selection].map({
                telemetry.capacityTokens > $0.capacityTokens
            }) ?? true {
                maximumKVTunerScoringTelemetry[selection] = telemetry
            }
        case .kvtunerCandidate:
            preconditionFailure(
                "KVTuner candidates are unavailable to generic scoring paths")
        case .turboQuant:
            guard let tq = cache.first as? TurboQuantKVCache, tq.offset >= minTokens else {
                preconditionFailure("TurboQuant tier requested but the quantized scoring cache did not engage")
            }
        case .kvarn(let cell):
            guard cache.first is KVarNKVCache else {
                preconditionFailure("KVarN tier requested but the KVarN scoring cache did not engage")
            }
            let kvarnCaches = cache.compactMap { $0 as? KVarNKVCache }
            precondition(
                kvarnCaches.count == cache.count,
                "KVarN scoring cache contains a different cache type")
            let telemetry = KVarNKVCacheTelemetry.capture(caches: kvarnCaches)
            precondition(
                telemetry.cachedTokens >= minTokens,
                "KVarN tier requested but the KVarN scoring cache did not engage")
            precondition(
                telemetry.tier == cell.tier
                    && telemetry.iterations == cell.iterations
                    && telemetry.executionMode == .uncompiledCorrectness
                    && (kvarnStorageDType == nil
                        || telemetry.storageKeyDType == kvarnStorageDType)
                    && (kvarnStorageDType == nil
                        || telemetry.storageValueDType == kvarnStorageDType),
                "KVarN scoring telemetry does not match its requested runtime cell")
            precondition(
                telemetry.attentionOperation
                    == expectedKVarNAttentionOperation(kvarnAttentionMode),
                "KVarN scoring cache executed a different attention operation")
            let key = KVarNScoringTelemetryKey(
                cell: cell, attentionMode: kvarnAttentionMode,
                storageDType: kvarnStorageDType)
            if maximumKVarNScoringTelemetry[key].map({
                telemetry.capacityTokens > $0.capacityTokens
            }) ?? true {
                maximumKVarNScoringTelemetry[key] = telemetry
            }
        }
    }

    private func expectedAttentionOperation(
        _ mode: AffineKVAttentionMode
    ) -> AffineKVAttentionOperation {
        switch mode {
        case .materialize: .materializedKV
        case .splitQuantizedMM: .splitQuantizedMM
        }
    }

    private func expectedKVarNAttentionOperation(
        _ mode: KVarNKVAttentionMode
    ) -> KVarNKVAttentionOperation {
        switch mode {
        case .materialize: .materializedKV
        case .splitQuantizedMM: .splitQuantizedMM
        }
    }

    func affineScoringTelemetry() -> AffineKVCacheTelemetry? {
        maximumAffineScoringTelemetry
    }

    func kvtunerScoringTelemetry(
        for selection: KVTunerRuntimeSelection
    ) -> KVTunerKVCacheTelemetry? {
        maximumKVTunerScoringTelemetry[selection]
    }

    func kvarnScoringTelemetry(
        for cell: KVarNKVRuntimeCell,
        attentionMode: KVarNKVAttentionMode,
        storageDType: KVarNKVScalarDType? = nil
    ) -> KVarNKVCacheTelemetry? {
        maximumKVarNScoringTelemetry[KVarNScoringTelemetryKey(
            cell: cell, attentionMode: attentionMode,
            storageDType: storageDType)]
    }

    /// 512 balances per-chunk transient size (a [1, 512, vocab] fp16 logits buffer ~150MB)
    /// against forward-call count; matches mlx-lm's default prefill step size.
    private static let scoringChunkSize = 512
}

/// In-process `EngineDriver` over the compiled decode core — the only MLX-touching harness impl.
struct SwiftEngineDriver: EngineDriver {
    let engine: HarnessEngineActor
    let eos: Int
    let compressedKVAttentionAdmission:
        CompressedKVAttentionRuntimeAdmission?

    init(
        engine: HarnessEngineActor,
        eos: Int,
        compressedKVAttentionAdmission:
            CompressedKVAttentionRuntimeAdmission? = nil
    ) {
        self.engine = engine
        self.eos = eos
        self.compressedKVAttentionAdmission =
            compressedKVAttentionAdmission
    }

    func generate(prompt: [Int], config: RunConfig) async throws -> RunResult {
        let selection = try cacheSelection(config)
        let kind = selection.kind
        if config.exactPrefixRequest != nil {
            guard config.specDecode == nil else {
                throw SwiftEngineDriverError.unsupportedConfig(
                    "exact prefix cache with speculative decoding")
            }
            guard kind == .fp16,
                selection.affineAttentionMode == .materialize,
                selection.kvarnAttentionMode == .materialize,
                selection.kvarnStorageDType == nil
            else {
                throw SwiftEngineDriverError.unsupportedConfig(
                    "exact prefix cache requires scalar compiled dense-half full attention")
            }
        }
        if let spec = try Self.specConfig(config) {
            let out = await engine.generateSpec(
                prompt: prompt, maxTokens: config.maxTokens, eos: eos, kvCache: kind, spec: spec)
            // Engagement telemetry for the spec triad: `spec_drafted` is the marker proving
            // drafting actually happened (byte-identical output with zero drafts would be a
            // vacuous equivalence "pass"); accepted/steps feed the measurement verdict.
            let counts = [
                "decode": out.tokens.count,
                "spec_drafted": out.stats.drafted,
                "spec_accepted": out.stats.accepted,
                "spec_verify_steps": out.stats.verifySteps,
                "spec_normal_steps": out.stats.normalSteps,
                "spec_gate_disabled_steps": out.stats.gateDisabledSteps,
            ]
            return RunResult(
                tokens: out.tokens,
                engagement: .init(counts),
                acceptanceRate: out.stats.acceptanceRate,
                submitTime: out.submitTime,
                tokenTimes: out.tokenTimes,
                prefillDurationSeconds: out.prefillDurationSeconds)
        }
        let out = try await engine.generate(
            prompt: prompt, maxTokens: config.maxTokens, eos: eos,
            kvCache: kind,
            affineAttentionMode: selection.affineAttentionMode,
            kvarnAttentionMode: selection.kvarnAttentionMode,
            kvarnStorageDType: selection.kvarnStorageDType,
            exactPrefixRequest: config.exactPrefixRequest)
        var counts = ["decode": out.tokens.count]
        if let tq = out.turboQuantTokens {
            // In-graph cached-token count from the quantized cache — the lossy triad's
            // engagement marker (delta-checked by `verify`; absent on fp16 runs).
            counts["turboquant_tokens"] = tq
        }
        if let affine = out.affineTelemetry {
            counts["affine_tokens"] = affine.cachedTokens
            counts["affine_layers"] = affine.layerCount
            counts["affine_capacity_tokens"] = affine.capacityTokens
            counts["affine_payload_bytes"] = affine.payloadBytes
            counts["affine_metadata_bytes"] = affine.metadataBytes
            counts["affine_control_bytes"] = affine.controlBytes
            counts["affine_workspace_bytes"] = affine.workspaceBytes
            counts["affine_materialization_bytes"] =
                affine.materializationWorkspaceBytes
            counts["affine_attention_workspace_bytes"] =
                affine.attentionWorkspaceBytes
            counts["affine_attention_split"] =
                affine.attentionOperation == .splitQuantizedMM ? 1 : 0
            counts["affine_attention_materialized"] =
                affine.attentionOperation == .materializedKV ? 1 : 0
        }
        if let kvtuner = out.kvtunerTelemetry {
            counts["kvtuner_tokens"] = kvtuner.cachedTokens
            counts["kvtuner_layers"] = kvtuner.layerCount
            counts["kvtuner_capacity_tokens"] = kvtuner.capacityTokens
            counts["kvtuner_payload_bytes"] = kvtuner.payloadBytes
            counts["kvtuner_metadata_bytes"] = kvtuner.metadataBytes
            counts["kvtuner_control_bytes"] = kvtuner.controlBytes
            counts["kvtuner_workspace_bytes"] = kvtuner.workspaceBytes
            counts["kvtuner_materialization_bytes"] =
                kvtuner.materializationWorkspaceBytes
            counts["kvtuner_attention_workspace_bytes"] =
                kvtuner.attentionWorkspaceBytes
            counts["kvtuner_attention_split"] =
                kvtuner.attentionOperation == .splitQuantizedMM ? 1 : 0
            counts["kvtuner_attention_materialized"] =
                kvtuner.attentionOperation == .materializedKV ? 1 : 0
        }
        if let kvarn = out.kvarnTelemetry {
            counts["kvarn_tokens"] = kvarn.cachedTokens
            counts["kvarn_layers"] = kvarn.layerCount
            counts["kvarn_capacity_tokens"] = kvarn.capacityTokens
            counts["kvarn_completed_tiles"] = kvarn.completedTileCount
            counts["kvarn_compressed_tokens"] = kvarn.compressedTokens
            counts["kvarn_payload_bytes"] = kvarn.payloadBytes
            counts["kvarn_metadata_bytes"] = kvarn.metadataBytes
            counts["kvarn_alignment_padding_bytes"] = kvarn.alignmentPaddingBytes
            counts["kvarn_fp16_sink_bytes"] = kvarn.fp16SinkBytes
            counts["kvarn_fp16_tail_bytes"] = kvarn.fp16TailBytes
            counts["kvarn_control_bytes"] = kvarn.controlBytes
            counts["kvarn_workspace_bytes"] = kvarn.workspaceBytes
            counts["kvarn_materialization_bytes"] =
                kvarn.materializationWorkspaceBytes
            counts["kvarn_attention_workspace_bytes"] =
                kvarn.attentionWorkspaceBytes
            counts["kvarn_attention_split"] =
                kvarn.attentionOperation == .splitQuantizedMM ? 1 : 0
            counts["kvarn_attention_materialized"] =
                kvarn.attentionOperation == .materializedKV ? 1 : 0
            counts["kvarn_codec_iterations"] = kvarn.iterations
            counts["kvarn_uncompiled_correctness"] =
                kvarn.executionMode == .uncompiledCorrectness ? 1 : 0
            counts["kvarn_compiled"] =
                kvarn.executionMode == .compiled ? 1 : 0
            counts.merge(
                kvarnIngressEngagement(kvarn),
                uniquingKeysWith: { _, new in new })
        }
        if let requestStartMetrics = out.requestStartMetrics {
            counts.merge(
                exactPrefixEngagement(requestStartMetrics),
                uniquingKeysWith: { _, new in new })
        }
        return RunResult(
            tokens: out.tokens,
            engagement: .init(counts),
            acceptanceRate: nil, // plain (non-speculative) decode
            submitTime: out.submitTime,
            tokenTimes: out.tokenTimes,
            prefillDurationSeconds: out.prefillDurationSeconds,
            requestStartMetrics: out.requestStartMetrics)
    }

    func logprobs(prompt: [Int], config: RunConfig) async throws -> [[Float]] {
        let selection = try cacheSelection(config, allowSpec: false)
        return await engine.logprobs(
            prompt: prompt, maxTokens: config.maxTokens, eos: eos,
            kvCache: selection.kind,
            affineAttentionMode: selection.affineAttentionMode,
            kvarnAttentionMode: selection.kvarnAttentionMode,
            kvarnStorageDType: selection.kvarnStorageDType)
    }

    func taskChoiceLogits(
        prompt: [Int], config: RunConfig
    ) async throws -> TaskChoiceLogitsResult {
        guard !prompt.isEmpty else {
            throw SwiftEngineDriverError.unsupportedConfig(
                "empty task-scoring prompt")
        }
        guard config.maxTokens == 1 else {
            throw SwiftEngineDriverError.unsupportedConfig(
                "maxTokens=\(config.maxTokens) on one-position task scoring")
        }
        let selection = try cacheSelection(config, allowSpec: false)
        let kind = selection.kind
        if case .turboQuant = kind {
            throw SwiftEngineDriverError.unsupportedConfig(
                "TurboQuant has no authenticated task-coherence cell")
        }
        return await engine.taskChoiceLogits(
            prompt: prompt, kvCache: kind,
            affineAttentionMode: selection.affineAttentionMode,
            kvarnAttentionMode: selection.kvarnAttentionMode,
            kvarnStorageDType: selection.kvarnStorageDType)
    }

    /// Private, policy-typed qualification seam. Unlike `generate(config:)`, there is no string
    /// parsing or user-selectable dial route which could execute an unevaluated candidate.
    func evaluateKVTunerCandidateCohort(
        prompts: [[Int]],
        maxTokens: Int,
        policy: KVTunerCandidateRuntimePolicy
    ) async throws -> [KVTunerCandidateRunResult] {
        guard !prompts.isEmpty,
            prompts.allSatisfy({ !$0.isEmpty })
        else {
            throw SwiftEngineDriverError.unsupportedConfig(
                "empty KVTuner candidate cohort or prompt")
        }
        guard maxTokens > 0 else {
            throw SwiftEngineDriverError.unsupportedConfig(
                "KVTuner candidate maxTokens=\(maxTokens)")
        }
        return try await engine.evaluateKVTunerCandidateCohort(
            prompts: prompts,
            maxTokens: maxTokens,
            policy: policy)
    }

    /// Qualification-only sensitivity capture. Canonical cardinality and group sizes are
    /// enforced before entering the actor so partial or unsupported experiments cannot be
    /// serialized as protocol evidence.
    func captureKVTunerSensitivity(
        prompts: [[Int]],
        groupSize: Int,
        expectedRuntimeIdentity: KVTunerCandidateRuntimeIdentity
    ) async throws -> [KVTunerSensitivitySample] {
        guard prompts.count
                == KVTunerSensitivityArtifact.requiredSensitivityPromptCount
        else {
            throw SwiftEngineDriverError.unsupportedConfig(
                "KVTuner sensitivity prompt count=\(prompts.count)")
        }
        guard prompts.allSatisfy({ !$0.isEmpty }) else {
            throw SwiftEngineDriverError.unsupportedConfig(
                "empty KVTuner sensitivity prompt")
        }
        guard [64, 128].contains(groupSize) else {
            throw SwiftEngineDriverError.unsupportedConfig(
                "KVTuner sensitivity groupSize=\(groupSize)")
        }
        return try await engine.captureKVTunerSensitivity(
            promptTokenIDs: prompts,
            groupSize: groupSize,
            expectedRuntimeIdentity: expectedRuntimeIdentity)
    }

    func logprobs(prompt: [Int], forcedContinuation: [Int], config: RunConfig) async throws -> [[Float]] {
        let selection = try cacheSelection(config, allowSpec: false)
        return await engine.teacherForcedLogprobs(
            prompt: prompt, forced: forcedContinuation,
            kvCache: selection.kind,
            affineAttentionMode: selection.affineAttentionMode,
            kvarnAttentionMode: selection.kvarnAttentionMode,
            kvarnStorageDType: selection.kvarnStorageDType)
    }

    func logprobs(prompt: [Int], forcedContinuation: [Int], atPositions positions: [Int], config: RunConfig) async throws -> [[Float]] {
        let selection = try cacheSelection(config, allowSpec: false)
        return await engine.teacherForcedLogprobsAtPositions(
            prompt: prompt, forced: forcedContinuation, positions: positions,
            kvCache: selection.kind,
            affineAttentionMode: selection.affineAttentionMode,
            kvarnAttentionMode: selection.kvarnAttentionMode,
            kvarnStorageDType: selection.kvarnStorageDType)
    }

    func affineScoringTelemetry() async -> AffineKVCacheTelemetry? {
        await engine.affineScoringTelemetry()
    }

    func kvtunerScoringTelemetry(
        for selection: KVTunerRuntimeSelection
    ) async -> KVTunerKVCacheTelemetry? {
        await engine.kvtunerScoringTelemetry(for: selection)
    }

    func kvarnScoringTelemetry(
        for cell: KVarNKVRuntimeCell,
        attentionMode: KVarNKVAttentionMode
    ) async -> KVarNKVCacheTelemetry? {
        await engine.kvarnScoringTelemetry(
            for: cell, attentionMode: attentionMode,
            storageDType: resolvedKVarNStorageDType())
    }

    private func resolvedKVarNStorageDType() -> KVarNKVScalarDType? {
        switch compressedKVAttentionAdmission?.modelNativeDType {
        case .float16: .float16
        case .bfloat16: .bfloat16
        case .float32, nil: nil
        }
    }

    /// Validates the whole config and maps `kvQuant` through `KVCacheKind`'s closed affine /
    /// KVarN / TurboQuant allowlist. Anything else throws — a measurement must never silently
    /// run a different cache from the one requested.
    /// The scoring paths pass `allowSpec: false`: speculation changes how a decode loop steps,
    /// not what a teacher-forced forward scores, so a spec config there is a caller bug.
    private func cacheSelection(
        _ config: RunConfig, allowSpec: Bool = true
    ) throws -> SwiftEngineCacheSelection {
        guard config.temperature == 0 else {
            throw SwiftEngineDriverError.unsupportedConfig("temperature=\(config.temperature) (greedy-only engine)")
        }
        if !allowSpec, let spec = config.specDecode {
            throw SwiftEngineDriverError.unsupportedConfig("specDecode=\(spec) on a scoring path (decode-only feature)")
        }
        return try resolveSwiftEngineCacheSelection(
            config: config,
            compressedKVAttentionAdmission:
                compressedKVAttentionAdmission)
    }

    /// Maps `RunConfig.specDecode` to the engine's spec-decode configuration. nil → plain decode;
    /// "pld" → prompt-lookup drafter with the config's ngram/K/compile-strategy knobs. Unknown
    /// drafters and every unmeasured spec + lossy-KV combination fail loudly.
    private static func specConfig(_ config: RunConfig) throws -> SpecDecodeConfig? {
        guard let spec = config.specDecode else { return nil }
        guard spec == "pld" else {
            throw SwiftEngineDriverError.unsupportedConfig("specDecode=\(spec) (known drafters: pld)")
        }
        if let kv = config.kvQuant, kv != "fp16" {
            // Structurally supported (truncate is on the CompiledCache protocol) but never
            // measured together — reject rather than silently "measure" an untested combo.
            throw SwiftEngineDriverError.unsupportedConfig("specDecode=pld with kvQuant=\(kv) (unmeasured combination; use fp16)")
        }
        return SpecDecodeConfig(
            drafter: PromptLookupDrafter(ngram: config.specNgram ?? 3),
            maxDraft: config.specMaxDraft ?? 8,
            compiledVerify: config.specCompiledVerify ?? false)
    }
}

/// Loads model + tokenizer from a local directory and builds the in-process driver.
/// The (Sendable) tokenizer is bound into its own local BEFORE `ctx.model` is sent into the
/// actor init — detaching it from the region that transfers — so CPU-side encode/decode stays
/// usable afterward (same region discipline as the spike's `loadActor`).
func captureKVTunerQualificationRuntimeSourceSnapshot(
    modelPath: String
) throws -> KVTunerCandidateRuntimeSourceSnapshot {
    let modelDirectory = URL(fileURLWithPath: modelPath)
    let exactModelConfigData = try Data(
        contentsOf: modelDirectory.appendingPathComponent("config.json"))
    return try KVTunerCandidateRuntimeSourceSnapshot.load(
        exactModelConfigData: exactModelConfigData,
        checkpointManifestHash: try ProvenanceCLI.checkpointManifestHash(
            at: modelPath),
        checkpointContentSHA256:
            try ProvenanceCLI.fullContentCheckpointManifestSHA256(
                at: modelPath,
                exactConfigData: exactModelConfigData),
        tokenizerSHA256: try ProvenanceCLI.tokenizerManifestSHA256(
            at: modelPath))
}

func captureCompressedKVAttentionRuntimeSourceSnapshot(
    modelPath: String
) throws -> CompressedKVAttentionRuntimeSourceSnapshot {
    let modelDirectory = URL(fileURLWithPath: modelPath)
    let exactModelConfigData = try Data(
        contentsOf: modelDirectory.appendingPathComponent("config.json"))
    return try CompressedKVAttentionRuntimeSourceSnapshot.load(
        exactModelConfigData: exactModelConfigData,
        checkpointManifestHash: try ProvenanceCLI.checkpointManifestHash(
            at: modelPath),
        checkpointContentSHA256:
            try ProvenanceCLI.fullContentCheckpointManifestSHA256(
                at: modelPath,
                exactConfigData: exactModelConfigData),
        tokenizerSHA256: try ProvenanceCLI.tokenizerManifestSHA256(
            at: modelPath))
}

/// Turns the source snapshots sampled around model loading into the path-free identity carried
/// by the model actor. Disabled caches do not consume an identity even when the same snapshots
/// were captured for compressed-attention admission.
func resolveExactPrefixRuntimeIdentitySource(
    configuration: ExactPrefixCacheConfiguration,
    sourceBeforeLoad:
        CompressedKVAttentionRuntimeSourceSnapshot?,
    sourceAfterLoad:
        CompressedKVAttentionRuntimeSourceSnapshot?,
    modelInstanceID: String = UUID().uuidString.lowercased()
) throws -> ExactPrefixRuntimeIdentitySource? {
    guard configuration.policy.isEnabled else {
        return nil
    }
    guard let sourceBeforeLoad, let sourceAfterLoad else {
        throw ExactPrefixRuntimeError.missingRuntimeIdentity
    }
    let stableSource = try
        CompressedKVAttentionRuntimeSourceSnapshot.validateUnchanged(
            before: sourceBeforeLoad,
            after: sourceAfterLoad)
    let admission = try CompressedKVAttentionRuntimeAdmission.load(
        sourceSnapshot: stableSource)
    return try ExactPrefixRuntimeIdentitySource(
        admission: admission,
        modelInstanceID: modelInstanceID)
}

func loadSwiftDriver(
    modelPath: String,
    requireKVTunerQualificationIdentity: Bool = false,
    kvQuantTier: String? = nil,
    kvtunerSelection: KVTunerRuntimeSelection? = nil,
    compressedKVAttention: CompressedKVAttentionRequest? = nil,
    compressedKVAttentionExpectedCheckpointContentSHA256:
        String? = nil,
    exactPrefixCacheConfiguration:
        ExactPrefixCacheConfiguration = .disabled,
    memoryLimitBytes: Int? = nil,
    memoryCacheLimitBytes: Int =
        KVTunerSensitivityCaptureEnvironment.requiredMemoryCacheLimitBytes
) async throws -> (
    driver: SwiftEngineDriver,
    tokenizer: MLXLMCommon.Tokenizer,
    eos: Int
) {
    // Bound MLX's buffer cache for the measurement process. The default cache limit tracks the
    // (raised) GPU memory limit, so unreusable transients can hoard tens of GB before anything
    // evicts — and the harness runs a Python reference process with its own allocator on the
    // same box. 8GB is far above any measurement path's steady-state reuse working set (decode
    // transients are MBs; a scoring chunk's logits are ~150MB) but keeps two co-resident
    // processes comfortably inside physical RAM. harness_reference.py sets the same bound.
    if let memoryLimitBytes {
        Memory.memoryLimit = memoryLimitBytes
    }
    Memory.cacheLimit = memoryCacheLimitBytes
    let sourceIdentityBeforeLoad = requireKVTunerQualificationIdentity
        ? try captureKVTunerQualificationRuntimeSourceSnapshot(
            modelPath: modelPath)
        : nil
    let requiresCompressedAttentionAdmission =
        compressedKVAttention != nil
        || compressedKVAttentionExpectedCheckpointContentSHA256 != nil
        || kvtunerSelection != nil
        || exactPrefixCacheConfiguration.policy.isEnabled
    let compressedSourceBeforeLoad = requiresCompressedAttentionAdmission
        ? try captureCompressedKVAttentionRuntimeSourceSnapshot(
            modelPath: modelPath)
        : nil
    if let compressedSourceBeforeLoad {
        let admission = try CompressedKVAttentionRuntimeAdmission.load(
            sourceSnapshot: compressedSourceBeforeLoad)
        _ = try resolveSwiftEngineCacheSelection(
            config: RunConfig(
                kvQuant: kvQuantTier ?? kvtunerSelection?.cellID,
                kvtunerSelection: kvtunerSelection,
                compressedKVAttention: compressedKVAttention,
                compressedKVAttentionExpectedCheckpointContentSHA256:
                    compressedKVAttentionExpectedCheckpointContentSHA256),
            compressedKVAttentionAdmission: admission)
    }
    let ctx = try await loadModel(
        from: URL(fileURLWithPath: modelPath),
        using: #huggingFaceTokenizerLoader()
    )
    let tokenizer = ctx.tokenizer
    let eos = tokenizer.eosToken.flatMap { tokenizer.convertTokenToId($0) } ?? -1
    let runtimeIdentity: KVTunerCandidateRuntimeIdentity?
    if let sourceIdentityBeforeLoad {
        let sourceIdentityAfterLoad = try
            captureKVTunerQualificationRuntimeSourceSnapshot(
                modelPath: modelPath)
        let stableSourceIdentity = try
            KVTunerCandidateRuntimeSourceSnapshot.validateUnchanged(
                before: sourceIdentityBeforeLoad,
                after: sourceIdentityAfterLoad)
        runtimeIdentity = try KVTunerCandidateRuntimeIdentity.load(
            sourceSnapshot: stableSourceIdentity,
            eosTokenID: eos)
    } else {
        runtimeIdentity = nil
    }
    let compressedAdmission:
        CompressedKVAttentionRuntimeAdmission?
    let compressedSourceAfterLoad:
        CompressedKVAttentionRuntimeSourceSnapshot?
    if let compressedSourceBeforeLoad {
        let loadedCompressedSourceAfterLoad = try
            captureCompressedKVAttentionRuntimeSourceSnapshot(
                modelPath: modelPath)
        compressedSourceAfterLoad =
            loadedCompressedSourceAfterLoad
        let stableCompressedSource = try
            CompressedKVAttentionRuntimeSourceSnapshot.validateUnchanged(
                before: compressedSourceBeforeLoad,
                after: loadedCompressedSourceAfterLoad)
        let admission = try CompressedKVAttentionRuntimeAdmission.load(
            sourceSnapshot: stableCompressedSource)
        _ = try resolveSwiftEngineCacheSelection(
            config: RunConfig(
                kvQuant: kvQuantTier ?? kvtunerSelection?.cellID,
                kvtunerSelection: kvtunerSelection,
                compressedKVAttention: compressedKVAttention,
                compressedKVAttentionExpectedCheckpointContentSHA256:
                    compressedKVAttentionExpectedCheckpointContentSHA256),
            compressedKVAttentionAdmission: admission)
        compressedAdmission = admission
    } else {
        compressedAdmission = nil
        compressedSourceAfterLoad = nil
    }
    let exactPrefixRuntimeIdentitySource = try
        resolveExactPrefixRuntimeIdentitySource(
            configuration: exactPrefixCacheConfiguration,
            sourceBeforeLoad: compressedSourceBeforeLoad,
            sourceAfterLoad: compressedSourceAfterLoad)
    let engine = HarnessEngineActor(
        model: ctx.model,
        kvtunerRuntimeIdentity: runtimeIdentity,
        exactPrefixCacheConfiguration:
            exactPrefixCacheConfiguration,
        exactPrefixRuntimeIdentitySource:
            exactPrefixRuntimeIdentitySource)
    await engine.prepareExactPrefixRuntimeIdentity()
    if exactPrefixCacheConfiguration.eagerWarmupEnabled {
        _ = try await engine.performExactPrefixWarmup()
    }
    return (
        SwiftEngineDriver(
            engine: engine,
            eos: eos,
            compressedKVAttentionAdmission: compressedAdmission),
        tokenizer,
        eos)
}
