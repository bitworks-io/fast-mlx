import Foundation

import HarnessCore
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import ServingCore
import SpikeCore
import Tokenizers

public enum ContinuousServingModelLoadError: Error, Equatable, Sendable {
    case invalidModelIdentifier
    case modelDirectoryMustBeAbsolute
    case modelDirectoryUnavailable
    case invalidMemoryLimit
    case invalidCacheLimit
    case cacheLimitExceedsMemoryLimit
    case invalidReservedKVLimit
    case reservedKVLimitExceedsMemoryLimit
    case invalidContextLimit
    case invalidReservedContextLimit
    case invalidAllocationChunk
    case invalidInitialDecodeReserve
    case defaultCompletionBudgetExceedsInitialDecodeReserve(
        defaultCompletionTokens: Int,
        initialDecodeReserve: Int)
    case invalidPublicationCapacity
    case invalidTraceLimit
    case soloPLDPolicyMismatch
    case memoryLimitNotApplied(expected: Int, observed: Int)
    case cacheLimitNotApplied(expected: Int, observed: Int)
    case invalidStopStrings
    case emptyStartupPrompt
    case startupResourceTelemetryUnavailable
    case startupSlotsNotReleased(Int)
    case startupKVBytesNotReleased(Int)
    /// The requested KV tier passed `selectKVCacheQuant` (runtime-wired) but the continuous runtime does
    /// not yet build its quantized caches (it hardcodes `kvCacheKind: .fp16`). Guarded so a future
    /// `runtimeWiredKVTiers` flip cannot silently serve fp16 for a quantized request. Today unreachable.
    case kvQuantTierConstructionUnavailable(KVQuantTier)
    /// An admitted qwen3_5 hybrid checkpoint whose linear key head dim (Dk) is not a multiple of 32,
    /// which the gated-delta Metal kernel requires (`n_per_t = Dk / 32`,
    /// Vendor/mlx-swift-lm/Libraries/MLXLMCommon/GatedDelta.swift:29). Fails closed on the REAL serving
    /// path before any weight load, so a bad checkpoint refuses cleanly instead of truncating/faulting
    /// at decode. Carries the offending Dk. (The proof deliberately does not enforce this — it is shared
    /// with the fp32 toy runtime tests that use Dk=1 and never invoke the Metal kernel.)
    case hybridKernelKeyHeadDimUnaligned(Int)
}

public struct ContinuousServingSoloPLDPolicy:
    Codable, Equatable, Sendable
{
    /// The only experimental PLD shape admitted for the current source-locked Qwen3 boundary.
    ///
    /// Drafts wider than one token changed a loaded temperature-zero argmax even when compilation
    /// was disabled. One drafted token retained exact scalar parity and is intentionally pinned.
    /// The three-token lookup key is the previously verified policy shape; a measured two-token
    /// experiment increased matches but did not meet the service speed gate. After a speculative
    /// bonus the standard PLD transition forwards `[last, draft]`. This policy remains
    /// non-promoted unless a later identical-workload service qualification clears its speed gate.
    public static let qwen3WidthOne = Self(
        ngram: 3,
        maxDraft: 1,
        lookback: 4_096,
        compiledVerify: true)

    public let ngram: Int
    public let maxDraft: Int
    public let lookback: Int
    public let compiledVerify: Bool

    public init(
        ngram: Int,
        maxDraft: Int,
        lookback: Int,
        compiledVerify: Bool
    ) {
        precondition(ngram > 0, "ngram must be positive")
        precondition(maxDraft > 0, "maxDraft must be positive")
        precondition(lookback > 0, "lookback must be positive")
        self.ngram = ngram
        self.maxDraft = maxDraft
        self.lookback = lookback
        self.compiledVerify = compiledVerify
    }

    func runtimeConfiguration() -> SpecDecodeConfig {
        SpecDecodeConfig(
            drafter: PromptLookupDrafter(ngram: ngram),
            maxDraft: maxDraft,
            lookback: lookback,
            compiledVerify: compiledVerify)
    }
}

public struct ContinuousServingModelLoadConfiguration: Sendable {
    public static let defaultStartupMessages = [
        OpenAIChatMessage(
            role: .user,
            text: "Reply with one short word.")
    ]

    public let launchedModel: String
    public let modelDirectory: URL
    public let memoryLimitBytes: Int
    public let cacheLimitBytes: Int
    public let maxReservedKVBytes: Int
    public let maxContextTokens: Int?
    public let maxReservedContextTokens: Int?
    public let allocationChunk: Int
    public let initialDecodeReserve: Int
    public let coordinatorConfiguration: ContinuousBatchConfiguration
    public let publicationCapacity: Int
    public let traceLimit: Int
    public let backendConfiguration: ContinuousServingBackendConfiguration
    public let soloPLDPolicy: ContinuousServingSoloPLDPolicy?
    public let startupMessages: [OpenAIChatMessage]
    /// Requested KV-cache storage tier. Default `.fp16` — the continuous runtime builds fp16 caches
    /// unconditionally (`kvCacheKind: .fp16`). Non-fp16 remains fail-closed at selection; the explicit
    /// validation below also prevents a future scalar-only capability flip from leaking into this route.
    public let kvQuantTier: KVQuantTier
    /// Opt-in admission of the qwen3_5 hybrid architecture onto this continuous route (operator flag
    /// `--allow-hybrid-qwen35`). Default `false` — the proof rejects the hybrid family with
    /// `unsupportedModelFamily` (the executable then falls back to scalar serving). When `true`, it is
    /// threaded into `DenseContinuousBatchModelProof.verifying` so the proof carries qwen3_5's
    /// config-hash-pinned `HybridCacheGeometry`, and the cache-layout validator admits `.recurrentState`.
    public let allowHybridQwen35: Bool
    /// `--default-sampling` (`FastMLXServeArguments.defaultSampling`). `.generationConfig` makes the load
    /// resolve `<modelDirectory>/generation_config.json` (shared with the scalar route:
    /// `resolveServingDefaultSampling`) AFTER the architecture proof and BEFORE any `Memory.*` mutation
    /// or weight load, failing closed with `ScalarServingModelLoadError.defaultSamplingGenerationConfigUnavailable`,
    /// and forwards the result into `ContinuousServingBackendConfiguration.samplingDefaults`. Default
    /// `.off`: no defaults, byte-for-byte the prior behavior. A pack the proof hands to the scalar
    /// fallback gets the scalar route's own resolution (the executable forwards the flag there too).
    public let defaultSampling: FastMLXServeDefaultSampling

    public init(
        launchedModel: String,
        modelDirectory: URL,
        memoryLimitBytes: Int,
        cacheLimitBytes: Int,
        maxReservedKVBytes: Int,
        maxContextTokens: Int? = nil,
        maxReservedContextTokens: Int? = nil,
        allocationChunk: Int = 256,
        initialDecodeReserve: Int = 512,
        coordinatorConfiguration: ContinuousBatchConfiguration,
        publicationCapacity: Int,
        traceLimit: Int = 0,
        backendConfiguration: ContinuousServingBackendConfiguration,
        soloPLDPolicy: ContinuousServingSoloPLDPolicy? = nil,
        startupMessages: [OpenAIChatMessage] = Self.defaultStartupMessages,
        kvQuantTier: KVQuantTier = .fp16,
        allowHybridQwen35: Bool = false,
        defaultSampling: FastMLXServeDefaultSampling = .off
    ) {
        self.launchedModel = launchedModel
        self.modelDirectory = modelDirectory
        self.memoryLimitBytes = memoryLimitBytes
        self.cacheLimitBytes = cacheLimitBytes
        self.maxReservedKVBytes = maxReservedKVBytes
        self.maxContextTokens = maxContextTokens
        self.maxReservedContextTokens = maxReservedContextTokens
        self.allocationChunk = allocationChunk
        self.initialDecodeReserve = initialDecodeReserve
        self.coordinatorConfiguration = coordinatorConfiguration
        self.publicationCapacity = publicationCapacity
        self.traceLimit = traceLimit
        self.backendConfiguration = backendConfiguration
        self.soloPLDPolicy = soloPLDPolicy
        self.startupMessages = startupMessages
        self.kvQuantTier = kvQuantTier
        self.allowHybridQwen35 = allowHybridQwen35
        self.defaultSampling = defaultSampling
    }
}

public struct ContinuousServingModelStartupReport: Equatable, Sendable {
    public let launchedModel: String
    public let route: ServingExecutionRoute
    public let memoryLimitBytes: Int
    public let cacheLimitBytes: Int
    public let maxReservedKVBytes: Int
    public let maxContextTokens: Int
    public let maxReservedContextTokens: Int
    public let modelFamily: CompressedKVAttentionModelFamily
    public let modelConfigurationSHA256: String
    public let layerCount: Int
    public let keyValueHeadCount: Int
    public let headDimension: Int
    public let stopTokenCount: Int
    public let stopStringCount: Int
    public let nativeCacheKinds: [ScalarServingNativeCacheKind]
    public let startupPromptTokenCount: Int
    public let startupGeneratedTokenCount: Int
    public let maxActiveSlots: Int
    public let maxPrefillSlots: Int
    public let prefillChunkSize: Int
    public let maxQueuedRequests: Int
    public let publicationCapacity: Int
    public let soloPLDPolicy: ContinuousServingSoloPLDPolicy?
    public let modelProofVerified: Bool
    /// The RESOLVED `--default-sampling` values this load assigned to the backend configuration's
    /// `samplingDefaults` (what omitted-parameter requests actually receive), or `nil` when none
    /// are applied. The startup line announces it via `ServingDefaultSamplingAnnouncement`.
    public let samplingDefaults: ServingSamplingDefaults?

    public init(
        launchedModel: String,
        route: ServingExecutionRoute,
        memoryLimitBytes: Int,
        cacheLimitBytes: Int,
        maxReservedKVBytes: Int,
        maxContextTokens: Int,
        maxReservedContextTokens: Int,
        modelFamily: CompressedKVAttentionModelFamily,
        modelConfigurationSHA256: String,
        layerCount: Int,
        keyValueHeadCount: Int,
        headDimension: Int,
        stopTokenCount: Int,
        stopStringCount: Int,
        nativeCacheKinds: [ScalarServingNativeCacheKind],
        startupPromptTokenCount: Int,
        startupGeneratedTokenCount: Int,
        maxActiveSlots: Int,
        maxPrefillSlots: Int,
        prefillChunkSize: Int,
        maxQueuedRequests: Int,
        publicationCapacity: Int,
        soloPLDPolicy: ContinuousServingSoloPLDPolicy?,
        modelProofVerified: Bool,
        samplingDefaults: ServingSamplingDefaults? = nil
    ) {
        self.launchedModel = launchedModel
        self.route = route
        self.memoryLimitBytes = memoryLimitBytes
        self.cacheLimitBytes = cacheLimitBytes
        self.maxReservedKVBytes = maxReservedKVBytes
        self.maxContextTokens = maxContextTokens
        self.maxReservedContextTokens = maxReservedContextTokens
        self.modelFamily = modelFamily
        self.modelConfigurationSHA256 = modelConfigurationSHA256
        self.layerCount = layerCount
        self.keyValueHeadCount = keyValueHeadCount
        self.headDimension = headDimension
        self.stopTokenCount = stopTokenCount
        self.stopStringCount = stopStringCount
        self.nativeCacheKinds = nativeCacheKinds
        self.startupPromptTokenCount = startupPromptTokenCount
        self.startupGeneratedTokenCount = startupGeneratedTokenCount
        self.maxActiveSlots = maxActiveSlots
        self.maxPrefillSlots = maxPrefillSlots
        self.prefillChunkSize = prefillChunkSize
        self.maxQueuedRequests = maxQueuedRequests
        self.publicationCapacity = publicationCapacity
        self.soloPLDPolicy = soloPLDPolicy
        self.modelProofVerified = modelProofVerified
        self.samplingDefaults = samplingDefaults
    }
}

public struct LoadedContinuousServingModel: Sendable {
    public let backend: ContinuousServingBackend
    public let startupReport: ContinuousServingModelStartupReport
    private let retainedProductionRouteEvidenceAuthorization:
        ContinuousServingProductionRouteEvidenceAuthorization?

    public init(
        backend: ContinuousServingBackend,
        startupReport: ContinuousServingModelStartupReport
    ) {
        self.backend = backend
        self.startupReport = startupReport
        self.retainedProductionRouteEvidenceAuthorization = nil
    }

    init(
        backend: ContinuousServingBackend,
        startupReport: ContinuousServingModelStartupReport,
        provenance: ContinuousServingLoadedModelProvenance
    ) {
        self.backend = backend
        self.startupReport = startupReport
        switch provenance {
        case .loadedContinuousServingModel:
            self.retainedProductionRouteEvidenceAuthorization = .init(backend: backend)
        }
    }

    @_spi(ProductionRouteEvidence)
    public func productionRouteEvidenceAuthorization() throws
        -> ContinuousServingProductionRouteEvidenceAuthorization
    {
        guard let retainedProductionRouteEvidenceAuthorization else {
            throw ContinuousServingProductionRouteEvidenceError
                .missingLoadedModelProvenance
        }
        return retainedProductionRouteEvidenceAuthorization
    }

    #if DEBUG
        @_spi(ProductionRouteEvidence)
        public static func testingLoadedContinuousServingModelWithLoaderProvenance(
            backend: ContinuousServingBackend,
            startupReport: ContinuousServingModelStartupReport
        ) -> Self {
            Self(
                backend: backend,
                startupReport: startupReport,
                provenance: .loadedContinuousServingModel)
        }
    #endif
}

enum ContinuousServingLoadedModelProvenance: Sendable {
    case loadedContinuousServingModel
}

@discardableResult
public func validateContinuousServingModelLoadConfiguration(
    _ configuration: ContinuousServingModelLoadConfiguration
) throws -> ContinuousServingModelLoadConfiguration {
    guard !configuration.launchedModel.trimmingCharacters(
        in: .whitespacesAndNewlines
    ).isEmpty else {
        throw ContinuousServingModelLoadError.invalidModelIdentifier
    }
    guard configuration.modelDirectory.isFileURL,
        configuration.modelDirectory.path.hasPrefix("/")
    else {
        throw ContinuousServingModelLoadError.modelDirectoryMustBeAbsolute
    }
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(
        atPath: configuration.modelDirectory.path,
        isDirectory: &isDirectory),
        isDirectory.boolValue
    else {
        throw ContinuousServingModelLoadError.modelDirectoryUnavailable
    }
    guard configuration.memoryLimitBytes > 0 else {
        throw ContinuousServingModelLoadError.invalidMemoryLimit
    }
    guard configuration.cacheLimitBytes > 0 else {
        throw ContinuousServingModelLoadError.invalidCacheLimit
    }
    guard configuration.cacheLimitBytes <= configuration.memoryLimitBytes else {
        throw ContinuousServingModelLoadError.cacheLimitExceedsMemoryLimit
    }
    guard configuration.maxReservedKVBytes > 0 else {
        throw ContinuousServingModelLoadError.invalidReservedKVLimit
    }
    guard configuration.maxReservedKVBytes <= configuration.memoryLimitBytes else {
        throw ContinuousServingModelLoadError.reservedKVLimitExceedsMemoryLimit
    }
    if let maxContextTokens = configuration.maxContextTokens,
        maxContextTokens <= 0
    {
        throw ContinuousServingModelLoadError.invalidContextLimit
    }
    if let maxReservedContextTokens = configuration.maxReservedContextTokens,
        maxReservedContextTokens <= 0
    {
        throw ContinuousServingModelLoadError.invalidReservedContextLimit
    }
    guard configuration.allocationChunk > 0 else {
        throw ContinuousServingModelLoadError.invalidAllocationChunk
    }
    guard configuration.initialDecodeReserve > 0 else {
        throw ContinuousServingModelLoadError.invalidInitialDecodeReserve
    }
    guard
        configuration.backendConfiguration.defaultMaximumCompletionTokens
            <= configuration.initialDecodeReserve
    else {
        throw ContinuousServingModelLoadError
            .defaultCompletionBudgetExceedsInitialDecodeReserve(
                defaultCompletionTokens:
                    configuration.backendConfiguration
                    .defaultMaximumCompletionTokens,
                initialDecodeReserve: configuration.initialDecodeReserve)
    }
    guard configuration.publicationCapacity > 0 else {
        throw ContinuousServingModelLoadError.invalidPublicationCapacity
    }
    guard configuration.traceLimit >= 0 else {
        throw ContinuousServingModelLoadError.invalidTraceLimit
    }
    let soloPLDQualified: Bool
    switch configuration.backendConfiguration.admission {
    case .immediateBatchNoSpec:
        soloPLDQualified = false
    case .dynamic(let admission, _):
        soloPLDQualified = admission.soloPLDQualified
    }
    guard soloPLDQualified == (configuration.soloPLDPolicy != nil),
        !soloPLDQualified
            || configuration.soloPLDPolicy == .qwen3WidthOne
    else {
        throw ContinuousServingModelLoadError.soloPLDPolicyMismatch
    }
    guard !configuration.startupMessages.isEmpty else {
        throw ContinuousServingModelLoadError.emptyStartupPrompt
    }
    return configuration
}

/// Whether the continuous route's loaded checkpoint begins generation inside the reasoning block, so
/// `ContinuousServingBackend.start` may separate `<think>` reasoning for it. Keyed on the SAME
/// artifact-derived inputs as the scalar route (`model_type` from `config.json` and whether the
/// checkpoint's own chat template attests the `<think>`/`</think>` markers), with the decoder route taken
/// from the proof: `.nativeHeterogeneous` when hybrid geometry was certified, `.compiled` (dense) otherwise.
/// Fail-closed: an unknown family, a template without the markers, or an unreadable artifact yields
/// `false` (byte-identical passthrough). This route has no `--chat-template` override.
func continuousServingThinksByDefault(
    modelDirectory: URL,
    hybridAdmitted: Bool
) -> Bool {
    servingThinksByDefault(
        route: hybridAdmitted ? .nativeHeterogeneous : .compiled,
        modelType: scalarServingModelType(modelDirectory: modelDirectory),
        templateAttestsThinkMarkers: scalarServingChatTemplateAttestsThinkMarkers(
            modelDirectory: modelDirectory))
}

/// Load one source-locked dense model into a single actor-confined continuous route.
public func loadContinuousServingModel(
    configuration rawConfiguration: ContinuousServingModelLoadConfiguration
) async throws -> LoadedContinuousServingModel {
    let configuration = try validateContinuousServingModelLoadConfiguration(
        rawConfiguration)

    // The proof reads and rejects unsupported architecture/geometry before weight loading. With
    // `allowHybridQwen35` off (default) a qwen3_5 config throws `unsupportedModelFamily` here — the
    // executable then falls back to scalar serving. With it on, the proof carries qwen3_5's
    // config-hash-pinned hybrid geometry through instead of throwing (opt-in continuous admission).
    let proof = try DenseContinuousBatchModelProof.verifying(
        modelDirectory: configuration.modelDirectory,
        allowHybridQwen35: configuration.allowHybridQwen35)

    // Real-kernel viability guard for the admitted hybrid path. The gated-delta Metal kernel processes
    // the linear key head dim (Dk) in fixed 32-wide chunks (GatedDelta.swift:29, `n_per_t = Dk / 32`),
    // so a Dk not divisible by 32 truncates/faults at decode. Refuse the checkpoint HERE — before any
    // weight load or global Memory mutation — rather than reach the kernel. Deliberately in the real
    // serving adapter, not the shared proof: the fp32 toy runtime tests admit Dk=1 and never invoke the
    // Metal kernel, so guarding in the proof would wrongly reject them.
    if let hybridGeometry = proof.verifiedHybridGeometry,
        hybridGeometry.recurrent.keyHeadDim % 32 != 0
    {
        throw ContinuousServingModelLoadError.hybridKernelKeyHeadDimUnaligned(
            hybridGeometry.recurrent.keyHeadDim)
    }

    // `--default-sampling generation-config`: resolve the checkpoint's sampling defaults HERE, through
    // the SAME helper the scalar load uses -- after the proof and Dk guard above, BEFORE any `Memory.*`
    // process-global mutation or weight load, so a missing/unparseable/`do_sample:false`/out-of-range
    // `generation_config.json` fails in milliseconds with the shared
    // `defaultSamplingGenerationConfigUnavailable` refusal. `nil` unless `.generationConfig`.
    let defaultSamplingDefaults = try resolveServingDefaultSampling(
        configuration.defaultSampling, modelDirectory: configuration.modelDirectory)

    Memory.memoryLimit = configuration.memoryLimitBytes
    Memory.cacheLimit = configuration.cacheLimitBytes
    Memory.clearCache()
    try validateContinuousServingMemoryLimits(configuration)

    let context = try await loadModel(
        from: configuration.modelDirectory,
        using: #huggingFaceTokenizerLoader())
    try validateContinuousServingMemoryLimits(configuration)

    // Prompt token IDs the model's own input validation rejects (empty for every family without such a
    // validation). Captured BEFORE `context.model` is handed to the runtime below, the same way the
    // scalar load captures it before sending the model into the decoder actor, and assigned to the
    // backend configuration so `ContinuousServingBackend.start` can answer 400 instead of letting the
    // prompt reach a `preconditionFailure`.
    let rejectedPromptTokenIDs = servingRejectedPromptTokenIDs(model: context.model)
    let nativeCacheKinds = classifyScalarServingNativeCaches(
        context.model.newCache(parameters: nil))
    // Continuous-route cache-layout admission. `.recurrentState` (the hybrid GatedDeltaNet-linear
    // layers) is admitted ONLY when the proof certified qwen3_5 hybrid geometry — i.e. the operator
    // opted in AND the proof carries a `HybridCacheGeometry`. With the flag off this is byte-identical
    // to the prior `validateScalarServingCacheLayout` call (a hybrid layout still fails closed here).
    try validateContinuousServingCacheLayout(
        nativeCacheKinds,
        hybridAdmitted: proof.verifiedHybridGeometry != nil)
    // Fail-closed KV-cache tier selection before the runtime builds any cache (it hardcodes
    // `kvCacheKind: .fp16` below). Non-fp16 currently throws at selection; the route-specific validation
    // also rejects an injected/future int8 decision rather than silently serving fp16.
    let kvCacheDecision = try selectKVCacheQuant(
        requested: configuration.kvQuantTier, nativeKinds: nativeCacheKinds)
    try validateContinuousServingKVCacheDecision(
        kvCacheDecision, requestedTier: configuration.kvQuantTier)
    let codec = MLXScalarTextCodec(tokenizer: context.tokenizer)
    let stopTokenIDs = try resolveScalarServingStopTokenIDs(
        configuration: context.configuration,
        tokenizer: context.tokenizer)
    let stopStrings = context.configuration.effectiveStopStrings
    guard stopStrings.allSatisfy({ !$0.isEmpty }) else {
        throw ContinuousServingModelLoadError.invalidStopStrings
    }
    let startupPrompt = try codec.render(
        messages: configuration.startupMessages,
        tools: [],
        enableThinking: nil,
        reasoningEffort: nil,
        addGenerationPrompt: nil)
    guard !startupPrompt.isEmpty else {
        throw ContinuousServingModelLoadError.emptyStartupPrompt
    }

    let maxContextTokens =
        configuration.maxContextTokens ?? proof.maximumContextTokens
    let maxReservedContextTokens =
        configuration.maxReservedContextTokens ?? maxContextTokens
    let runtime = try DenseContinuousBatchRuntime(
        model: context.model,
        verifiedBy: proof,
        allocationChunk: configuration.allocationChunk,
        maxContextTokens: maxContextTokens,
        maxReservedContextTokens: maxReservedContextTokens,
        initialDecodeReserve: configuration.initialDecodeReserve,
        maxReservedKVBytes: configuration.maxReservedKVBytes,
        kvCacheKind: .fp16,
        affineAttentionMode: .materialize,
        soloPLDConfiguration:
            configuration.soloPLDPolicy?.runtimeConfiguration())
    let coordinator = ContinuousBatchCoordinator(
        configuration: configuration.coordinatorConfiguration,
        runtime: runtime,
        automaticDrive: true,
        publicationCapacity: configuration.publicationCapacity,
        traceLimit: configuration.traceLimit)
    let startupGeneratedTokenCount = try await validateContinuousServingStartup(
        coordinator: coordinator,
        promptTokens: startupPrompt,
        stopTokenIDs: stopTokenIDs)
    var backendConfiguration = configuration.backendConfiguration
    backendConfiguration.toolCallFormat = servingToolCallFormat(
        inferred: context.configuration.toolCallFormat)
    // Streaming reasoning separation is keyed on the loaded family AND its own chat template (the
    // route is `.nativeHeterogeneous` only when the proof certified hybrid geometry). The tool-time
    // `disableThinkingWhenToolsActive` workaround stays whatever the caller configured.
    backendConfiguration.thinksByDefault = continuousServingThinksByDefault(
        modelDirectory: configuration.modelDirectory,
        hybridAdmitted: proof.verifiedHybridGeometry != nil)
    backendConfiguration.scorecardTraceCapacity = configuration.traceLimit
    backendConfiguration.rejectedPromptTokenIDs = rejectedPromptTokenIDs
    // `logit_bias`: the dense runtime applies it per row; the model's own `vocab_size` (the same
    // `config.json` value the runtime's verified proof carries) bounds every bias id at admission. An
    // unreadable config leaves it `nil`, which keeps `ContinuousServingBackend.supportsLogitBias`
    // `false` (a non-empty bias is then refused, never applied unchecked).
    backendConfiguration.logitBiasVocabularySize = scalarServingModelVocabSize(
        modelDirectory: configuration.modelDirectory)
    // Named/`"required"` `tool_choice` and `response_format` `json_object`/`json_schema`: the dense
    // runtime applies the grammar as a per-row, per-step token mask. Built the way the scalar load
    // builds it, from THIS checkpoint's own tokenizer; never lets loading fail (any problem logs one
    // `support=disabled` line and leaves the support `nil`, which keeps the capability `false` so
    // the request is refused with a 400 rather than served as free text). `json_schema` support is
    // derived from the `json_object` support (identical prerequisites, one shared trie build).
    backendConfiguration.jsonObjectConstraintSupport = loadScalarServingJSONObjectConstraintSupport(
        modelDirectory: configuration.modelDirectory,
        tokenizer: context.tokenizer)
    backendConfiguration.jsonSchemaConstraintSupport = loadScalarServingJSONSchemaConstraintSupport(
        sharing: backendConfiguration.jsonObjectConstraintSupport)
    // Request-time sampling defaults (`nil` unless `--default-sampling generation-config` resolved
    // above). Applied only where `ContinuousServingBackend.start` resolves a request's policy; the
    // startup validation above runs the coordinator directly and is unaffected.
    backendConfiguration.samplingDefaults = defaultSamplingDefaults
    let backend = ContinuousServingBackend(
        launchedModel: configuration.launchedModel,
        coordinator: coordinator,
        codec: codec,
        stopTokenIDs: stopTokenIDs,
        modelStopStrings: stopStrings,
        configuration: backendConfiguration)
    let scheduler = configuration.coordinatorConfiguration
    let report = ContinuousServingModelStartupReport(
        launchedModel: configuration.launchedModel,
        route: .continuousBatchNoSpec,
        memoryLimitBytes: Memory.memoryLimit,
        cacheLimitBytes: Memory.cacheLimit,
        maxReservedKVBytes: configuration.maxReservedKVBytes,
        maxContextTokens: maxContextTokens,
        maxReservedContextTokens: maxReservedContextTokens,
        modelFamily: proof.verifiedModelFamily,
        modelConfigurationSHA256: proof.modelConfigurationSHA256,
        layerCount: proof.verifiedLayerCount,
        keyValueHeadCount: proof.verifiedKeyValueHeadCount,
        headDimension: proof.verifiedHeadDimension,
        stopTokenCount: stopTokenIDs.count,
        stopStringCount: stopStrings.count,
        nativeCacheKinds: nativeCacheKinds,
        startupPromptTokenCount: startupPrompt.count,
        startupGeneratedTokenCount: startupGeneratedTokenCount,
        maxActiveSlots: scheduler.maxActiveSlots,
        maxPrefillSlots: scheduler.maxPrefillSlots,
        prefillChunkSize: scheduler.prefillChunkSize,
        maxQueuedRequests: scheduler.maxQueuedRequests,
        publicationCapacity: configuration.publicationCapacity,
        soloPLDPolicy: configuration.soloPLDPolicy,
        modelProofVerified: true,
        samplingDefaults: defaultSamplingDefaults)
    return LoadedContinuousServingModel(
        backend: backend,
        startupReport: report,
        provenance: .loadedContinuousServingModel)
}

func validateContinuousServingKVCacheDecision(
    _ decision: KVCacheQuantDecision,
    requestedTier: KVQuantTier
) throws {
    guard case .fp16 = decision else {
        // Continuous serving retains its existing fp16 cache family until a separate
        // construction/parity/quality gate wires compressed KV.
        throw ContinuousServingModelLoadError.kvQuantTierConstructionUnavailable(requestedTier)
    }
}

/// Exercise the exact continuous runtime before publishing a ready startup report.
///
/// The retained coordinator is reused for serving only after one bounded non-speculative
/// request has calibrated cache geometry, executed chunked prefill and decode, and released
/// every logical and physical reservation. Startup trace events are then discarded so later
/// request telemetry begins from a clean interval.
func validateContinuousServingStartup(
    coordinator: ContinuousBatchCoordinator,
    promptTokens: [Int],
    stopTokenIDs: Set<Int>
) async throws -> Int {
    let handle = try await coordinator.submit(
        ContinuousBatchSubmission(
            promptTokens: promptTokens,
            maxOutputTokens: 1,
            stopTokenIDs: stopTokenIDs,
            architecture: .denseAttention,
            requestsSpeculation: false))
    var generatedTokenCount = 0
    for try await _ in handle.tokens {
        generatedTokenCount += 1
    }
    await coordinator.waitUntilIdle()

    let slots = await coordinator.snapshots()
    guard slots.isEmpty else {
        throw ContinuousServingModelLoadError.startupSlotsNotReleased(
            slots.count)
    }
    guard let resources = await coordinator.runtimeResourceSnapshot() else {
        throw ContinuousServingModelLoadError
            .startupResourceTelemetryUnavailable
    }
    guard resources.reservedKVBytes == 0 else {
        throw ContinuousServingModelLoadError.startupKVBytesNotReleased(
            resources.reservedKVBytes)
    }
    _ = await coordinator.takeExecutionTrace()
    _ = await coordinator.takePlanObservations()
    _ = await coordinator.takeTimingTrace()
    return generatedTokenCount
}

private func validateContinuousServingMemoryLimits(
    _ configuration: ContinuousServingModelLoadConfiguration
) throws {
    let observedMemoryLimit = Memory.memoryLimit
    guard observedMemoryLimit == configuration.memoryLimitBytes else {
        throw ContinuousServingModelLoadError.memoryLimitNotApplied(
            expected: configuration.memoryLimitBytes,
            observed: observedMemoryLimit)
    }
    let observedCacheLimit = Memory.cacheLimit
    guard observedCacheLimit == configuration.cacheLimitBytes else {
        throw ContinuousServingModelLoadError.cacheLimitNotApplied(
            expected: configuration.cacheLimitBytes,
            observed: observedCacheLimit)
    }
}
