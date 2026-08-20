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
    /// unconditionally (`kvCacheKind: .fp16`). A non-fp16 tier is resolved fail-closed at load via
    /// `selectKVCacheQuant`: a tier the runtime cannot store yet refuses to start rather than silently
    /// serve fp16 while the fit-check sized for the smaller tier (matches the scalar route).
    public let kvQuantTier: KVQuantTier
    /// Opt-in admission of the qwen3_5 hybrid architecture onto this continuous route (operator flag
    /// `--allow-hybrid-qwen35`). Default `false` — the proof rejects the hybrid family with
    /// `unsupportedModelFamily` (the executable then falls back to scalar serving). When `true`, it is
    /// threaded into `DenseContinuousBatchModelProof.verifying` so the proof carries qwen3_5's
    /// config-hash-pinned `HybridCacheGeometry`, and the cache-layout validator admits `.recurrentState`.
    public let allowHybridQwen35: Bool

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
        allowHybridQwen35: Bool = false
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
        modelProofVerified: Bool
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
    }
}

public struct LoadedContinuousServingModel: Sendable {
    public let backend: ContinuousServingBackend
    public let startupReport: ContinuousServingModelStartupReport

    public init(
        backend: ContinuousServingBackend,
        startupReport: ContinuousServingModelStartupReport
    ) {
        self.backend = backend
        self.startupReport = startupReport
    }
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

    Memory.memoryLimit = configuration.memoryLimitBytes
    Memory.cacheLimit = configuration.cacheLimitBytes
    Memory.clearCache()
    try validateContinuousServingMemoryLimits(configuration)

    let context = try await loadModel(
        from: configuration.modelDirectory,
        using: #huggingFaceTokenizerLoader())
    try validateContinuousServingMemoryLimits(configuration)

    let nativeCacheKinds = classifyScalarServingNativeCaches(
        context.model.newCache(parameters: nil))
    // Continuous-route cache-layout admission. `.recurrentState` (the hybrid GatedDeltaNet-linear
    // layers) is admitted ONLY when the proof certified qwen3_5 hybrid geometry — i.e. the operator
    // opted in AND the proof carries a `HybridCacheGeometry`. With the flag off this is byte-identical
    // to the prior `validateScalarServingCacheLayout` call (a hybrid layout still fails closed here).
    try validateContinuousServingCacheLayout(
        nativeCacheKinds,
        hybridAdmitted: proof.verifiedHybridGeometry != nil)
    // Fail-closed KV-cache tier selection, BEFORE the runtime builds any cache (it hardcodes
    // `kvCacheKind: .fp16` below). A non-fp16 tier the runtime cannot store yet throws here — the
    // continuous route refuses to start rather than silently serve fp16 under a smaller-tier-sized
    // verdict, matching the scalar route. fp16 (default/omitted) resolves to `.fp16`, byte-identical.
    let kvCacheDecision = try selectKVCacheQuant(
        requested: configuration.kvQuantTier, nativeKinds: nativeCacheKinds)
    guard case .fp16 = kvCacheDecision else {
        // Unreachable today (selection yields only `.fp16`). Defensive: a future `runtimeWiredKVTiers`
        // flip must not reach the fp16-hardcoded runtime below for a quantized request.
        throw ContinuousServingModelLoadError.kvQuantTierConstructionUnavailable(configuration.kvQuantTier)
    }
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
        reasoningEffort: nil)
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
        modelProofVerified: true)
    return LoadedContinuousServingModel(
        backend: backend,
        startupReport: report)
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
