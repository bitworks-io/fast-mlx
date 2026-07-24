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
    case invalidPublicationCapacity
    case invalidTraceLimit
    case memoryLimitNotApplied(expected: Int, observed: Int)
    case cacheLimitNotApplied(expected: Int, observed: Int)
    case invalidStopStrings
    case emptyStartupPrompt
    case startupResourceTelemetryUnavailable
    case startupSlotsNotReleased(Int)
    case startupKVBytesNotReleased(Int)
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
    public let startupMessages: [OpenAIChatMessage]

    public init(
        launchedModel: String,
        modelDirectory: URL,
        memoryLimitBytes: Int,
        cacheLimitBytes: Int,
        maxReservedKVBytes: Int,
        maxContextTokens: Int? = nil,
        maxReservedContextTokens: Int? = nil,
        allocationChunk: Int = 256,
        initialDecodeReserve: Int = 384,
        coordinatorConfiguration: ContinuousBatchConfiguration,
        publicationCapacity: Int,
        traceLimit: Int = 0,
        backendConfiguration: ContinuousServingBackendConfiguration,
        startupMessages: [OpenAIChatMessage] = Self.defaultStartupMessages
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
        self.startupMessages = startupMessages
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
    guard configuration.publicationCapacity > 0 else {
        throw ContinuousServingModelLoadError.invalidPublicationCapacity
    }
    guard configuration.traceLimit >= 0 else {
        throw ContinuousServingModelLoadError.invalidTraceLimit
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

    // The proof reads and rejects unsupported architecture/geometry before weight loading.
    let proof = try DenseContinuousBatchModelProof.verifying(
        modelDirectory: configuration.modelDirectory)

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
    try validateScalarServingCacheLayout(nativeCacheKinds)
    let codec = MLXScalarTextCodec(tokenizer: context.tokenizer)
    let stopTokenIDs = try resolveScalarServingStopTokenIDs(
        configuration: context.configuration,
        tokenizer: context.tokenizer)
    let stopStrings = context.configuration.effectiveStopStrings
    guard stopStrings.allSatisfy({ !$0.isEmpty }) else {
        throw ContinuousServingModelLoadError.invalidStopStrings
    }
    let startupPrompt = try codec.render(messages: configuration.startupMessages)
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
        affineAttentionMode: .materialize)
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
    let backend = ContinuousServingBackend(
        launchedModel: configuration.launchedModel,
        coordinator: coordinator,
        codec: codec,
        stopTokenIDs: stopTokenIDs,
        modelStopStrings: stopStrings,
        configuration: configuration.backendConfiguration)
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
