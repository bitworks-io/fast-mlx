import Foundation

import MLX
import MLXHuggingFace
import MLXLMCommon
import HuggingFace
import ServingCore
import SpikeCore
import Tokenizers

/// CPU-side chat-template and incremental-detokenization bridge for the pinned MLX tokenizer.
public struct MLXScalarTextCodec: ScalarServingTextCodec {
    private let tokenizer: any MLXLMCommon.Tokenizer

    public init(tokenizer: any MLXLMCommon.Tokenizer) {
        self.tokenizer = tokenizer
    }

    public func render(messages: [OpenAIChatMessage]) throws -> [Int] {
        let templateMessages: [[String: any Sendable]] = messages.map { message in
            [
                "role": message.role.rawValue,
                "content": message.text,
            ]
        }
        return try tokenizer.applyChatTemplate(messages: templateMessages)
    }

    public func makeDetokenizer() -> any ScalarServingDetokenizer {
        MLXScalarDetokenizer(tokenizer: tokenizer)
    }
}

public enum ScalarServingModelLoadError: Error, Equatable, Sendable {
    case invalidModelIdentifier
    case modelDirectoryMustBeAbsolute
    case modelDirectoryUnavailable
    case invalidMemoryLimit
    case invalidCacheLimit
    case cacheLimitExceedsMemoryLimit
    case memoryLimitNotApplied(expected: Int, observed: Int)
    case cacheLimitNotApplied(expected: Int, observed: Int)
    case invalidStopTokenIDs
    case invalidStopStrings
    case emptyStartupPrompt
    case startupDidNotGenerateToken
    case startupParityMismatch
}

public struct ScalarServingModelLoadConfiguration: Sendable {
    public static let defaultStartupMessages = [
        OpenAIChatMessage(
            role: .user,
            text: "Reply with one short word.")
    ]

    public let launchedModel: String
    public let modelDirectory: URL
    public let memoryLimitBytes: Int
    public let cacheLimitBytes: Int
    public let backendConfiguration: ScalarServingBackendConfiguration
    public let startupMessages: [OpenAIChatMessage]

    public init(
        launchedModel: String,
        modelDirectory: URL,
        memoryLimitBytes: Int,
        cacheLimitBytes: Int,
        backendConfiguration: ScalarServingBackendConfiguration,
        startupMessages: [OpenAIChatMessage] = Self.defaultStartupMessages
    ) {
        self.launchedModel = launchedModel
        self.modelDirectory = modelDirectory
        self.memoryLimitBytes = memoryLimitBytes
        self.cacheLimitBytes = cacheLimitBytes
        self.backendConfiguration = backendConfiguration
        self.startupMessages = startupMessages
    }
}

public struct ScalarServingStartupParity: Equatable, Sendable {
    public let promptTokenCount: Int
    public let generatedTokenCount: Int
    public let verified: Bool

    public init(
        promptTokenCount: Int,
        generatedTokenCount: Int,
        verified: Bool
    ) {
        self.promptTokenCount = promptTokenCount
        self.generatedTokenCount = generatedTokenCount
        self.verified = verified
    }
}

public struct ScalarServingModelStartupReport: Equatable, Sendable {
    public let launchedModel: String
    public let route: ServingExecutionRoute
    public let memoryLimitBytes: Int
    public let cacheLimitBytes: Int
    public let stopTokenCount: Int
    public let stopStringCount: Int
    public let nativeCacheKinds: [ScalarServingNativeCacheKind]
    public let startupPromptTokenCount: Int
    public let startupGeneratedTokenCount: Int
    public let resetParityVerified: Bool

    public init(
        launchedModel: String,
        route: ServingExecutionRoute,
        memoryLimitBytes: Int,
        cacheLimitBytes: Int,
        stopTokenCount: Int,
        stopStringCount: Int,
        nativeCacheKinds: [ScalarServingNativeCacheKind],
        startupPromptTokenCount: Int,
        startupGeneratedTokenCount: Int,
        resetParityVerified: Bool
    ) {
        self.launchedModel = launchedModel
        self.route = route
        self.memoryLimitBytes = memoryLimitBytes
        self.cacheLimitBytes = cacheLimitBytes
        self.stopTokenCount = stopTokenCount
        self.stopStringCount = stopStringCount
        self.nativeCacheKinds = nativeCacheKinds
        self.startupPromptTokenCount = startupPromptTokenCount
        self.startupGeneratedTokenCount = startupGeneratedTokenCount
        self.resetParityVerified = resetParityVerified
    }
}

public struct LoadedScalarServingModel: Sendable {
    public let backend: ScalarServingBackend
    public let startupReport: ScalarServingModelStartupReport

    public init(
        backend: ScalarServingBackend,
        startupReport: ScalarServingModelStartupReport
    ) {
        self.backend = backend
        self.startupReport = startupReport
    }
}

@discardableResult
public func validateScalarServingModelLoadConfiguration(
    _ configuration: ScalarServingModelLoadConfiguration
) throws -> ScalarServingModelLoadConfiguration {
    guard !configuration.launchedModel.trimmingCharacters(
        in: .whitespacesAndNewlines
    ).isEmpty else {
        throw ScalarServingModelLoadError.invalidModelIdentifier
    }
    guard configuration.modelDirectory.isFileURL,
        configuration.modelDirectory.path.hasPrefix("/")
    else {
        throw ScalarServingModelLoadError.modelDirectoryMustBeAbsolute
    }
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(
        atPath: configuration.modelDirectory.path,
        isDirectory: &isDirectory),
        isDirectory.boolValue
    else {
        throw ScalarServingModelLoadError.modelDirectoryUnavailable
    }
    guard configuration.memoryLimitBytes > 0 else {
        throw ScalarServingModelLoadError.invalidMemoryLimit
    }
    guard configuration.cacheLimitBytes > 0 else {
        throw ScalarServingModelLoadError.invalidCacheLimit
    }
    guard configuration.cacheLimitBytes <= configuration.memoryLimitBytes else {
        throw ScalarServingModelLoadError.cacheLimitExceedsMemoryLimit
    }
    guard !configuration.startupMessages.isEmpty else {
        throw ScalarServingModelLoadError.emptyStartupPrompt
    }
    return configuration
}

/// Load one local text model into the actor-confined scalar route and prove reset parity.
public func loadScalarServingModel(
    configuration rawConfiguration: ScalarServingModelLoadConfiguration
) async throws -> LoadedScalarServingModel {
    let configuration = try validateScalarServingModelLoadConfiguration(
        rawConfiguration)

    Memory.memoryLimit = configuration.memoryLimitBytes
    Memory.cacheLimit = configuration.cacheLimitBytes
    Memory.clearCache()
    try validateScalarServingMemoryLimits(configuration)

    let context = try await loadModel(
        from: configuration.modelDirectory,
        using: #huggingFaceTokenizerLoader())
    try validateScalarServingMemoryLimits(configuration)

    let tokenizer = context.tokenizer
    let modelConfiguration = context.configuration
    let nativeCacheKinds = classifyScalarServingNativeCaches(
        context.model.newCache(parameters: nil))
    try validateScalarServingCacheLayout(nativeCacheKinds)
    let codec = MLXScalarTextCodec(tokenizer: tokenizer)
    let stopTokenIDs = try resolveScalarServingStopTokenIDs(
        configuration: modelConfiguration,
        tokenizer: tokenizer)
    let stopStrings = modelConfiguration.effectiveStopStrings
    guard stopStrings.allSatisfy({ !$0.isEmpty }) else {
        throw ScalarServingModelLoadError.invalidStopStrings
    }
    let startupPrompt = try codec.render(
        messages: configuration.startupMessages)
    guard !startupPrompt.isEmpty else {
        throw ScalarServingModelLoadError.emptyStartupPrompt
    }

    let inference = InferenceActor(
        decoder: CompiledMLXDecoder(model: context.model))
    let parity = try await verifyScalarServingResetParity(
        inference: inference,
        promptTokens: startupPrompt,
        stopTokenIDs: stopTokenIDs)
    let backend = ScalarServingBackend(
        launchedModel: configuration.launchedModel,
        inference: inference,
        codec: codec,
        stopTokenIDs: stopTokenIDs,
        modelStopStrings: stopStrings,
        configuration: configuration.backendConfiguration)
    let report = ScalarServingModelStartupReport(
        launchedModel: configuration.launchedModel,
        route: .scalarGreedy,
        memoryLimitBytes: Memory.memoryLimit,
        cacheLimitBytes: Memory.cacheLimit,
        stopTokenCount: stopTokenIDs.count,
        stopStringCount: stopStrings.count,
        nativeCacheKinds: nativeCacheKinds,
        startupPromptTokenCount: parity.promptTokenCount,
        startupGeneratedTokenCount: parity.generatedTokenCount,
        resetParityVerified: parity.verified)
    return LoadedScalarServingModel(
        backend: backend,
        startupReport: report)
}

/// Map the loaded model's native state shape into the pure serving compatibility contract.
public func classifyScalarServingNativeCaches(
    _ caches: [any KVCache]
) -> [ScalarServingNativeCacheKind] {
    caches.map { cache in
        switch cache {
        case is CacheList:
            .composite
        case is MambaCache, is ArraysCache:
            .recurrentState
        case is RotatingKVCache:
            .rotatingAttention
        case is KVCacheSimple:
            .denseAttention
        default:
            .unknown
        }
    }
}

/// Match the pinned MLX generation loop's complete stop-token construction.
public func resolveScalarServingStopTokenIDs(
    configuration: ModelConfiguration,
    tokenizer: any MLXLMCommon.Tokenizer
) throws -> Set<Int> {
    var stopTokenIDs = configuration.eosTokenIds
    if let tokenizerEOS = tokenizer.eosTokenId {
        stopTokenIDs.insert(tokenizerEOS)
    }
    for token in configuration.extraEOSTokens {
        if let tokenID = tokenizer.convertTokenToId(token) {
            stopTokenIDs.insert(tokenID)
        }
    }
    if let unknownTokenID = tokenizer.unknownTokenId {
        stopTokenIDs.insert(unknownTokenID)
    }
    guard !stopTokenIDs.isEmpty,
        stopTokenIDs.allSatisfy({ $0 >= 0 })
    else {
        throw ScalarServingModelLoadError.invalidStopTokenIDs
    }
    return stopTokenIDs
}

/// Prove that the actor's request-start reset yields the same one-token greedy result twice.
public func verifyScalarServingResetParity(
    inference: InferenceActor,
    promptTokens: [Int],
    stopTokenIDs: Set<Int>
) async throws -> ScalarServingStartupParity {
    let first = try await runScalarServingStartupProbe(
        inference: inference,
        promptTokens: promptTokens,
        stopTokenIDs: stopTokenIDs)
    let second = try await runScalarServingStartupProbe(
        inference: inference,
        promptTokens: promptTokens,
        stopTokenIDs: stopTokenIDs)

    guard first.tokens == second.tokens,
        first.summary == second.summary
    else {
        throw ScalarServingModelLoadError.startupParityMismatch
    }
    guard first.tokens.count == 1,
        first.summary.generatedTokenCount == 1
    else {
        throw ScalarServingModelLoadError.startupDidNotGenerateToken
    }

    return ScalarServingStartupParity(
        promptTokenCount: first.summary.promptTokenCount,
        generatedTokenCount: first.summary.generatedTokenCount,
        verified: true)
}

private struct MLXScalarDetokenizer: ScalarServingDetokenizer {
    private var base: NaiveStreamingDetokenizer

    init(tokenizer: any MLXLMCommon.Tokenizer) {
        base = NaiveStreamingDetokenizer(tokenizer: tokenizer)
    }

    mutating func append(token: Int) {
        base.append(token: token)
    }

    mutating func next() -> String? {
        base.next()
    }
}

private struct ScalarServingStartupProbeResult {
    let tokens: [Int]
    let summary: InferenceRunSummary
}

private actor ScalarServingTokenAccumulator {
    private var tokens: [Int] = []

    func append(_ token: Int) {
        tokens.append(token)
    }

    func value() -> [Int] {
        tokens
    }
}

private func runScalarServingStartupProbe(
    inference: InferenceActor,
    promptTokens: [Int],
    stopTokenIDs: Set<Int>
) async throws -> ScalarServingStartupProbeResult {
    let accumulator = ScalarServingTokenAccumulator()
    let summary = try await inference.generateBounded(
        promptTokens: promptTokens,
        maxTokens: 1,
        stopTokenIDs: stopTokenIDs
    ) { token in
        await accumulator.append(token)
        return .continueGeneration
    }
    return ScalarServingStartupProbeResult(
        tokens: await accumulator.value(),
        summary: summary)
}

private func validateScalarServingMemoryLimits(
    _ configuration: ScalarServingModelLoadConfiguration
) throws {
    let observedMemoryLimit = Memory.memoryLimit
    guard observedMemoryLimit == configuration.memoryLimitBytes else {
        throw ScalarServingModelLoadError.memoryLimitNotApplied(
            expected: configuration.memoryLimitBytes,
            observed: observedMemoryLimit)
    }
    let observedCacheLimit = Memory.cacheLimit
    guard observedCacheLimit == configuration.cacheLimitBytes else {
        throw ScalarServingModelLoadError.cacheLimitNotApplied(
            expected: configuration.cacheLimitBytes,
            observed: observedCacheLimit)
    }
}
