import Foundation

import MLX
import MLXHuggingFace
import MLXLMCommon
import HarnessCore
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

    public func render(
        messages: [OpenAIChatMessage],
        tools: [OpenAIToolSpec],
        enableThinking: Bool?,
        reasoningEffort: String?
    ) throws -> [Int] {
        let templateMessages: [[String: any Sendable]] = messages.map { message in
            var dict: [String: any Sendable] = [
                "role": message.role.rawValue,
                "content": message.text,
            ]
            if !message.toolCalls.isEmpty {
                dict["tool_calls"] = message.toolCalls.map { call -> [String: any Sendable] in
                    let argumentsObject: any Sendable
                    if let data = call.function.arguments.data(using: .utf8),
                        let parsed = try? JSONSerialization.jsonObject(with: data) {
                        argumentsObject = ServingJSONValue(foundation: parsed).asSendable
                    } else {
                        argumentsObject = [String: any Sendable]()
                    }
                    return [
                        "type": "function",
                        "function": [
                            "name": call.function.name,
                            "arguments": argumentsObject,
                        ] as [String: any Sendable],
                    ]
                }
            }
            if let toolCallId = message.toolCallId { dict["tool_call_id"] = toolCallId }
            if let name = message.name { dict["name"] = name }
            return dict
        }
        let toolSpecs: [ToolSpec]? = tools.isEmpty ? nil : tools.compactMap { $0.raw.asObjectSendable }
        var additionalContext: [String: any Sendable]? = nil
        if enableThinking != nil || reasoningEffort != nil {
            var context: [String: any Sendable] = [:]
            if let enableThinking { context["enable_thinking"] = enableThinking }
            if let reasoningEffort { context["reasoning_effort"] = reasoningEffort }
            additionalContext = context
        }
        return try tokenizer.applyChatTemplate(
            messages: templateMessages,
            tools: toolSpecs,
            additionalContext: additionalContext)
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
    /// The requested KV tier passed `selectKVCacheQuant` (it is runtime-wired) but the runtime does not
    /// yet build its quantized caches — a future state guarded so a `runtimeWiredKVTiers` flip cannot
    /// silently reach fp16 construction. Today unreachable (int8+ fail closed at selection).
    case kvQuantTierConstructionUnavailable(KVQuantTier)
    /// A qwen3_5 hybrid checkpoint (the default scalar-fallback route for the family) whose linear key
    /// head dim (Dk) is not a multiple of 32, which the gated-delta Metal kernel requires
    /// (`n_per_t = Dk / 32`, GatedDelta.swift:29). A misaligned Dk truncates/faults in the kernel at
    /// decode, so refuse the checkpoint at load — BEFORE any weight load or global `Memory` mutation —
    /// rather than reach the kernel. Mirrors the continuous adapter's incr-4 guard
    /// (`ContinuousServingModelLoadError.hybridKernelKeyHeadDimUnaligned`) for the scalar route that
    /// serves qwen3_5 by default (continuous admission is opt-in). Carries the offending Dk.
    case hybridKernelKeyHeadDimUnaligned(Int)
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
    /// Requested KV-cache storage tier for this serve. Default `.fp16` (the runtime's always-valid
    /// native storage). Non-fp16 tiers are resolved fail-closed at load via `selectKVCacheQuant`: a tier
    /// the runtime cannot store yet (int8 until its quality gate flips `runtimeWiredKVTiers`) refuses to
    /// start rather than silently serve fp16 while the fit-check sized for the smaller tier.
    public let kvQuantTier: KVQuantTier

    public init(
        launchedModel: String,
        modelDirectory: URL,
        memoryLimitBytes: Int,
        cacheLimitBytes: Int,
        backendConfiguration: ScalarServingBackendConfiguration,
        startupMessages: [OpenAIChatMessage] = Self.defaultStartupMessages,
        kvQuantTier: KVQuantTier = .fp16
    ) {
        self.launchedModel = launchedModel
        self.modelDirectory = modelDirectory
        self.memoryLimitBytes = memoryLimitBytes
        self.cacheLimitBytes = cacheLimitBytes
        self.backendConfiguration = backendConfiguration
        self.startupMessages = startupMessages
        self.kvQuantTier = kvQuantTier
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
    /// MLX allocator bytes sampled AFTER the startup parity generation (weights +
    /// one short greedy pass), so the KV footprint is included — the observable the
    /// sizer's KV estimate is cross-checked against. Default 0 keeps the init
    /// backward compatible for fixtures that don't exercise the live path.
    public let mlxActiveBytes: Int
    public let mlxCacheBytes: Int
    public let mlxPeakBytes: Int

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
        resetParityVerified: Bool,
        mlxActiveBytes: Int = 0,
        mlxCacheBytes: Int = 0,
        mlxPeakBytes: Int = 0
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
        self.mlxActiveBytes = mlxActiveBytes
        self.mlxCacheBytes = mlxCacheBytes
        self.mlxPeakBytes = mlxPeakBytes
    }

    /// Machine-readable startup-line fragment for the sampled MLX allocator bytes,
    /// using the startup line's snake_case convention. Mirrors the field names the
    /// continuous route publishes (`mlxActiveBytes`/`mlxCacheBytes`/`mlxPeakBytes`),
    /// so a KV cross-check can compare the two routes on identical keys.
    public var memoryFieldsFragment: String {
        "mlx_active_bytes=\(mlxActiveBytes) mlx_cache_bytes=\(mlxCacheBytes) "
            + "mlx_peak_bytes=\(mlxPeakBytes)"
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

    // Real-kernel viability guard for the qwen3_5 hybrid family on the DEFAULT scalar route (continuous
    // admission is opt-in, so an un-flagged qwen3_5 checkpoint lands here). The gated-delta Metal kernel
    // processes the linear key head dim (Dk) in fixed 32-wide chunks (GatedDelta.swift:29,
    // `n_per_t = Dk / 32`), so a Dk not divisible by 32 truncates/faults at decode. Read config.json and
    // refuse the checkpoint HERE — before any weight load or global `Memory` mutation — rather than reach
    // the kernel. Mirrors the continuous adapter's incr-4 guard. Only the qwen3_5 family is inspected;
    // dense (compiled) and every non-qwen3_5 family are untouched (probe returns nil → no-op).
    if let recurrentKeyHeadDim = scalarServingQwen35RecurrentKeyHeadDim(
        modelDirectory: configuration.modelDirectory),
        recurrentKeyHeadDim % 32 != 0
    {
        throw ScalarServingModelLoadError.hybridKernelKeyHeadDimUnaligned(recurrentKeyHeadDim)
    }

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
    let decoderRoute = try classifyScalarServingDecoderRoute(nativeCacheKinds)
    // Fail-closed KV-cache tier selection, BEFORE any live cache is built. A non-fp16 tier the runtime
    // cannot store yet (int8/turbo/tq* until a dated quality gate flips `runtimeWiredKVTiers`) throws
    // here — the process refuses to start rather than silently downgrade to fp16, which would leave the
    // fit-check's smaller-tier GREEN served at the larger fp16 footprint. fp16 (default/omitted) resolves
    // to `.fp16`, byte-identical to today. int8 QuantizedKVCache CONSTRUCTION + its long-session quality
    // gate is the M5-128 operator step (docs/task-inbox/2026-08-19-runtime-kv-quant-quality.md); the
    // selection wiring is done and inert until then.
    let kvCacheDecision = try selectKVCacheQuant(
        requested: configuration.kvQuantTier, nativeKinds: nativeCacheKinds)
    guard case .fp16 = kvCacheDecision else {
        // Unreachable today (selection yields only `.fp16`). Defensive: if a future `runtimeWiredKVTiers`
        // flip admits int8 through selection before the quantized-cache construction below is wired, fail
        // closed here instead of silently building fp16 caches for an int8-sized serve.
        throw ScalarServingModelLoadError.kvQuantTierConstructionUnavailable(configuration.kvQuantTier)
    }
    let codec = MLXScalarTextCodec(tokenizer: tokenizer)
    let stopTokenIDs = try resolveScalarServingStopTokenIDs(
        configuration: modelConfiguration,
        tokenizer: tokenizer)
    let stopStrings = modelConfiguration.effectiveStopStrings
    guard stopStrings.allSatisfy({ !$0.isEmpty }) else {
        throw ScalarServingModelLoadError.invalidStopStrings
    }
    let startupPrompt = try codec.render(
        messages: configuration.startupMessages,
        tools: [],
        enableThinking: nil,
        reasoningEffort: nil)
    guard !startupPrompt.isEmpty else {
        throw ScalarServingModelLoadError.emptyStartupPrompt
    }

    let inference: InferenceActor
    switch decoderRoute {
    case .compiled:
        inference = InferenceActor(decoder: CompiledMLXDecoder(model: context.model))
    case .nativeHeterogeneous:
        // Route the KV cache through the construction seam so the selected tier drives what is built
        // rather than being computed and discarded. `kvCacheDecision` is `.fp16` here by construction
        // (the fail-closed guard above throws otherwise, and int8 is rejected at selection for this
        // route's recurrent caches), so `buildRouteKVCaches` returns the native caches unchanged —
        // byte-identical to `context.model.newCache(parameters: nil)`. int8 construction activates only
        // when the M5 quality gate relaxes that guard.
        inference = InferenceActor(
            decoder: MLXDecoder(
                model: context.model,
                cache: buildRouteKVCaches(
                    decision: kvCacheDecision,
                    nativeCaches: context.model.newCache(parameters: nil))))
    }
    let parity = try await verifyScalarServingResetParity(
        inference: inference,
        promptTokens: startupPrompt,
        stopTokenIDs: stopTokenIDs)
    // Sample AFTER the parity generation so peak/active include the KV footprint of a
    // real (if short) decode — the value the sizer's KV estimate is cross-checked
    // against. Sampling right after weight load would report weights-only and make the
    // comparison meaningless.
    let memory = Memory.snapshot()
    var backendConfiguration = configuration.backendConfiguration
    backendConfiguration.toolCallFormat = servingToolCallFormat(
        inferred: modelConfiguration.toolCallFormat)
    // Thinking-with-tools policy. The agentic hybrid family (qwen3_5: Qwen3.5/3.6/3.8, the
    // `.nativeHeterogeneous` route) is trained to think AND call tools, so respect its template
    // default (flag off). Keep the legacy `enable_thinking:false` workaround only for dense
    // (compiled) models, where thinking-with-tools regressed reliability (QwenLM/Qwen3 #1817).
    backendConfiguration.disableThinkingWhenToolsActive = (decoderRoute == .compiled)
    // Streaming reasoning separation: the qwen3_5 hybrid family (.nativeHeterogeneous) emits its
    // reasoning block from token 0 with no leading <think> opener (live-attested 93e606a), so its
    // streamed thinking must be split into reasoning_content/content. Dense/compiled stays false
    // (passthrough) until a live capture attests its streamed shape — a recorded handoff, not a guess.
    backendConfiguration.thinksByDefault = servingThinksByDefault(route: decoderRoute)
    let backend = ScalarServingBackend(
        launchedModel: configuration.launchedModel,
        inference: inference,
        codec: codec,
        stopTokenIDs: stopTokenIDs,
        modelStopStrings: stopStrings,
        configuration: backendConfiguration)
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
        resetParityVerified: parity.verified,
        mlxActiveBytes: memory.activeMemory,
        mlxCacheBytes: memory.cacheMemory,
        mlxPeakBytes: memory.peakMemory)
    return LoadedScalarServingModel(
        backend: backend,
        startupReport: report)
}

/// Pre-load probe: the recurrent linear key head dim (Dk) of a qwen3_5 hybrid checkpoint, or nil for any
/// other family (or an unreadable/broken config). Reads config.json without loading weights. Non-qwen3_5
/// configs return nil so the caller's viability guard is a strict no-op for every other model; a qwen3_5
/// config that fails the strict geometry decode also returns nil, leaving the existing load path to
/// surface that error where it always did (this guard only newly rejects the valid-geometry-but-bad-Dk
/// case). The top-level `model_type` probe matches the continuous proof's `ModelTypeProbe`.
func scalarServingQwen35RecurrentKeyHeadDim(modelDirectory: URL) -> Int? {
    let configURL = modelDirectory.appendingPathComponent("config.json")
    guard let data = try? Data(contentsOf: configURL),
        let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
        (root["model_type"] as? String) == "qwen3_5",
        let geometry = try? ModelConfigDecoder.qwen35HybridGeometry(configJSON: data)
    else {
        return nil
    }
    return geometry.recurrent.keyHeadDim
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
