import Foundation

import MLX
import MLXHuggingFace
import MLXLLM
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
    /// A selected tier has no construction strategy for this scalar route. The production policy keeps
    /// int8 unreachable until quality approval; this remains the defensive boundary for future pairings.
    case kvQuantTierConstructionUnavailable(KVQuantTier)
    /// A qwen3_5 hybrid checkpoint (the default scalar-fallback route for the family) whose linear key
    /// head dim (Dk) is not a multiple of 32, which the gated-delta Metal kernel requires
    /// (`n_per_t = Dk / 32`, GatedDelta.swift:29). A misaligned Dk truncates/faults in the kernel at
    /// decode, so refuse the checkpoint at load — BEFORE any weight load or global `Memory` mutation —
    /// rather than reach the kernel. Mirrors the continuous adapter's incr-4 guard
    /// (`ContinuousServingModelLoadError.hybridKernelKeyHeadDimUnaligned`) for the scalar route that
    /// serves qwen3_5 by default (continuous admission is opt-in). Carries the offending Dk.
    case hybridKernelKeyHeadDimUnaligned(Int)
    /// At least one of the model's native caches classified ONLY via the family-neutral
    /// `ServingCacheKindReporting` marker protocol (no concrete-type match), and the model's family
    /// is not admitted for the load path that actually resolved — see
    /// `markerClassifiedFamiliesProvenOnlyViaResolvedOffloadedNGramPlan`. The marker protocol lets a
    /// bespoke cache wrapper report a serving-compatible shape without the classifier naming the
    /// family, but a shape match alone is not a serving proof — only a RECORDED live serving run is.
    /// Carries the lowercased family name; lifted by recording that proof and (for a family proven
    /// only on the offloaded n-gram route) confirming the offload plan actually resolved.
    case unprovenServingFamily(String)
    /// `ScalarServingModelLoadConfiguration.ngramOffloadPlanURL` was supplied but is not an absolute
    /// file URL — mirrors `modelDirectoryMustBeAbsolute`'s guard for the model directory.
    case ngramOffloadPlanMustBeAbsolute
    /// `ScalarServingModelLoadConfiguration.ngramOffloadPlanURL` was supplied but does not resolve to
    /// an existing regular file (missing path, or a directory) — mirrors
    /// `modelDirectoryUnavailable`'s existence guard for the model directory.
    case ngramOffloadPlanUnavailable
    /// A plan URL was supplied, but the checkpoint at `modelDirectory` is not the qwen4_exp family the
    /// offloaded n-gram path is built for (per `scalarServingModelType(modelDirectory:)`). The
    /// offloaded path's on-disk layout (sealed row file, chunk seal, PLE geometry) is specific to that
    /// checkpoint, so a plan supplied against any other family is an operator error, refused here
    /// before any weight load. Carries the observed `model_type` (`nil` when `config.json` is
    /// unreadable/absent).
    case ngramOffloadPlanUnsupportedFamily(String?)
}

enum ScalarServingDecoderStrategy: Equatable {
    /// Preserve the existing compiled fp16 fast path byte-for-byte.
    case compiledFP16
    /// Use the model's native forward path with caches constructed from the selected storage tier.
    /// The int8 branch is inert until the production policy's quality gate admits it.
    case nativeCaches(KVCacheQuantDecision)
}

func scalarServingDecoderStrategy(
    route: ScalarServingDecoderRoute,
    kvCacheDecision: KVCacheQuantDecision
) throws -> ScalarServingDecoderStrategy {
    switch (route, kvCacheDecision) {
    case (.compiled, .fp16):
        return .compiledFP16
    case (.compiled, .int8):
        return .nativeCaches(kvCacheDecision)
    case (.nativeHeterogeneous, .fp16):
        return .nativeCaches(.fp16)
    case (.nativeHeterogeneous, .int8):
        // Selection rejects this earlier because recurrent state cannot be represented by the
        // dense-only quantized cache. Retain a defensive construction boundary if call order drifts.
        throw ScalarServingModelLoadError.kvQuantTierConstructionUnavailable(.int8)
    }
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
    /// native storage). Non-fp16 tiers remain fail-closed until their runtime + quality gate passes.
    public let kvQuantTier: KVQuantTier
    /// When supplied, selects the offloaded n-gram load path (`loadOffloadedNGramModelContext`)
    /// instead of the default `loadModel(from:using:)`. Absent (`nil`, the default) means the
    /// default path — every existing call site is unaffected. Restricted to the qwen4_exp family;
    /// see `ScalarServingModelLoadError.ngramOffloadPlanUnsupportedFamily`.
    public let ngramOffloadPlanURL: URL?

    public init(
        launchedModel: String,
        modelDirectory: URL,
        memoryLimitBytes: Int,
        cacheLimitBytes: Int,
        backendConfiguration: ScalarServingBackendConfiguration,
        startupMessages: [OpenAIChatMessage] = Self.defaultStartupMessages,
        kvQuantTier: KVQuantTier = .fp16,
        ngramOffloadPlanURL: URL? = nil
    ) {
        self.launchedModel = launchedModel
        self.modelDirectory = modelDirectory
        self.memoryLimitBytes = memoryLimitBytes
        self.cacheLimitBytes = cacheLimitBytes
        self.backendConfiguration = backendConfiguration
        self.startupMessages = startupMessages
        self.kvQuantTier = kvQuantTier
        self.ngramOffloadPlanURL = ngramOffloadPlanURL
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
    if let ngramOffloadPlanURL = configuration.ngramOffloadPlanURL {
        guard ngramOffloadPlanURL.isFileURL,
            ngramOffloadPlanURL.path.hasPrefix("/")
        else {
            throw ScalarServingModelLoadError.ngramOffloadPlanMustBeAbsolute
        }
        var planIsDirectory: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: ngramOffloadPlanURL.path,
            isDirectory: &planIsDirectory),
            !planIsDirectory.boolValue
        else {
            throw ScalarServingModelLoadError.ngramOffloadPlanUnavailable
        }
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

    // Fail-closed family check for the offloaded n-gram dispatch seam. The offloaded path's on-disk
    // layout (sealed row file, chunk seal, PLE geometry) is specific to the qwen4_exp checkpoint, so a
    // plan file supplied against any other family is an operator error — refuse it HERE, before any
    // weight load or global `Memory` mutation, exactly like the qwen3_5 guard above.
    if configuration.ngramOffloadPlanURL != nil {
        let observedModelType = scalarServingModelType(modelDirectory: configuration.modelDirectory)
        guard observedModelType == "qwen4_exp" else {
            throw ScalarServingModelLoadError.ngramOffloadPlanUnsupportedFamily(observedModelType)
        }
    }

    Memory.memoryLimit = configuration.memoryLimitBytes
    Memory.cacheLimit = configuration.cacheLimitBytes
    Memory.clearCache()
    try validateScalarServingMemoryLimits(configuration)

    // Both branches yield the same `ModelContext`, so decoder-route classification and KV tier
    // selection downstream are unaffected by which path loaded it. The marker-family admission gate
    // below, however, DOES care which branch actually resolved: `offloadedNGramPlanResolved` is set
    // immediately after the branch that ran it succeeds, so it reflects the real load path rather
    // than a constant or the mere presence of a requested plan URL.
    let context: ModelContext
    let offloadedNGramPlanResolved: Bool
    if let ngramOffloadPlanURL = configuration.ngramOffloadPlanURL {
        context = try await loadOffloadedNGramModelContext(
            modelDirectory: configuration.modelDirectory,
            planFileURL: ngramOffloadPlanURL,
            tokenizerLoader: #huggingFaceTokenizerLoader())
        offloadedNGramPlanResolved = true
    } else {
        context = try await loadModel(
            from: configuration.modelDirectory,
            using: #huggingFaceTokenizerLoader())
        offloadedNGramPlanResolved = false
    }
    try validateScalarServingMemoryLimits(configuration)

    let tokenizer = context.tokenizer
    let modelConfiguration = context.configuration
    // Read this BEFORE `context.model` is `sending`-consumed into the `InferenceActor` below (both
    // decoder-strategy branches send it into the actor). A subsequent read of the same non-Sendable
    // model value after it has been sent into another isolation domain is a data-race risk the
    // compiler correctly rejects, so this must be captured now and threaded through as a plain value.
    let rejectedPromptTokenIDs = servingRejectedPromptTokenIDs(model: context.model)
    let nativeCacheClassifications = context.model.newCache(parameters: nil)
        .map(classifyScalarServingNativeCacheEntry)
    let nativeCacheKinds = nativeCacheClassifications.map(\.kind)
    // Fail-closed admission gate: a cache classified ONLY via the family-neutral marker protocol
    // (no concrete-type match) makes the family LOADABLE but proves nothing about live serving
    // correctness — the marker only asserts a cache-shape claim. Refuse here for the honest reason
    // unless this family already has a RECORDED live serving proof on the load path that actually
    // resolved (`ScalarServingModelLoadError` doc comment and
    // `markerClassifiedFamiliesProvenOnlyViaResolvedOffloadedNGramPlan`). Scoped strictly to
    // marker-classified entries: any family classified entirely by concrete type (qwen3, qwen3_5,
    // every dense model) is completely unaffected.
    if let admissionError = scalarServingMarkerFamilyAdmissionError(
        classifications: nativeCacheClassifications,
        family: scalarServingModelType(modelDirectory: configuration.modelDirectory),
        offloadedNGramPlanResolved: offloadedNGramPlanResolved) {
        throw admissionError
    }
    let decoderRoute = try classifyScalarServingDecoderRoute(nativeCacheKinds)
    // Fail-closed KV-cache tier selection before a live request cache is built. Production policy keeps
    // int8 unwired after its dated quality NO-GO, so a request throws instead of silently downgrading.
    // The construction strategy below remains compiled and testable for a future quality-approved flip.
    let kvCacheDecision = try selectKVCacheQuant(
        requested: configuration.kvQuantTier, nativeKinds: nativeCacheKinds)
    let decoderStrategy = try scalarServingDecoderStrategy(
        route: decoderRoute, kvCacheDecision: kvCacheDecision)
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
    switch decoderStrategy {
    case .compiledFP16:
        inference = InferenceActor(decoder: CompiledMLXDecoder(model: context.model))
    case .nativeCaches(let decision):
        // The same factory owns initial construction and every later request reset. For fp16 it returns
        // the model's native instances unchanged. If the quality gate later admits dense int8, this
        // already-built seam prevents reset from reverting quantized caches to native fp16.
        let model = context.model
        inference = InferenceActor(
            decoder: MLXDecoder(
                model: model,
                cacheFactory: {
                    buildRouteKVCaches(
                        decision: decision,
                        nativeCaches: model.newCache(parameters: nil))
                }))
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
    // Thinking-with-tools policy and streaming reasoning separation are keyed on the model FAMILY
    // (`model_type`), not the decoder route alone: `.nativeHeterogeneous` is a cache-shape route that
    // can carry more than one family, and only the live-attested qwen3_5 hybrid family (Qwen3.5/3.6/3.8)
    // is known to think AND call tools with a no-opener reasoning stream (live-attested 93e606a). Any
    // other `.nativeHeterogeneous` family conservatively keeps the legacy `enable_thinking:false`
    // workaround (dense/compiled always does, per QwenLM/Qwen3 #1817) and stays passthrough
    // (non-separating) until it is live-captured — see `StreamingReasoningPolicy.swift`.
    backendConfiguration.disableThinkingWhenToolsActive = servingDisablesThinkingWhenToolsActive(
        route: decoderRoute,
        modelType: scalarServingModelType(modelDirectory: configuration.modelDirectory))
    backendConfiguration.thinksByDefault = servingThinksByDefault(
        route: decoderRoute,
        modelType: scalarServingModelType(modelDirectory: configuration.modelDirectory))
    // Some model families `preconditionFailure` inside their forward path on specific input token
    // IDs (media sentinels the model never expects as free-standing prompt tokens); on the scalar
    // route that precondition failure aborts the server process. Generated tokens are already
    // masked away from these IDs, but a rendered PROMPT is not, so without this screen an admitted
    // chat request that merely tokenizes to one of them would be an HTTP-reachable process kill.
    // `rejectedPromptTokenIDs` was captured earlier, before `context.model` was sent into the
    // decoder actor — see the comment at its capture site.
    backendConfiguration.rejectedPromptTokenIDs = rejectedPromptTokenIDs
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

/// The loaded model's `model_type` string from `config.json`, or nil if it is missing/unreadable.
/// `ModelConfiguration` (the vendored loader's identifier/name/EOS type, `context.configuration`) has
/// no model-type field — it only carries the model's HF id/directory and display name — so config.json
/// is the only reliable source available here. Used to key the family-scoped thinking-flag gates
/// (`servingThinksByDefault`/`servingDisablesThinkingWhenToolsActive` in `StreamingReasoningPolicy.swift`)
/// to the actual model family rather than the decoder route alone, so a second family sharing
/// `.nativeHeterogeneous`'s cache shape does not silently inherit qwen3_5's attested streamed shape.
/// Mirrors `scalarServingQwen35RecurrentKeyHeadDim`'s config.json read above.
func scalarServingModelType(modelDirectory: URL) -> String? {
    let configURL = modelDirectory.appendingPathComponent("config.json")
    guard let data = try? Data(contentsOf: configURL),
        let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    else {
        return nil
    }
    return root["model_type"] as? String
}

/// How one cache's serving kind was determined by `classifyScalarServingNativeCacheEntry`.
public enum ScalarServingCacheClassificationSource: Equatable, Sendable {
    /// Matched one of the five audited concrete cache types.
    case concreteType
    /// Matched none of the five concrete types but conformed to `ServingCacheKindReporting`.
    case markerProtocol
    /// Matched neither — classified as `.unknown`.
    case unclassified
}

/// Classify one native cache into its serving-compatibility kind, and record how that kind was
/// determined.
///
/// The five concrete types are checked FIRST, in the same fixed order the classifier has always
/// used, each yielding `.concreteType`. Only in the fall-through arm — a cache that is none of the
/// five — is the family-neutral `ServingCacheKindReporting` marker protocol consulted, yielding
/// `.markerProtocol`. This order is load-bearing, not incidental: if the marker probe ran BEFORE the
/// concrete-type switch, a future mlx-swift-lm release that added `ServingCacheKindReporting`
/// conformance to one of the five existing types (e.g. `RotatingKVCache`) would silently reroute
/// every already-supported family through the marker path instead of the audited concrete-type path
/// — an invisible regression in cache classification for families that work today. Keeping the
/// concrete-type switch first means the marker protocol can only ever affect a cache type that
/// matches none of the five, so it never overrides an existing, audited classification.
///
/// The marker's `ServingCacheLayerKind` is mapped through an EXHAUSTIVE switch with NO `default:`
/// clause, so a future 5th `ServingCacheLayerKind` case is a compile error here rather than silently
/// classifying as `.unknown`.
public func classifyScalarServingNativeCacheEntry(
    _ cache: any KVCache
) -> (kind: ScalarServingNativeCacheKind, source: ScalarServingCacheClassificationSource) {
    switch cache {
    case is CacheList:
        return (.composite, .concreteType)
    case is MambaCache, is ArraysCache:
        return (.recurrentState, .concreteType)
    case is RotatingKVCache:
        return (.rotatingAttention, .concreteType)
    case is KVCacheSimple:
        return (.denseAttention, .concreteType)
    default:
        if let reporting = cache as? ServingCacheKindReporting {
            let kind: ScalarServingNativeCacheKind
            switch reporting.servingCacheLayerKind {
            case .denseAttention:
                kind = .denseAttention
            case .rotatingAttention:
                kind = .rotatingAttention
            case .recurrentState:
                kind = .recurrentState
            case .composite:
                kind = .composite
            }
            return (kind, .markerProtocol)
        }
        return (.unknown, .unclassified)
    }
}

/// Map the loaded model's native state shape into the pure serving compatibility contract.
public func classifyScalarServingNativeCaches(
    _ caches: [any KVCache]
) -> [ScalarServingNativeCacheKind] {
    caches.map { classifyScalarServingNativeCacheEntry($0).kind }
}

/// Families whose caches classify ONLY via `ServingCacheKindReporting` (no concrete-type match) and
/// that have a RECORDED LIVE serving proof — but ONLY on the specific load path that proof was
/// captured on. The marker protocol lets a bespoke cache wrapper report a serving-compatible shape
/// without the classifier naming the family, but a shape claim alone is not a serving proof.
///
/// Membership here means the family's live serving proof exists ONLY for the offloaded n-gram
/// route (`loadOffloadedNGramModelContext`), never for the plain, fully-resident `loadModel` route
/// that runs when no offload plan is configured. `qwen4_exp` (Qwen3.8-Flash-Next) is proven this
/// way: the fully-resident load is a ~106 GiB checkpoint with no live serving attestation of its
/// own and is a memory-fit disaster on a 128 GB production host, so a bare family allowlist would
/// wrongly admit that unproven, dangerous path too. `scalarServingMarkerFamilyAdmissionError` refuses
/// a listed family unless the caller also attests, via `offloadedNGramPlanResolved`, that the
/// offloaded route is the one that actually resolved for THIS load.
///
/// A family not listed here stays fail-closed exactly as before this gate existed, regardless of
/// which path loaded it. Mirrors `scalarHybridServingFamilies`
/// (`ServingCore/ScalarServingCacheLayoutPolicy.swift`): fail-closed by construction, exact-match
/// only (lowercased compare), and narrower than any structural classification — this is a
/// *serving-proof* allowlist, not a cache-shape allowlist.
private let markerClassifiedFamiliesProvenOnlyViaResolvedOffloadedNGramPlan: Set<String> = [
    "qwen4_exp"
]

/// Pure admission decision for the marker-classified serving gate. Separated from
/// `loadScalarServingModel` specifically so it is testable without a real checkpoint on disk:
/// the gate previously lived inline after `loadModel`, where no unit test could reach it.
///
/// Refuses only when at least one classification has source `.markerProtocol`. The family is
/// lowercased before comparison (a `nil`/missing family is treated as `"unknown"`), so comparisons
/// and the carried refusal string are both exact-match on the lowercased form. A family not in
/// `markerClassifiedFamiliesProvenOnlyViaResolvedOffloadedNGramPlan` is refused unconditionally. A
/// LISTED family is admitted only when `offloadedNGramPlanResolved` is also `true` — a resolved
/// offload plan is never a blanket bypass for every marker-classified family, and a listed family
/// is never admitted on the plain, fully-resident load path. Returns `nil` to admit.
public func scalarServingMarkerFamilyAdmissionError(
    classifications: [(kind: ScalarServingNativeCacheKind, source: ScalarServingCacheClassificationSource)],
    family: String?,
    offloadedNGramPlanResolved: Bool
) -> ScalarServingModelLoadError? {
    guard classifications.contains(where: { $0.source == .markerProtocol }) else {
        return nil
    }
    let resolvedFamily = (family ?? "unknown").lowercased()
    guard markerClassifiedFamiliesProvenOnlyViaResolvedOffloadedNGramPlan.contains(resolvedFamily)
    else {
        return .unprovenServingFamily(resolvedFamily)
    }
    guard offloadedNGramPlanResolved else {
        return .unprovenServingFamily(resolvedFamily)
    }
    return nil
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
