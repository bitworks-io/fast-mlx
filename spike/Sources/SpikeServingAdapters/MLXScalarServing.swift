import CryptoKit
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
    /// `ScalarServingModelLoadConfiguration.inCheckpointMTPSelection` was supplied, but the checkpoint
    /// at `modelDirectory` is not the qwen4_exp family the in-checkpoint MTP drafter is loaded
    /// from -- the drafter's tensors live inside the TARGET checkpoint's own shard set, so a
    /// selection supplied against any other family is an operator error, refused here before any
    /// weight load. Mirrors `ngramOffloadPlanUnsupportedFamily`'s shape exactly. Carries the
    /// observed `model_type` (`nil` when `config.json` is unreadable/absent).
    case inCheckpointMTPUnsupportedFamily(String?)
    /// The loaded in-checkpoint MTP drafter's greedy token sequence, run through
    /// `MTPSpeculativeTokenIterator` against the SAME target, diverged from the target's own
    /// scalar-greedy decode of the identical startup prompt. What IS guaranteed: every token the
    /// speculative iterator emits is the target's own argmax as evaluated by its verify forward --
    /// deterministic and self-consistent. What is NOT guaranteed: token-for-token identity with a
    /// scalar single-position decode of the same prompt, since the two forward geometries do not
    /// round identically on this build. This gate refuses to serve on divergence anyway, since a
    /// mismatch is evidence the pairing may be unsafe, even though it is not proof the drafter
    /// itself is broken.
    case inCheckpointMTPStartupTokenSequenceMismatch
    /// The token sequences matched, but the iterator did not GENUINELY speculate: either it
    /// engaged sticky passthrough (carried, when known, as the iterator's own
    /// `passthroughReason`) or `drafter.draftBlock` was never actually invoked
    /// (`proposedDraftTokens == 0`). Without this check the equivalence gate above passes
    /// vacuously by comparing scalar decode to scalar decode -- `MTPSpeculativeTokenIterator`
    /// degrades to single-token passthrough SILENTLY rather than failing
    /// (`MTPSpeculativeTokenIterator.swift:167-176`), so a passing token-sequence comparison alone
    /// proves nothing about whether the drafter was ever exercised. Carries the observed telemetry
    /// (`proposedDraftTokens`, `acceptedDraftTokens`) alongside the iterator's `passthroughReason`
    /// so an operator hitting this sees what was measured, not just that the gate refused. NOTE:
    /// `acceptedDraftTokens == 0` alone is NOT what this case gates on -- see
    /// `inCheckpointMTPStartupEquivalenceDecision`'s clause (ii) comment for why gating startup on
    /// acceptance (rather than proposal) was a production-availability bug: a correct drafter whose
    /// first proposal for a round legitimately diverges from the target's greedy token completes
    /// that round with `accepted == 0`, and would previously never boot.
    case inCheckpointMTPStartupDidNotSpeculate(
        reason: String?, proposedDraftTokens: Int, acceptedDraftTokens: Int)
    /// `configuration.inCheckpointMTPSelection` was supplied but the RESOLVED decoder strategy is
    /// `.compiledFP16` -- a distinct, greedy-only compiled decoder path with no `cacheFactory` seam
    /// to hand `MTPSpeculativeDecoder` a target KV cache (that decoder's `init` requires one). A
    /// selection paired with that strategy would mean silently serving one and dropping the other,
    /// so this refuses at load instead. See `scalarServingInCheckpointMTPDecoderStrategyError`'s
    /// doc comment for why this combination is provably unreachable for qwen4_exp today, and why
    /// the guard is asserted here anyway rather than assumed.
    case inCheckpointMTPIncompatibleWithCompiledDecoderStrategy
}

enum ScalarServingDecoderStrategy: Equatable {
    /// Preserve the existing compiled fp16 fast path byte-for-byte.
    case compiledFP16
    /// Use the model's native forward path with caches constructed from the selected storage tier.
    /// The int8 branch is inert until the production policy's quality gate admits it.
    case nativeCaches(KVCacheQuantDecision)
}

/// Fail-closed compatibility guard between `configuration.inCheckpointMTPSelection` and the
/// RESOLVED `ScalarServingDecoderStrategy`, per
/// `docs/task-inbox/2026-09-07-mtp-decoder-bridge-DECISION.md`'s bridge constraint on decoder
/// strategy: MTP needs the `cacheFactory` seam `.nativeCaches` builds (shared with
/// `MTPSpeculativeDecoder`'s init), and `.compiledFP16` has no such seam. Returns the error to
/// throw at load, or `nil` to admit.
///
/// In production this combination cannot occur today: `loadScalarServingModel` already refuses
/// `inCheckpointMTPSelection` for any family other than `qwen4_exp` (the
/// `inCheckpointMTPUnsupportedFamily` guard above it), and every qwen4_exp checkpoint's native
/// cache classifies at least one layer `.recurrentState` -- its gated-delta-net layers map to
/// `.recurrentState` through that family's own serving cache-layer kind, and
/// `classifyScalarServingDecoderRoute`
/// (`ScalarServingCacheLayoutPolicy.swift`) returns `.nativeHeterogeneous` -- never `.compiled` --
/// whenever ANY layer classifies `.recurrentState`. So `scalarServingDecoderStrategy` can only
/// resolve qwen4_exp to `.nativeCaches`, never `.compiledFP16`. Checked here rather than assumed,
/// in case that upstream invariant ever changes.
func scalarServingInCheckpointMTPDecoderStrategyError(
    selection: FastMLXInCheckpointMTPSelection?,
    decoderStrategy: ScalarServingDecoderStrategy
) -> ScalarServingModelLoadError? {
    guard selection != nil, decoderStrategy == .compiledFP16 else {
        return nil
    }
    return .inCheckpointMTPIncompatibleWithCompiledDecoderStrategy
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

/// The `.nativeCaches` decoder strategy's KV cache factory, factored out into ONE named function so
/// `loadScalarServingModel` can hand the IDENTICAL factory to both `MLXDecoder` (the non-MTP
/// request path) and `MTPSpeculativeDecoder` (the MTP-selected path) — constraint #1 of
/// `docs/task-inbox/2026-09-07-mtp-decoder-bridge-DECISION.md`'s bridge: if MTP silently reverted
/// KV storage to the model's native fp16 cache while the operator selected a quantized tier, that
/// would be a silent downgrade. Every call to the returned closure re-derives fresh caches from
/// `model.newCache(parameters: nil)` through the SAME `decision`, so a `.fp16` decision is a
/// provable identity pass-through (`buildRouteKVCaches`'s own doc comment) and an `.int8` decision
/// wraps every native cache in the SAME quantized wrapper, regardless of which decoder branch
/// invokes it.
func scalarServingNativeCacheFactory(
    decision: KVCacheQuantDecision,
    model: any LanguageModel
) -> () -> [KVCache] {
    {
        buildRouteKVCaches(decision: decision, nativeCaches: model.newCache(parameters: nil))
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
    /// When supplied, loads the qwen4_exp (Qwen3.8-Flash-Next) in-checkpoint MTP drafter from
    /// `modelDirectory` and proves it at startup — see `verifyInCheckpointMTPStartupEquivalence`.
    /// Absent (`nil`, the default) skips the drafter entirely — every existing call site is
    /// unaffected. Restricted to the qwen4_exp family; see
    /// `ScalarServingModelLoadError.inCheckpointMTPUnsupportedFamily`.
    ///
    /// UPDATED (`docs/task-inbox/2026-09-07-mtp-decoder-bridge-DECISION.md`, "Option A"): the
    /// drafter now STAYS resident for the life of the returned `LoadedScalarServingModel` — it is
    /// `sending`-transferred into the `InferenceActor` this load builds, non-Sendable and all,
    /// exactly the way `context.model` already is, and every served request is decoded through
    /// `MTPSpeculativeDecoder` rather than the plain `MLXDecoder`. It is never exposed as a field
    /// on `LoadedScalarServingModel` itself (that type stays `Sendable`, and the drafter doesn't
    /// need to be reachable from outside the actor to be used), but it is no longer released after
    /// the startup gate the way this comment used to claim.
    public let inCheckpointMTPSelection: FastMLXInCheckpointMTPSelection?

    public init(
        launchedModel: String,
        modelDirectory: URL,
        memoryLimitBytes: Int,
        cacheLimitBytes: Int,
        backendConfiguration: ScalarServingBackendConfiguration,
        startupMessages: [OpenAIChatMessage] = Self.defaultStartupMessages,
        kvQuantTier: KVQuantTier = .fp16,
        ngramOffloadPlanURL: URL? = nil,
        inCheckpointMTPSelection: FastMLXInCheckpointMTPSelection? = nil
    ) {
        self.launchedModel = launchedModel
        self.modelDirectory = modelDirectory
        self.memoryLimitBytes = memoryLimitBytes
        self.cacheLimitBytes = cacheLimitBytes
        self.backendConfiguration = backendConfiguration
        self.startupMessages = startupMessages
        self.kvQuantTier = kvQuantTier
        self.ngramOffloadPlanURL = ngramOffloadPlanURL
        self.inCheckpointMTPSelection = inCheckpointMTPSelection
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

/// Result of the startup-time equivalence gate `verifyInCheckpointMTPStartupEquivalence` runs when
/// `ScalarServingModelLoadConfiguration.inCheckpointMTPSelection` is supplied. GREEDY ONLY — this
/// proves the greedy decode path matches the target's own scalar decode and that
/// `MTPSpeculativeTokenIterator` genuinely speculated (not sticky passthrough); it says nothing
/// about the SAMPLED (temperature != 0) generation path.
public struct ScalarServingInCheckpointMTPStartupVerdict: Equatable, Sendable {
    public let namespace: FastMLXInCheckpointMTPNamespace
    public let revision: String
    /// The artifact's own index-declared MTP source-key count under `namespace`'s prefix, as
    /// independently re-verified by `loadInCheckpointMTPDrafter` at load time (never re-derived).
    public let sourceKeyCount: Int
    public let promptTokenCount: Int
    public let generatedTokenCount: Int
    public let proposedDraftTokens: Int
    public let acceptedDraftTokens: Int
    /// Bytes attributable to the drafter alone, sampled as the delta between `Memory.snapshot()`
    /// immediately before `loadInCheckpointMTPDrafter` and immediately after — reported separately
    /// so an operator can see the drafter's own footprint in isolation, decomposed from the
    /// target's.
    ///
    /// UPDATED (`docs/task-inbox/2026-09-07-mtp-decoder-bridge-DECISION.md`, "Memory accounting"):
    /// this delta does NOT exclude the drafter from
    /// `ScalarServingModelStartupReport.mlxActiveBytes`/`mlxCacheBytes` the way it used to. The
    /// drafter now stays resident for the life of the process (see
    /// `ScalarServingModelLoadConfiguration.inCheckpointMTPSelection`'s doc comment), so the
    /// post-parity sample that field is built from — taken AFTER this delta, once the drafter is
    /// already alive for the rest of the load — genuinely includes the drafter's bytes. Excluding
    /// them there would be how a load passes a memory check and then OOMs on the first real KV
    /// allocation. This delta is a decomposition of that later, larger sample, not a carve-out from
    /// it.
    public let drafterActiveBytesDelta: Int
    public let drafterCacheBytesDelta: Int

    public init(
        namespace: FastMLXInCheckpointMTPNamespace,
        revision: String,
        sourceKeyCount: Int,
        promptTokenCount: Int,
        generatedTokenCount: Int,
        proposedDraftTokens: Int,
        acceptedDraftTokens: Int,
        drafterActiveBytesDelta: Int,
        drafterCacheBytesDelta: Int
    ) {
        self.namespace = namespace
        self.revision = revision
        self.sourceKeyCount = sourceKeyCount
        self.promptTokenCount = promptTokenCount
        self.generatedTokenCount = generatedTokenCount
        self.proposedDraftTokens = proposedDraftTokens
        self.acceptedDraftTokens = acceptedDraftTokens
        self.drafterActiveBytesDelta = drafterActiveBytesDelta
        self.drafterCacheBytesDelta = drafterCacheBytesDelta
    }

    /// Machine-readable startup-line fragment proving the qwen4_exp in-checkpoint MTP gate
    /// genuinely ran and genuinely speculated -- mirrors
    /// `ExactQwen35MTPServeStartupReport.machineReadableFields()`'s style. Every key is prefixed
    /// `in_checkpoint_mtp` (never bare `fit_`/`exact_mtp_`) so it cannot collide with either
    /// frozen key namespace already emitted on the same startup line. All fields are always
    /// present, including zero-valued ones (`accepted_draft_tokens=0` is a legitimate, meaningful
    /// outcome per this type's own doc comment -- it must be visible, not silently omitted).
    public func machineReadableFields() -> String {
        [
            "in_checkpoint_mtp=true",
            "in_checkpoint_mtp_namespace=\(namespace.rawValue)",
            "in_checkpoint_mtp_revision=\(revision)",
            "in_checkpoint_mtp_source_key_count=\(sourceKeyCount)",
            "in_checkpoint_mtp_prompt_token_count=\(promptTokenCount)",
            "in_checkpoint_mtp_generated_token_count=\(generatedTokenCount)",
            "in_checkpoint_mtp_proposed_draft_tokens=\(proposedDraftTokens)",
            "in_checkpoint_mtp_accepted_draft_tokens=\(acceptedDraftTokens)",
            "in_checkpoint_mtp_drafter_active_bytes_delta=\(drafterActiveBytesDelta)",
            "in_checkpoint_mtp_drafter_cache_bytes_delta=\(drafterCacheBytesDelta)",
        ].joined(separator: " ")
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
    ///
    /// When `inCheckpointMTPSelection` was supplied, this sample is taken AFTER the drafter is
    /// loaded and retained (it stays resident for the life of the process — see that
    /// configuration field's doc comment), so it genuinely includes the drafter's weight bytes on
    /// top of the target's. It is no longer drafter-bytes-excluded the way it was before the
    /// decoder bridge landed; `ScalarServingInCheckpointMTPStartupVerdict.drafterActiveBytesDelta`
    /// is a decomposition of (part of) this same total, not a carve-out from it.
    public let mlxActiveBytes: Int
    public let mlxCacheBytes: Int
    public let mlxPeakBytes: Int
    /// Present only when `ScalarServingModelLoadConfiguration.inCheckpointMTPSelection` was supplied
    /// AND the startup equivalence gate passed (a failure throws before this report is built, so
    /// this field is never populated with a failed verdict). `nil` (the default) is the ordinary,
    /// unaffected case for every existing call site — no request ever reaches the qwen4_exp
    /// in-checkpoint MTP drafter unless the caller opted in.
    public let inCheckpointMTPStartupVerdict: ScalarServingInCheckpointMTPStartupVerdict?

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
        mlxPeakBytes: Int = 0,
        inCheckpointMTPStartupVerdict: ScalarServingInCheckpointMTPStartupVerdict? = nil
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
        self.inCheckpointMTPStartupVerdict = inCheckpointMTPStartupVerdict
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

    // Same fail-closed shape as the guard immediately above: the in-checkpoint MTP drafter's
    // tensors live inside the qwen4_exp target checkpoint's own shard set, so a selection supplied
    // against any other family is an operator error — refuse it HERE, before any weight load or
    // global `Memory` mutation.
    if configuration.inCheckpointMTPSelection != nil {
        let observedModelType = scalarServingModelType(modelDirectory: configuration.modelDirectory)
        guard observedModelType == "qwen4_exp" else {
            throw ScalarServingModelLoadError.inCheckpointMTPUnsupportedFamily(observedModelType)
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
    // Fail-closed BEFORE any drafter weight load: see `scalarServingInCheckpointMTPDecoderStrategyError`'s
    // doc comment for why this specific combination is unreachable for qwen4_exp today, and why it
    // is still asserted rather than assumed.
    if let decoderStrategyError = scalarServingInCheckpointMTPDecoderStrategyError(
        selection: configuration.inCheckpointMTPSelection, decoderStrategy: decoderStrategy) {
        throw decoderStrategyError
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

    // qwen4_exp in-checkpoint MTP drafter: load it and prove it BEFORE `context.model` is
    // `sending`-consumed into the `InferenceActor` below — the same region-isolation reason as
    // `rejectedPromptTokenIDs`'s capture above. This gate is its own pre-send reference
    // generation with its own `context.model.newCache(parameters:)`, run through both the plain
    // scalar decode path and `MTPSpeculativeTokenIterator` against the identical target.
    //
    // UPDATED (`docs/task-inbox/2026-09-07-mtp-decoder-bridge-DECISION.md`, "Option A"): the
    // drafter is no longer released after this gate. `retainedInCheckpointMTPDrafter` carries it
    // (declared OUTSIDE this `if let`'s scope so ARC doesn't drop it at the closing brace) into
    // the decoder-strategy switch below, where it is `sending`-transferred into the
    // `InferenceActor` as `MTPSpeculativeDecoder`'s drafter — the same actor-confinement discipline
    // `context.model` already gets, never exposed as a field on the `Sendable`
    // `LoadedScalarServingModel`.
    var inCheckpointMTPStartupVerdict: ScalarServingInCheckpointMTPStartupVerdict?
    var retainedInCheckpointMTPDrafter: (any MTPDrafterModel)?
    if let inCheckpointMTPSelection = configuration.inCheckpointMTPSelection {
        // Sampled BEFORE the drafter load so its OWN contribution can be reported separately on
        // the verdict (`drafterActiveBytesDelta`/`drafterCacheBytesDelta`) — a decomposition of,
        // not a carve-out from, the later `mlxActiveBytes`/`mlxCacheBytes`/`mlxPeakBytes` sample
        // below, which now genuinely includes the drafter's bytes since it stays resident (see
        // that report field's doc comment).
        let preDrafterMemory = Memory.snapshot()
        let loadedDrafter = try loadInCheckpointMTPDrafter(
            modelDirectory: configuration.modelDirectory,
            expectedNamespace: inCheckpointMTPRuntimeNamespace(inCheckpointMTPSelection.namespace),
            expectedSourceKeyCount: inCheckpointMTPSelection.expectedSourceKeyCount,
            revision: inCheckpointMTPSelection.revision)
        let postDrafterMemory = Memory.snapshot()
        let equivalence = try verifyInCheckpointMTPStartupEquivalence(
            mainModel: context.model,
            drafter: loadedDrafter.drafter,
            promptTokens: startupPrompt,
            stopTokenIDs: stopTokenIDs)
        inCheckpointMTPStartupVerdict = ScalarServingInCheckpointMTPStartupVerdict(
            namespace: inCheckpointMTPSelection.namespace,
            revision: inCheckpointMTPSelection.revision,
            sourceKeyCount: loadedDrafter.sourceKeyCount,
            promptTokenCount: equivalence.promptTokenCount,
            generatedTokenCount: equivalence.generatedTokenCount,
            proposedDraftTokens: equivalence.proposedDraftTokens,
            acceptedDraftTokens: equivalence.acceptedDraftTokens,
            drafterActiveBytesDelta: postDrafterMemory.activeMemory - preDrafterMemory.activeMemory,
            drafterCacheBytesDelta: postDrafterMemory.cacheMemory - preDrafterMemory.cacheMemory)
        // Retained across this block's closing brace — see the comment above this `if let` for
        // why. `Memory.clearCache()` below only reclaims the equivalence gate's own transient
        // scratch buffers (its reference + speculative decode passes); it does not and must not
        // free the drafter's weight buffers, which stay referenced through this variable.
        retainedInCheckpointMTPDrafter = loadedDrafter.drafter
        Memory.clearCache()
        try validateScalarServingMemoryLimits(configuration)
    }

    let inference: InferenceActor
    switch decoderStrategy {
    case .compiledFP16:
        inference = InferenceActor(decoder: CompiledMLXDecoder(model: context.model))
    case .nativeCaches(let decision):
        // The same factory owns initial construction and every later request reset, for BOTH
        // `MLXDecoder` and (when MTP is selected) `MTPSpeculativeDecoder` below — a single
        // definition shared between the two branches so MTP can never silently revert KV storage
        // to the model's native fp16 cache while the operator selected a quantized tier. For fp16
        // it returns the model's native instances unchanged. If the quality gate later admits
        // dense int8, this already-built seam prevents reset from reverting quantized caches to
        // native fp16.
        let model = context.model
        let cacheFactory = scalarServingNativeCacheFactory(decision: decision, model: model)
        if let drafter = retainedInCheckpointMTPDrafter {
            // `MTPSpeculativeDecoder.init` throws (blockSize is pinned to
            // `MTPSpeculativeDecoder.servingBlockSize`, matching
            // `inCheckpointMTPStartupGateBlockSize` above) — propagated, never `try!`.
            inference = InferenceActor(
                decoder: try MTPSpeculativeDecoder(
                    target: model,
                    drafter: drafter,
                    cacheFactory: cacheFactory))
        } else {
            inference = InferenceActor(
                decoder: MLXDecoder(
                    model: model,
                    cacheFactory: cacheFactory))
        }
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
    // (`model_type`) AND the loaded checkpoint's own chat template, not the decoder route alone:
    // `.nativeHeterogeneous` is a cache-shape route that can carry more than one family, and a family
    // string alone does not prove a specific checkpoint's template actually emits the markers the
    // streaming splitter hardcodes. Two families are live-attested today: qwen3_5 (Qwen3.5/3.6/3.8,
    // 93e606a) and qwen4_exp (Flash Next, captured THIS cycle at harness c771346e — see
    // `StreamingReasoningPolicy.swift` for the full capture). Any other `.nativeHeterogeneous` family,
    // or an attested family whose loaded template does not attest the markers, conservatively keeps
    // the legacy `enable_thinking:false` workaround (dense/compiled always does, per QwenLM/Qwen3
    // #1817) and stays passthrough (non-separating) until it is live-captured.
    let observedFamilyModelType = scalarServingModelType(modelDirectory: configuration.modelDirectory)
    let templateAttestsThinkMarkers = scalarServingChatTemplateAttestsThinkMarkers(
        modelDirectory: configuration.modelDirectory)
    backendConfiguration.disableThinkingWhenToolsActive = servingDisablesThinkingWhenToolsActive(
        route: decoderRoute,
        modelType: observedFamilyModelType,
        templateAttestsThinkMarkers: templateAttestsThinkMarkers)
    backendConfiguration.thinksByDefault = servingThinksByDefault(
        route: decoderRoute,
        modelType: observedFamilyModelType,
        templateAttestsThinkMarkers: templateAttestsThinkMarkers)
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
        mlxPeakBytes: memory.peakMemory,
        inCheckpointMTPStartupVerdict: inCheckpointMTPStartupVerdict)
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

/// The literal markers `StreamingReasoningGate`/the streaming splitter hardcode when deciding whether
/// a stream carries a separable reasoning block.
private let scalarServingThinkMarkerAttestationStrings = ["<think>", "</think>"]

/// Artifact-derived attestation probe: does the LOADED checkpoint's own chat template contain the
/// `<think>`/`</think>` markers the streaming reasoning splitter hardcodes?
///
/// Resolves the same template Hugging Face tokenizers actually render from — `tokenizer_config.json`'s
/// `chat_template` string field takes precedence, falling back to a sibling `chat_template.jinja` file
/// only when `tokenizer_config.json` is missing/unreadable or does not carry that key. Returns `false`
/// (fail-closed) when neither source resolves to a template, or when the resolved template text does
/// not contain BOTH markers: an unreadable or non-attesting template must degrade
/// `servingThinksByDefault` to today's byte-identical passthrough, never promote it to separation,
/// because promoting content to `reasoning_content` on an unproven template is exactly the
/// answer-loss class `StreamingReasoningPolicy.swift`'s callers structurally avoid.
///
/// Captured live this cycle (harness `c771346e`, checkpoint `flashnext-oq4-mtp`): the Flash Next
/// checkpoint's `tokenizer_config.json` `chat_template` and its sibling `chat_template.jinja` are
/// byte-identical (8952 bytes), each containing `<think>` x3 and `</think>` x2 — either resolution
/// path attests the same markers for that checkpoint. The qwen3_5 incumbent's template carries the
/// identical marker profile, so this probe does not regress it.
///
/// Mirrors `scalarServingModelType`'s config-directory read pattern above; used to key
/// `servingThinksByDefault`/`servingDisablesThinkingWhenToolsActive` (`StreamingReasoningPolicy.swift`)
/// to the loaded artifact rather than a compile-time assumption about the family.
func scalarServingChatTemplateAttestsThinkMarkers(modelDirectory: URL) -> Bool {
    guard let template = scalarServingResolvedChatTemplateText(modelDirectory: modelDirectory) else {
        return false
    }
    return scalarServingThinkMarkerAttestationStrings.allSatisfy(template.contains)
}

/// Resolve the chat template text the tokenizer actually renders from, preferring
/// `tokenizer_config.json`'s `chat_template` field and falling back to a sibling
/// `chat_template.jinja` file only when that field is absent from a readable
/// `tokenizer_config.json`, or `tokenizer_config.json` itself is missing/unreadable.
private func scalarServingResolvedChatTemplateText(modelDirectory: URL) -> String? {
    let tokenizerConfigURL = modelDirectory.appendingPathComponent("tokenizer_config.json")
    if let data = try? Data(contentsOf: tokenizerConfigURL),
        let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
        let template = root["chat_template"] as? String
    {
        return template
    }
    let jinjaURL = modelDirectory.appendingPathComponent("chat_template.jinja")
    guard let data = try? Data(contentsOf: jinjaURL),
        let text = String(data: data, encoding: .utf8)
    else {
        return nil
    }
    return text
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

// MARK: - qwen4_exp in-checkpoint MTP drafter startup gate

/// Maps `ServingCore`'s MLX-free `FastMLXInCheckpointMTPNamespace` to `MLXLLM`'s own vendored
/// `InCheckpointMTPNamespaceSelection` runtime enum (the vendored MTP drafter facade module).
/// Exhaustive, no `default:` clause, so a future third namespace case in either enum is a compile
/// error here rather than a silent mismap. Mirrors `exactMTPRuntimeSelection`
/// (`fastmlx-serve/FastMLXServe.swift:975`).
func inCheckpointMTPRuntimeNamespace(
    _ namespace: FastMLXInCheckpointMTPNamespace
) -> InCheckpointMTPNamespaceSelection {
    switch namespace {
    case .official: .official
    case .converted: .converted
    }
}

/// Fixed pinned depth for the qwen4_exp in-checkpoint MTP drafter: the vendored draft model's own
/// `maximumBlockSize` is `3`, and the artifact preflight's own
/// `unsupportedBlockSize` guard requires this exact value (`QwenMTPArtifactPreflight.swift:411`).
/// Not a tunable — passing a larger value would silently clamp to this inside
/// `MTPSpeculativeTokenIterator.init` (`Swift.min(blockSize, drafter.maximumBlockSize ?? blockSize)`).
let inCheckpointMTPStartupGateBlockSize = 3

/// How many tokens the startup equivalence gate generates. This is an UPPER BOUND on decode work,
/// NOT a guarantee that at least one full speculative round runs: if the prepare-time bonus token
/// is itself a stop token, the decode loop breaks immediately after draining that bonus -- zero
/// rounds run, `drafter.draftBlock` is never called, `proposedDraftTokens == 0` -- because the
/// stop-token break dominates this budget regardless of its size. A startup prompt whose very
/// first generated token is a stop token will still fail clause (ii)'s `proposedDraftTokens > 0`
/// check no matter how large this value is.
let inCheckpointMTPStartupGateMaxTokens = 8

/// One greedy decode's tokens plus the speculative-decoding telemetry that run produced (zeroed
/// for the plain scalar reference, which has none).
struct InCheckpointMTPGreedyDecodeResult {
    let tokens: [Int]
    let proposedDraftTokens: Int
    let acceptedDraftTokens: Int
    let passthroughReason: String?
}

/// Successful result of `verifyInCheckpointMTPStartupEquivalence`.
struct InCheckpointMTPStartupEquivalence {
    let promptTokenCount: Int
    let generatedTokenCount: Int
    let proposedDraftTokens: Int
    let acceptedDraftTokens: Int
}

/// Fail-closed gate: the target's in-checkpoint MTP drafter, once loaded, must reproduce the SAME
/// greedy token sequence as scalar decode against the SAME target, AND
/// `MTPSpeculativeTokenIterator` must have genuinely speculated rather than silently degraded to
/// passthrough (`MTPSpeculativeTokenIterator.swift:167-176` documents exactly that silent
/// degradation). Without both checks the gate would pass vacuously by comparing scalar decode to
/// scalar decode — see
/// `docs/task-inbox/2026-09-07-qwen4exp-mtp-serving-wiring-DECISION.md`'s Increment B.
///
/// Protocol-typed (`any LanguageModel`/`any MTPDrafterModel`) rather than concretely bound to
/// `context.model`/the production vendored draft model, so this function is itself reusable
/// against a future tiny fixture target+drafter pair to exercise its SUCCESS path in a unit test.
/// No fixture on this repository's 24 GiB dev box is small enough to exercise that path against
/// the REAL Flash Next artifact today (the converted checkpoint is 113 GB) — that remains provable
/// only on the larger production-shaped hosts.
func verifyInCheckpointMTPStartupEquivalence(
    mainModel: any LanguageModel,
    drafter: any MTPDrafterModel,
    promptTokens: [Int],
    stopTokenIDs: Set<Int>,
    maxTokens: Int = inCheckpointMTPStartupGateMaxTokens,
    blockSize: Int = inCheckpointMTPStartupGateBlockSize
) throws -> InCheckpointMTPStartupEquivalence {
    let scalar = try runInCheckpointMTPScalarReference(
        mainModel: mainModel,
        promptTokens: promptTokens,
        stopTokenIDs: stopTokenIDs,
        maxTokens: maxTokens)
    let speculative = try runInCheckpointMTPSpeculativeDecode(
        mainModel: mainModel,
        drafter: drafter,
        promptTokens: promptTokens,
        stopTokenIDs: stopTokenIDs,
        maxTokens: maxTokens,
        blockSize: blockSize)

    return try inCheckpointMTPStartupEquivalenceDecision(
        promptTokenCount: promptTokens.count,
        scalar: scalar,
        speculative: speculative)
}

/// Pure pass/fail decision for the startup equivalence gate, given ALREADY-COMPUTED greedy decode
/// results from both paths. Separated from `verifyInCheckpointMTPStartupEquivalence` specifically so
/// both clauses — including clause (ii), the anti-vacuity clause — are unit-testable without a
/// real MLX model or drafter, mirroring `scalarServingMarkerFamilyAdmissionError`'s separation
/// from `loadScalarServingModel` for the identical reason.
func inCheckpointMTPStartupEquivalenceDecision(
    promptTokenCount: Int,
    scalar: InCheckpointMTPGreedyDecodeResult,
    speculative: InCheckpointMTPGreedyDecodeResult
) throws -> InCheckpointMTPStartupEquivalence {
    // Clause (i): identical greedy token sequences. Count first (a cheap, always-safe scalar
    // check), THEN a SHA-256 fold of the sequence — never `Array ==`, which triggers a Myers diff
    // on failure (`never-expect-equality-of-large-collections`).
    guard scalar.tokens.count == speculative.tokens.count else {
        throw ScalarServingModelLoadError.inCheckpointMTPStartupTokenSequenceMismatch
    }
    guard inCheckpointMTPTokenSequenceFingerprint(scalar.tokens)
        == inCheckpointMTPTokenSequenceFingerprint(speculative.tokens)
    else {
        throw ScalarServingModelLoadError.inCheckpointMTPStartupTokenSequenceMismatch
    }

    // Clause (ii), the anti-vacuity clause: the iterator must have genuinely speculated. Sticky
    // passthrough (`passthroughReason != nil`) means this run proved nothing about the drafter —
    // it degraded to (or never left) scalar decode. This check runs AFTER clause (i) passes, which
    // is exactly the vacuous-pass scenario it exists to catch: identical token sequences alone are
    // not evidence of speculation.
    guard speculative.passthroughReason == nil else {
        throw ScalarServingModelLoadError.inCheckpointMTPStartupDidNotSpeculate(
            reason: speculative.passthroughReason,
            proposedDraftTokens: speculative.proposedDraftTokens,
            acceptedDraftTokens: speculative.acceptedDraftTokens)
    }
    // The remaining fail-closed conjunct is `proposedDraftTokens > 0`, deliberately NOT
    // `acceptedDraftTokens > 0` (a prior, now-fixed production-availability bug). `proposedCount`
    // is incremented in `runInCheckpointMTPSpeculativeDecode` ONLY after a real
    // `drafter.draftBlock` call and a real target verify pass; every degradation path returns
    // before that line and sets `passthroughReason` first (guarded above). So there is no state
    // with `passthroughReason == nil && proposedDraftTokens > 0` in which `draftBlock` was not
    // actually invoked. Since `accepted > 0` implies `proposed > 0`, this conjunct is strictly NO
    // WEAKER than the accepted-based one at detecting "the drafter was never exercised" — it
    // states that property exactly, instead of conflating it with "the drafter's proposals were
    // accepted", which is a PERFORMANCE property, not an availability one.
    //
    // `MTPSpeculativeTokenIterator`'s greedy acceptance walk sets `accepted == 0` whenever the
    // drafter's FIRST proposal in a round differs from the target's own greedy token — the
    // iterator still emits the target's greedy token (so clause (i) above stays byte-exact) and
    // trims the rejected drafts from the cache. `accepted == 0` is a legitimate outcome of a
    // COMPLETED round by a CORRECT drafter, not evidence the drafter was never exercised. Gating
    // startup on `acceptedDraftTokens > 0` made that single, deterministic Bernoulli trial (this
    // gate runs one short, fixed startup prompt with `maxTokens: 8`, `blockSize: 3` — effectively
    // one speculative round) decide, permanently, whether a correctly loaded artifact could ever
    // boot with MTP enabled.
    //
    // Tradeoff, stated honestly: this gives up a weak, low-power signal that a mis-bound or
    // mis-quantized drafter would propose garbage that is never accepted. `acceptedDraftTokens`
    // remains REPORTED on the passing verdict below, so an operator still sees `accepted == 0` —
    // it is simply no longer treated as an availability failure.
    guard speculative.proposedDraftTokens > 0 else {
        throw ScalarServingModelLoadError.inCheckpointMTPStartupDidNotSpeculate(
            reason: nil,
            proposedDraftTokens: speculative.proposedDraftTokens,
            acceptedDraftTokens: speculative.acceptedDraftTokens)
    }

    return InCheckpointMTPStartupEquivalence(
        promptTokenCount: promptTokenCount,
        generatedTokenCount: speculative.tokens.count,
        proposedDraftTokens: speculative.proposedDraftTokens,
        acceptedDraftTokens: speculative.acceptedDraftTokens)
}

private func runInCheckpointMTPScalarReference(
    mainModel: any LanguageModel,
    promptTokens: [Int],
    stopTokenIDs: Set<Int>,
    maxTokens: Int
) throws -> InCheckpointMTPGreedyDecodeResult {
    let parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
    let cache = mainModel.newCache(parameters: parameters)
    var iterator = try TokenIterator(
        input: LMInput(tokens: MLXArray(promptTokens.map { Int32($0) })),
        model: mainModel,
        cache: cache,
        parameters: parameters)
    var tokens: [Int] = []
    // `nextThrowing()`, never `next()`: a target-side validation failure during this reference
    // pass would otherwise reach `TokenIterator`'s non-throwing entry point, which has no way to
    // report it and aborts the process instead. Switching only the speculative loop below would
    // leave this gate half-fixed — see
    // `docs/task-inbox/2026-09-07-mtp-decoder-bridge-DECISION.md`, "The same hazard exists on the
    // scalar reference iterator".
    while let token = try iterator.nextThrowing() {
        if stopTokenIDs.contains(token) {
            iterator.discardGeneratedToken()
            break
        }
        tokens.append(token)
        if tokens.count >= maxTokens {
            break
        }
    }
    return InCheckpointMTPGreedyDecodeResult(
        tokens: tokens, proposedDraftTokens: 0, acceptedDraftTokens: 0, passthroughReason: nil)
}

private func runInCheckpointMTPSpeculativeDecode(
    mainModel: any LanguageModel,
    drafter: any MTPDrafterModel,
    promptTokens: [Int],
    stopTokenIDs: Set<Int>,
    maxTokens: Int,
    blockSize: Int
) throws -> InCheckpointMTPGreedyDecodeResult {
    let parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
    var iterator = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray(promptTokens.map { Int32($0) })),
        mainModel: mainModel,
        drafter: drafter,
        mainCache: mainModel.newCache(parameters: parameters),
        parameters: parameters,
        blockSize: blockSize)
    var tokens: [Int] = []
    // Same reason as the reference loop above: the drafter/target pair validates cache and PLE
    // ownership on every forward, and this is the FIRST loop that drives it with real weights.
    // A crash here after a 113 GB load is far more expensive to diagnose than a thrown error.
    while let token = try iterator.nextThrowing() {
        if stopTokenIDs.contains(token) {
            iterator.discardGeneratedToken()
            break
        }
        tokens.append(token)
        if tokens.count >= maxTokens {
            break
        }
    }
    return InCheckpointMTPGreedyDecodeResult(
        tokens: tokens,
        proposedDraftTokens: iterator.proposedDraftTokens,
        acceptedDraftTokens: iterator.acceptedDraftTokens,
        passthroughReason: iterator.passthroughReason)
}

/// SHA-256 fold of a token-ID sequence, mirroring the harness's own `tokenIDsSHA256` precedent
/// (`fastmlx-harness/QwenMTPCorpusCLI.swift`) — a scalar (`String`) comparison, never an `Array ==`
/// on the raw token sequence.
private func inCheckpointMTPTokenSequenceFingerprint(_ tokens: [Int]) -> String {
    var data = Data()
    data.reserveCapacity(tokens.count * MemoryLayout<Int64>.size)
    for token in tokens {
        var value = Int64(token).littleEndian
        withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
    }
    let digest = SHA256.hash(data: data)
    return digest.map { String(format: "%02x", $0) }.joined()
}
