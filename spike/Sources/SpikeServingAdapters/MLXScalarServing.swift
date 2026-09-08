import CryptoKit
import Foundation

import Jinja
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

    /// Maps the wire `role` OpenAI decodes (`OpenAIMessageRole`, a closed enum kept faithful to
    /// the OpenAI Chat Completions API — see `OpenAIChatCompletionsTests.swift`'s pin on
    /// `.developer` surviving decode) onto the role vocabulary the served chat template actually
    /// branches on. No Qwen template (stock or patched) has a `developer` branch — only `system`,
    /// `user`, `assistant`, and `tool` — so a `developer` message rendered verbatim hits the
    /// template's own `raise_exception('Unexpected message role.')` and surfaces as an HTTP 400.
    /// `developer` is OpenAI's request-shape refinement of `system` (both carry steering
    /// instructions to the model); mapping it onto `system` here is a faithful translation, not a
    /// content change. Every other role passes through unchanged.
    private static func scalarServingTemplateRoleName(for role: OpenAIMessageRole) -> String {
        role == .developer ? OpenAIMessageRole.system.rawValue : role.rawValue
    }

    public func render(
        messages: [OpenAIChatMessage],
        tools: [OpenAIToolSpec],
        enableThinking: Bool?,
        reasoningEffort: String?
    ) throws -> [Int] {
        let templateMessages: [[String: any Sendable]] = messages.map { message in
            var dict: [String: any Sendable] = [
                "role": Self.scalarServingTemplateRoleName(for: message.role),
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
        do {
            return try tokenizer.applyChatTemplate(
                messages: templateMessages,
                tools: toolSpecs,
                additionalContext: additionalContext)
        } catch {
            // The served model ships its own chat template, and that template is the ONLY
            // authority on message-ordering/content constraints (e.g. "system must be first") —
            // engines differ per model (some patched templates accept a non-leading system
            // message), so this layer must not encode or duplicate the rule itself. It only
            // reports whatever the template decided, faithfully, as a client-shape error instead
            // of letting it fall through to the generic 500 catch-all.
            //
            // `scalarServingTemplateRoleName(for:)` above is a different act from this: it maps
            // the WIRE role vocabulary (OpenAI's `developer`/`system`/`user`/`assistant`/`tool`)
            // onto the template's own vocabulary before the template ever sees a message, the same
            // way this method already reshapes `tool_calls` into the template's expected shape
            // above and translates `enable_thinking`/`reasoning_effort` into template context
            // below. Encoding the template's CONSTRAINTS over that vocabulary (ordering, content
            // rules) is this catch block's job; translating the VOCABULARY itself, so the
            // constraints even have a chance to evaluate sensibly, is the role mapper's job — this
            // layer still does not decide what the template permits, only what the template can
            // recognize as input. Accepted residual: if a served template ever grows its own
            // `developer` branch, this mapping renders `developer` messages through the template's
            // `system` branch instead of that new branch — a fidelity loss (the template loses the
            // chance to treat `developer` specially), not a breakage, because every template that
            // has a `developer` branch also still accepts `system`.
            throw ServingChatTemplateRefusal.translated(error)
        }
    }

    public func makeDetokenizer() -> any ScalarServingDetokenizer {
        MLXScalarDetokenizer(tokenizer: tokenizer)
    }
}

/// Translates a Jinja chat-template's own `raise_exception(...)` refusal — surfaced to Swift as
/// `Jinja.TemplateException` — into a typed `OpenAIServingError.invalidRequestWithCode`, so the
/// HTTP layer reports it as a client request-shape error (400) instead of an opaque 500. Every
/// other error type (including a missing/unparseable template, which IS a server misconfiguration)
/// must pass through completely unchanged.
public enum ServingChatTemplateRefusal {
    /// Falls back to this when `TemplateException`'s message can't be recovered.
    static let genericMessage =
        "The model's chat template rejected this request."

    /// Returns `error` unchanged unless it is a `Jinja.TemplateException`, in which case it
    /// returns an `OpenAIServingError.invalidRequestWithCode` carrying the template's own
    /// refusal text.
    public static func translated(_ error: any Error) -> any Error {
        guard error is TemplateException else { return error }
        let detail = extractMessage(from: error) ?? genericMessage
        return OpenAIServingError.invalidRequestWithCode(
            "The model's chat template rejected this request: \(detail)",
            param: "messages",
            code: "chat_template_rejected")
    }

    /// `TemplateException.message` is internal (not visible outside the `Jinja` module), so the
    /// only way to recover the template author's own text is `Mirror` reflection over the
    /// struct's single stored property, matched by name (`"message"`) rather than parsed back out
    /// of a printed description — a substring parse over `String(describing:)` would mis-extract
    /// any template message that itself contains a quote character. The reflected child's value
    /// arrives as `Any` wrapping the property's declared type, `Optional<String>`; Swift's dynamic
    /// casts can present that either as a bare `String` (when populated) or as `String?` (either
    /// case), so both shapes are tried explicitly rather than assumed. This is deliberately
    /// defensive: any shape mismatch (no `message` child, absent message, empty message, or a
    /// future rename/reflection-format change upstream) falls back to `genericMessage` rather than
    /// crashing or silently losing the refusal.
    private static func extractMessage(from error: any Error) -> String? {
        guard let messageChild = Mirror(reflecting: error).children
            .first(where: { $0.label == "message" })
        else { return nil }
        let text: String?
        if let unwrapped = messageChild.value as? String {
            text = unwrapped
        } else if let optionallyWrapped = messageChild.value as? String? {
            text = optionallyWrapped
        } else {
            text = nil
        }
        guard let text, !text.isEmpty else { return nil }
        return text
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
    /// At least one of the model's native caches classified ONLY via the family-neutral
    /// `ServingCacheKindReporting` marker protocol (no concrete-type match), and the resolved family
    /// IS listed in `markerClassifiedFamiliesProvenOnlyViaResolvedOffloadedNGramPlan` — its serving
    /// proof is real — but this load did not resolve the offloaded n-gram plan the proof was
    /// captured on. Distinct from `unprovenServingFamily`: that case means the family has NO serving
    /// proof at all; this case means the family's ONLY proof is the offloaded n-gram route, and this
    /// specific load took the plain, fully-resident route instead. Carries the lowercased family
    /// name. Lifted by loading through the offloaded n-gram route (`--ngram-offload-plan`) rather
    /// than by any change to this family's proof status.
    case servingFamilyRequiresResolvedOffloadedNGramPlan(String)
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
    /// The token sequences matched, but the iterator did not GENUINELY speculate: either it
    /// engaged sticky passthrough (carried, when known, as the iterator's own
    /// `passthroughReason`) or `drafter.draftBlock` was never actually invoked
    /// (`proposedDraftTokens == 0`). Without this check the readiness gate would pass vacuously by
    /// comparing scalar decode to scalar decode -- `MTPSpeculativeTokenIterator` degrades to
    /// single-token passthrough SILENTLY rather than failing
    /// (`MTPSpeculativeTokenIterator.swift:167-176`), so a passing token-sequence comparison alone
    /// proves nothing about whether the drafter was ever exercised. Carries the observed telemetry
    /// (`proposedDraftTokens`, `acceptedDraftTokens`) alongside the iterator's `passthroughReason`
    /// so an operator hitting this sees what was measured, not just that the gate refused. NOTE:
    /// `acceptedDraftTokens == 0` alone is NOT what this case gates on -- see
    /// `inCheckpointMTPStartupReadinessDecision`'s clause (ii) comment for why gating startup on
    /// acceptance (rather than proposal) was a production-availability bug: a correct drafter whose
    /// first proposal for a round legitimately diverges from the target's greedy token completes
    /// that round with `accepted == 0`, and would previously never boot.
    ///
    /// This case, unlike clause (i)'s former `inCheckpointMTPStartupTokenSequenceMismatch` (removed
    /// -- see `docs/task-inbox/2026-09-07-mtp-scalar-route-divergence-DECISION.md`), remains
    /// fail-closed: a passthrough/never-speculated run proves NOTHING about the drafter, whereas a
    /// token-sequence divergence alone is expected architecture, not evidence of a broken pairing.
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
    /// `modelDirectory` and proves it at startup — see `verifyInCheckpointMTPStartupReadiness`.
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

/// Result of the startup-time readiness gate `verifyInCheckpointMTPStartupReadiness` runs when
/// `ScalarServingModelLoadConfiguration.inCheckpointMTPSelection` is supplied. GREEDY ONLY — this
/// proves `MTPSpeculativeTokenIterator` genuinely speculated (not sticky passthrough) against the
/// target; it says nothing about the SAMPLED (temperature != 0) generation path. It does NOT prove
/// the speculative route reproduces the target's own single-position scalar decode
/// token-for-token — see `drafterServing`'s doc comment.
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
    /// Whether the drafter was actually RETAINED and is serving requests, distinct from both "MTP
    /// was never requested" (`ScalarServingModelStartupReport.inCheckpointMTPStartupVerdict ==
    /// nil` — this verdict is never constructed at all in that case) and "drafter bound and
    /// serving". `true` when this run's scalar-reference and speculative-decode arms produced
    /// token-identical greedy sequences (clause (i)); `false` when they diverged, in which case
    /// `loadScalarServingModel` did NOT retain the drafter and this load fell back to the plain
    /// `MLXDecoder` scalar route instead. `false` is an EXPECTED, architecture-level outcome, not
    /// evidence of a broken pairing — see `inCheckpointMTPStartupGateMaxTokens`'s doc comment and
    /// `docs/task-inbox/2026-09-07-mtp-scalar-route-divergence-DECISION.md`: the speculative verify
    /// forward evaluates multiple positions at once while scalar decode evaluates one, so the two
    /// routes' forward geometries do not round identically past prefill. This verdict is only ever
    /// constructed once clause (ii) (the anti-vacuity, genuinely-fail-closed check) has already
    /// passed, so `drafterServing == false` here always means "diverged but genuinely speculated",
    /// never "never exercised" — that state still throws before this type is built.
    public let drafterServing: Bool
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
        drafterServing: Bool,
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
        self.drafterServing = drafterServing
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
    /// `in_checkpoint_mtp_drafter_serving` is what makes a divergence-refused load LOUD on this
    /// line -- see `drafterServing`'s doc comment for why this must never render identically to
    /// "MTP was never requested" (that case never reaches this function at all: the whole
    /// `in_checkpoint_mtp*` fragment is empty, per `FastMLXServe.swift`'s
    /// `report.inCheckpointMTPStartupVerdict?.machineReadableFields() ?? ""`).
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
            "in_checkpoint_mtp_drafter_serving=\(drafterServing)",
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
    /// `scalarServingChatTemplateRefusesNonLeadingSystemMessage`'s result for the loaded checkpoint,
    /// sampled at the same point `templateAttestsThinkMarkers` is. Default `true` matches that
    /// probe's own fail-closed value (see its doc comment) so a fixture that doesn't pass this
    /// explicitly reports the conservative "still refuses" state rather than a silently-safe one.
    public let chatTemplateRefusesNonLeadingSystemMessage: Bool

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
        inCheckpointMTPStartupVerdict: ScalarServingInCheckpointMTPStartupVerdict? = nil,
        chatTemplateRefusesNonLeadingSystemMessage: Bool = true
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
        self.chatTemplateRefusesNonLeadingSystemMessage = chatTemplateRefusesNonLeadingSystemMessage
    }

    /// Machine-readable startup-line fragment for the sampled MLX allocator bytes,
    /// using the startup line's snake_case convention. Mirrors the field names the
    /// continuous route publishes (`mlxActiveBytes`/`mlxCacheBytes`/`mlxPeakBytes`),
    /// so a KV cross-check can compare the two routes on identical keys.
    public var memoryFieldsFragment: String {
        "mlx_active_bytes=\(mlxActiveBytes) mlx_cache_bytes=\(mlxCacheBytes) "
            + "mlx_peak_bytes=\(mlxPeakBytes)"
    }

    /// Machine-readable startup-line fragment attesting whether the resolved chat template still
    /// contains the stock non-leading-system-message refusal anchor. Mirrors `memoryFieldsFragment`'s
    /// style/naming convention. See `scalarServingChatTemplateRefusesNonLeadingSystemMessage`'s doc
    /// comment for what `true`/`false` do and do not prove. Because `MLXScalarTextCodec.render`
    /// maps a wire `developer` role onto `system` before the template ever sees it
    /// (`scalarServingTemplateRoleName(for:)`), this attestation also silently governs non-leading
    /// `developer` messages — they arrive at the template as `system` and are subject to the exact
    /// same anchor this probe checks for.
    public var chatTemplateRefusesNonLeadingSystemMessageFragment: String {
        "chat_template_refuses_non_leading_system_message="
            + "\(chatTemplateRefusesNonLeadingSystemMessage)"
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
    //
    // UPDATED AGAIN (`docs/task-inbox/2026-09-07-mtp-scalar-route-divergence-DECISION.md`): this
    // gate's clause (i) (token-sequence identity between the scalar and speculative arms) no
    // longer refuses the whole serve on divergence — see `verifyInCheckpointMTPStartupReadiness`'s
    // doc comment for why identity was never a real property of this build. A diverged run still
    // succeeds here; `readiness.tokenSequencesMatched` decides below whether the drafter is
    // retained, and a THROWN error from this call still means clause (ii) (genuine speculation)
    // failed, which stays fail-closed exactly as before.
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
        // `decoderStrategy` is provably `.nativeCaches` here: `scalarServingInCheckpointMTPDecoderStrategyError`
        // above already refused any pairing of `inCheckpointMTPSelection` with `.compiledFP16`
        // before this block could ever run. Extracted via `if case` (never force-unwrapped) so
        // that invariant stays machine-checked rather than assumed, matching this file's existing
        // style — see that error function's own doc comment for why the combination is provably
        // unreachable for qwen4_exp today. Building the SAME `scalarServingNativeCacheFactory` the
        // decoder-strategy switch below builds (rather than a bare `newCache(parameters: nil)`
        // call local to the gate) makes the readiness gate exercise the operator's actual
        // KV-cache tier selection, not always plain native fp16.
        guard case .nativeCaches(let inCheckpointMTPGateCacheDecision) = decoderStrategy else {
            throw ScalarServingModelLoadError.inCheckpointMTPIncompatibleWithCompiledDecoderStrategy
        }
        let inCheckpointMTPGateCacheFactory = scalarServingNativeCacheFactory(
            decision: inCheckpointMTPGateCacheDecision, model: context.model)
        let readiness = try verifyInCheckpointMTPStartupReadiness(
            mainModel: context.model,
            drafter: loadedDrafter.drafter,
            promptTokens: startupPrompt,
            stopTokenIDs: stopTokenIDs,
            cacheFactory: inCheckpointMTPGateCacheFactory)
        // The SINGLE call to `inCheckpointMTPDrafterRetentionDecision` for this readiness result —
        // both `drafterServing` below and the `retainedInCheckpointMTPDrafter` assignment read
        // THIS `let`, never `readiness.tokenSequencesMatched` directly, so they cannot disagree by
        // reading two different values of the same underlying field — see that function's doc
        // comment for the mutation test that found the prior desync hazard, and for the residual
        // (smaller, single-line) way this `if` guard could still be edited to reintroduce it.
        let drafterRetentionDecision = inCheckpointMTPDrafterRetentionDecision(readiness: readiness)
        inCheckpointMTPStartupVerdict = ScalarServingInCheckpointMTPStartupVerdict(
            namespace: inCheckpointMTPSelection.namespace,
            revision: inCheckpointMTPSelection.revision,
            sourceKeyCount: loadedDrafter.sourceKeyCount,
            promptTokenCount: readiness.promptTokenCount,
            generatedTokenCount: readiness.generatedTokenCount,
            proposedDraftTokens: readiness.proposedDraftTokens,
            acceptedDraftTokens: readiness.acceptedDraftTokens,
            drafterServing: drafterRetentionDecision,
            drafterActiveBytesDelta: postDrafterMemory.activeMemory - preDrafterMemory.activeMemory,
            drafterCacheBytesDelta: postDrafterMemory.cacheMemory - preDrafterMemory.cacheMemory)
        // Retained across this block's closing brace — see the comment above this `if let` for
        // why. `Memory.clearCache()` below only reclaims the readiness gate's own transient
        // scratch buffers (its reference + speculative decode passes); it does not and must not
        // free the drafter's weight buffers, which stay referenced through this variable.
        //
        // ONLY retained when `drafterRetentionDecision` is true (clause (i) matched — see
        // `inCheckpointMTPDrafterRetentionDecision`'s doc comment). A divergent run already passed
        // clause (ii) (genuine speculation, still fail-closed above via `try`), so the drafter
        // itself is provably not broken — but this run gave no evidence the PAIRING is safe to
        // serve, so `retainedInCheckpointMTPDrafter` stays nil and the `if let drafter =` below
        // falls through to the plain `MLXDecoder` scalar route instead of refusing the whole
        // serve.
        if drafterRetentionDecision {
            retainedInCheckpointMTPDrafter = loadedDrafter.drafter
        }
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
    // Boot-time visibility for the deployment runbook's manual "drop a patched chat_template.jinja
    // into the model directory" step: sampled from the SAME resolved template
    // `templateAttestsThinkMarkers` above reads, so a live serve on a checkpoint whose template was
    // never (re-)patched shows it on the startup line instead of shipping silently — see
    // `scalarServingChatTemplateRefusesNonLeadingSystemMessage`'s doc comment.
    let chatTemplateRefusesNonLeadingSystemMessage =
        scalarServingChatTemplateRefusesNonLeadingSystemMessage(
            modelDirectory: configuration.modelDirectory)
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
        inCheckpointMTPStartupVerdict: inCheckpointMTPStartupVerdict,
        chatTemplateRefusesNonLeadingSystemMessage: chatTemplateRefusesNonLeadingSystemMessage)
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
/// Resolves the same template Hugging Face tokenizers actually render from. swift-transformers'
/// `Hub.swift` ("Check for chat template and merge if available") prefers a sibling
/// `chat_template.jinja` file when present and OVERWRITES `tokenizer_config.json`'s `chat_template`
/// field with its contents; this probe mirrors that precedence exactly, falling back to
/// `tokenizer_config.json`'s `chat_template` string field only when `chat_template.jinja` is
/// missing or unreadable. Returns `false`
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

/// The literal `raise_exception(...)` argument the stock (un-patched) template anchor uses to
/// refuse any non-leading `system` message. A checkpoint whose template still carries this anchor
/// has not had the permissive patch applied to it. See
/// `ServingChatTemplateRefusalTests.swift`'s `nonLeadingSystemGuardTemplate` fixture,
/// which excerpts this exact guard to provoke a genuine `Jinja.TemplateException` at request time.
/// This probe checks for the anchor's TEXT at load time, before any request is ever rendered.
private let scalarServingNonLeadingSystemMessageRefusalAnchor =
    "raise_exception('System message must be at the beginning.')"

/// Artifact-derived attestation probe: does the LOADED checkpoint's own chat template still contain
/// the stock refusal anchor that rejects any `system` message which is not first in the list?
///
/// Uses the SAME `scalarServingResolvedChatTemplateText` resolution `scalarServingChatTemplateAttestsThinkMarkers`
/// uses above (`chat_template.jinja` preferred, `tokenizer_config.json` fallback) — this probe exists
/// specifically to catch the case where the deployment runbook's "drop a patched `chat_template.jinja`
/// into the model directory" step was never (re-)applied: a live production-candidate serve shipped
/// on exactly that stale checkpoint once, undetected, because nothing attested the resolved template
/// at boot (`docs/task-inbox/` cycle-96 finding). Reading from a different (wrong-precedence) source
/// than the tokenizer actually renders from would silently defeat this probe's whole purpose, which
/// is why it shares the resolver this file already fixed to match `Hub.swift`.
///
/// Fail-closed shape: returns `true` (i.e. "still refuses", the unsafe/alerting state) when NO
/// template resolves at all. An unresolvable template is NOT evidence that an operator successfully
/// patched it — it is evidence of nothing — so this probe must not report `false` (which reads as
/// "confirmed patched, safe to serve non-leading system messages") on the strength of an absence.
/// This is the opposite fail-closed direction from `scalarServingChatTemplateAttestsThinkMarkers`
/// (which fails closed to `false` because THAT probe's `true` is the claim requiring positive
/// evidence, to avoid over-promoting streamed reasoning separation); here `false` is the claim
/// requiring positive evidence — "the template was read AND does not contain the stock anchor" — so
/// the unresolvable case must land on `true`, the conservative "assume still broken, keep warning"
/// outcome, not the silent-success outcome that let cycle 96 ship undetected.
///
/// This probe checks for the stock `system`-only anchor text; it says nothing directly about
/// `developer` messages. But `MLXScalarTextCodec.render` maps `developer` onto `system`
/// (`scalarServingTemplateRoleName(for:)`) before any request reaches the template, so a
/// non-leading `developer` message is refused (or accepted) by the exact same anchor this probe
/// attests — this function governs both roles even though its name only says `system`.
func scalarServingChatTemplateRefusesNonLeadingSystemMessage(modelDirectory: URL) -> Bool {
    guard let template = scalarServingResolvedChatTemplateText(modelDirectory: modelDirectory) else {
        return true
    }
    return template.contains(scalarServingNonLeadingSystemMessageRefusalAnchor)
}

/// Resolve the chat template text the tokenizer actually renders from, preferring a sibling
/// `chat_template.jinja` file and falling back to `tokenizer_config.json`'s `chat_template` field
/// only when `chat_template.jinja` is missing or unreadable.
///
/// This mirrors swift-transformers' own precedence, not the reverse: `Hub.swift`
/// ("Check for chat template and merge if available") prefers `chat_template.jinja` when it exists
/// and OVERWRITES `tokenizer_config.json`'s `chat_template` field with its contents before handing
/// the merged config to the tokenizer. Reading `tokenizer_config.json` first would attest whatever
/// checkpoint template shipped originally even after an operator drops a patched
/// `chat_template.jinja` alongside it — the tokenizer would render the patched file while this
/// probe kept reading the stale one.
private func scalarServingResolvedChatTemplateText(modelDirectory: URL) -> String? {
    let jinjaURL = modelDirectory.appendingPathComponent("chat_template.jinja")
    if let data = try? Data(contentsOf: jinjaURL),
        let text = String(data: data, encoding: .utf8)
    {
        return text
    }
    let tokenizerConfigURL = modelDirectory.appendingPathComponent("tokenizer_config.json")
    guard let data = try? Data(contentsOf: tokenizerConfigURL),
        let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
        let template = root["chat_template"] as? String
    else {
        return nil
    }
    return template
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
/// and the carried refusal string are both exact-match on the lowercased form. This gate has TWO
/// structurally distinct refusals, returned as two distinct error cases so an operator sees which
/// one applies:
///   1. The family is NOT in `markerClassifiedFamiliesProvenOnlyViaResolvedOffloadedNGramPlan` —
///      genuinely unsupported; refused unconditionally as `.unprovenServingFamily`.
///   2. The family IS listed (its serving proof is real), but `offloadedNGramPlanResolved` is
///      `false` for this load — refused, actionably, as
///      `.servingFamilyRequiresResolvedOffloadedNGramPlan`; a resolved offload plan is never a
///      blanket bypass for every marker-classified family, and a listed family is never admitted
///      on the plain, fully-resident load path.
/// Returns `nil` to admit.
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
        return .servingFamilyRequiresResolvedOffloadedNGramPlan(resolvedFamily)
    }
    return nil
}

/// Operator-facing announce line for `FastMLXServe.main`'s top-level catch of
/// `ScalarServingModelLoadError`. Without this, an error that survives to `main()` unwrapped traps
/// via Swift's top-level fatalError (exit 133, doubled message) instead of exiting cleanly with
/// exit(2) — see the sibling `catch let error as ServingModelCapabilitiesError` arm in
/// `FastMLXServe.swift`, whose comment names this exact failure mode.
///
/// Declared HERE, next to `ScalarServingModelLoadError`, rather than alongside
/// `scalarHybridFallbackAnnounceLine` in `ServingCore/ScalarServingCacheLayoutPolicy.swift`:
/// `ServingCore` has zero target dependencies (see `Package.swift`) and cannot see this type at
/// all. `SpikeServingAdapters` depends on `ServingCore`, never the reverse, so `ServingCore` is not
/// an option for a function whose signature names a `SpikeServingAdapters` type.
///
/// `.servingFamilyRequiresResolvedOffloadedNGramPlan` is the one case with a genuine, concrete
/// remedy (`--ngram-offload-plan`) and gets a fixed-key-order, machine-readable line naming both
/// the offending family and the flag — mirroring `scalarHybridFallbackAnnounceLine`'s
/// `key=value`-per-token shape. This REPLACES `scalarServingRemedyDescription`, which had zero
/// production callers (only tests) and so never actually reached an operator; there is now exactly
/// one source of this message, not two overlapping ones.
///
/// Every OTHER case renders a generic, honest line carrying the case's own description rather than
/// inventing a remedy that doesn't exist — an accurate raw dump beats a misleading bespoke reason.
/// That generic branch is deliberately NOT forced into single-token `key=value` fields: some cases
/// carry associated values with spaces (e.g. `memoryLimitNotApplied(expected: 1, observed: 2)`),
/// and mirrors the existing `ServingModelCapabilitiesError` catch arm's own idiom of appending a
/// free-text description after a fixed, parseable prefix rather than fabricating false structure.
public func scalarServingModelLoadRefusalAnnounceLine(
    _ error: ScalarServingModelLoadError
) -> String {
    switch error {
    case .servingFamilyRequiresResolvedOffloadedNGramPlan(let family):
        return "fastmlx-serve configuration=refused reason=serving_family_requires_offload_plan "
            + "model_type=\(family) remedy=--ngram-offload-plan"
    default:
        return "fastmlx-serve configuration=refused reason=scalar_serving_model_load_error "
            + "detail=\(error)"
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

/// How many tokens the startup readiness gate generates. This is an UPPER BOUND on decode work,
/// NOT a guarantee that at least one full speculative round runs: if the prepare-time bonus token
/// is itself a stop token, the decode loop breaks immediately after draining that bonus -- zero
/// rounds run, `drafter.draftBlock` is never called, `proposedDraftTokens == 0` -- because the
/// stop-token break dominates this budget regardless of its size. A startup prompt whose very
/// first generated token is a stop token will still fail clause (ii)'s `proposedDraftTokens > 0`
/// check no matter how large this value is.
///
/// DO NOT raise this to "make clause (i) discriminating" or to try to reach the point where the
/// scalar and speculative arms are expected to diverge. That divergence is ARCHITECTURAL, not a
/// bug this budget is hiding: the speculative verify forward evaluates multiple positions per
/// call while scalar decode evaluates one, so the two arms' forward geometries do not round
/// identically once decode runs long enough — measured, on real weights, to first appear near
/// token ~200 (`docs/task-inbox/2026-09-07-mtp-scalar-route-divergence-DECISION.md`). Raising this
/// constant would not fix that; it would turn clause (i) from "rarely reached" into "reliably
/// reached and reliably refuses a correct integration" -- a false-refusal machine, not a stronger
/// gate. Clause (i)'s outcome is DATA (`InCheckpointMTPStartupReadiness.tokenSequencesMatched`),
/// not a throw, precisely because of this.
let inCheckpointMTPStartupGateMaxTokens = 8

/// One greedy decode's tokens plus the speculative-decoding telemetry that run produced (zeroed
/// for the plain scalar reference, which has none).
struct InCheckpointMTPGreedyDecodeResult {
    let tokens: [Int]
    let proposedDraftTokens: Int
    let acceptedDraftTokens: Int
    let passthroughReason: String?
}

/// Successful result of `verifyInCheckpointMTPStartupReadiness`.
struct InCheckpointMTPStartupReadiness {
    let promptTokenCount: Int
    let generatedTokenCount: Int
    let proposedDraftTokens: Int
    let acceptedDraftTokens: Int
    /// Clause (i)'s outcome, carried as DATA rather than thrown: whether this run's scalar
    /// reference and speculative-decode arms produced token-identical greedy sequences.
    ///
    /// `false` is an EXPECTED, architecture-level outcome once decode runs long enough to reach
    /// it (see `inCheckpointMTPStartupGateMaxTokens`'s doc comment) — it is NOT by itself evidence
    /// of a broken drafter or a broken pairing. The call site
    /// (`loadScalarServingModel`) treats `false` as "do not retain the drafter for this serve",
    /// never as a reason to refuse the whole load: clause (ii) below is what stays fail-closed.
    let tokenSequencesMatched: Bool
}

/// Startup readiness gate for the qwen4_exp in-checkpoint MTP drafter. Runs the SAME target
/// through both a plain scalar-greedy reference decode and `MTPSpeculativeTokenIterator`, and
/// reports two independent things:
///
///   (i) whether the two arms' greedy token sequences matched on this run (DATA, in the returned
///       `InCheckpointMTPStartupReadiness.tokenSequencesMatched` — never thrown; see that field's
///       doc comment for why token-for-token identity is not a real property of this build), and
///   (ii) whether `MTPSpeculativeTokenIterator` genuinely speculated rather than silently
///        degrading to passthrough (`MTPSpeculativeTokenIterator.swift:167-176` documents exactly
///        that silent degradation) — this clause STAYS fail-closed (`throws`), because a
///        passthrough/never-speculated run proves NOTHING about the drafter, whereas a token
///        sequence divergence alone is expected geometry, not evidence of a broken pairing. See
///        `docs/task-inbox/2026-09-07-mtp-scalar-route-divergence-DECISION.md` for the real-weight
///        A/B that established this, and
///        `docs/task-inbox/2026-09-07-qwen4exp-mtp-serving-wiring-DECISION.md`'s Increment B for
///        why clause (ii) exists at all (without it this gate would pass vacuously by comparing
///        scalar decode to scalar decode).
///
/// Protocol-typed (`any LanguageModel`/`any MTPDrafterModel`) rather than concretely bound to
/// `context.model`/the production vendored draft model, so this function is itself reusable
/// against a future tiny fixture target+drafter pair to exercise its SUCCESS path in a unit test.
/// No fixture on this repository's 24 GiB dev box is small enough to exercise that path against
/// the REAL Flash Next artifact today (the converted checkpoint is 113 GB) — that remains provable
/// only on the larger production-shaped hosts.
///
/// `cacheFactory` builds BOTH arms' KV caches — the SAME factory `loadScalarServingModel` hands
/// `MLXDecoder`/`MTPSpeculativeDecoder` for real requests (`scalarServingNativeCacheFactory`),
/// rather than a bare `mainModel.newCache(parameters: nil)` call local to this gate. Without this
/// the gate would always construct plain native (fp16) caches regardless of the operator's
/// selected KV-cache storage tier, making it blind to a non-fp16 tier's own cache wrapper. No
/// default: every caller must be explicit about which factory it is proving.
func verifyInCheckpointMTPStartupReadiness(
    mainModel: any LanguageModel,
    drafter: any MTPDrafterModel,
    promptTokens: [Int],
    stopTokenIDs: Set<Int>,
    cacheFactory: () -> [KVCache],
    maxTokens: Int = inCheckpointMTPStartupGateMaxTokens,
    blockSize: Int = inCheckpointMTPStartupGateBlockSize
) throws -> InCheckpointMTPStartupReadiness {
    let scalar = try runInCheckpointMTPScalarReference(
        mainModel: mainModel,
        promptTokens: promptTokens,
        stopTokenIDs: stopTokenIDs,
        maxTokens: maxTokens,
        cacheFactory: cacheFactory)
    let speculative = try runInCheckpointMTPSpeculativeDecode(
        mainModel: mainModel,
        drafter: drafter,
        promptTokens: promptTokens,
        stopTokenIDs: stopTokenIDs,
        maxTokens: maxTokens,
        blockSize: blockSize,
        cacheFactory: cacheFactory)

    return try inCheckpointMTPStartupReadinessDecision(
        promptTokenCount: promptTokens.count,
        scalar: scalar,
        speculative: speculative)
}

/// Pure pass/fail decision for the startup readiness gate, given ALREADY-COMPUTED greedy decode
/// results from both paths. Separated from `verifyInCheckpointMTPStartupReadiness` specifically so
/// both clauses — including clause (ii), the anti-vacuity clause — are unit-testable without a
/// real MLX model or drafter, mirroring `scalarServingMarkerFamilyAdmissionError`'s separation
/// from `loadScalarServingModel` for the identical reason.
///
/// Order matters and is deliberate: clause (ii) is evaluated and can throw EVEN WHEN clause (i)
/// diverged. A run that both diverged (clause (i)) AND degraded to passthrough (clause (ii)) must
/// still throw `didNotSpeculate` — passthrough is an availability failure regardless of what
/// clause (i) measured, so clause (i)'s demotion to data must never swallow it.
func inCheckpointMTPStartupReadinessDecision(
    promptTokenCount: Int,
    scalar: InCheckpointMTPGreedyDecodeResult,
    speculative: InCheckpointMTPGreedyDecodeResult
) throws -> InCheckpointMTPStartupReadiness {
    // Clause (i): identical greedy token sequences -- now DATA, never thrown (see
    // `InCheckpointMTPStartupReadiness.tokenSequencesMatched`'s doc comment). Count first (a
    // cheap, always-safe scalar check), THEN a SHA-256 fold of the sequence — never `Array ==`,
    // which triggers a Myers diff on failure (`never-expect-equality-of-large-collections`).
    let tokenSequencesMatched =
        scalar.tokens.count == speculative.tokens.count
        && inCheckpointMTPTokenSequenceFingerprint(scalar.tokens)
            == inCheckpointMTPTokenSequenceFingerprint(speculative.tokens)

    // Clause (ii), the anti-vacuity clause: the iterator must have genuinely speculated. Sticky
    // passthrough (`passthroughReason != nil`) means this run proved nothing about the drafter —
    // it degraded to (or never left) scalar decode. Evaluated regardless of clause (i)'s outcome
    // above (see this function's own doc comment on ordering) — a diverged run gets no pass here
    // just because it diverged, and a matched run is not spared this check either: identical
    // token sequences alone are not evidence of speculation.
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
    // iterator still emits the target's greedy token and trims the rejected drafts from the
    // cache. `accepted == 0` is a legitimate outcome of a COMPLETED round by a CORRECT drafter,
    // not evidence the drafter was never exercised. Gating startup on `acceptedDraftTokens > 0`
    // made that single, deterministic Bernoulli trial (this gate runs one short, fixed startup
    // prompt with `maxTokens: 8`, `blockSize: 3` — effectively one speculative round) decide,
    // permanently, whether a correctly loaded artifact could ever boot with MTP enabled.
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

    return InCheckpointMTPStartupReadiness(
        promptTokenCount: promptTokenCount,
        generatedTokenCount: speculative.tokens.count,
        proposedDraftTokens: speculative.proposedDraftTokens,
        acceptedDraftTokens: speculative.acceptedDraftTokens,
        tokenSequencesMatched: tokenSequencesMatched)
}

/// Pure retention decision for the qwen4_exp in-checkpoint MTP drafter, given the startup
/// readiness gate's already-computed verdict. Separated from `loadScalarServingModel` specifically
/// so it is unit-testable without a real MLX model or drafter, mirroring
/// `inCheckpointMTPStartupReadinessDecision`'s and `scalarServingMarkerFamilyAdmissionError`'s
/// separation from that same function for the identical reason.
///
/// This is the ONE function both the retained drafter's presence
/// (`retainedInCheckpointMTPDrafter` in `loadScalarServingModel`) and the reported
/// `ScalarServingInCheckpointMTPStartupVerdict.drafterServing` value must derive from — a mutation
/// test caught the desync hazard directly: before this extraction, the two were independent reads
/// of `readiness.tokenSequencesMatched`, and mutating the retention call site to unconditionally
/// retain the drafter (while `drafterServing` still read the real field) left the existing suite
/// green, because nothing exercised the two reads' agreement. `loadScalarServingModel` now binds
/// this call's result to a single local `let` (`drafterRetentionDecision`) that both the
/// retention `if` and the verdict's `drafterServing:` argument read, which closes the ORIGINAL
/// hazard class — two independent reads of the same source silently drifting apart. It does NOT
/// make every desync structurally impossible: a future edit could still delete the retention `if`
/// guard and unconditionally assign `retainedInCheckpointMTPDrafter`, reintroducing the same
/// mismatch in a different shape. That residual edit is a single, visually obvious line in a
/// small function, not two field reads scattered ~15 lines apart — and, like the rest of this
/// gate, it is exercised only by the unit tests below at the pure-function boundary, never by
/// executing `loadScalarServingModel`'s own body (fleet-only; see
/// `MTPDecoderBridgeSelectionTests.swift`'s class doc comment).
func inCheckpointMTPDrafterRetentionDecision(readiness: InCheckpointMTPStartupReadiness) -> Bool {
    readiness.tokenSequencesMatched
}

private func runInCheckpointMTPScalarReference(
    mainModel: any LanguageModel,
    promptTokens: [Int],
    stopTokenIDs: Set<Int>,
    maxTokens: Int,
    cacheFactory: () -> [KVCache]
) throws -> InCheckpointMTPGreedyDecodeResult {
    var parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
    // Otherwise this silently inherits the vendored `GenerateParameters` default of 512
    // (`Evaluate.swift:134`) instead of the 2048 production actually chunks prefill at
    // (`MLXDecoder.defaultPrefillChunkSize`, threaded into `MTPSpeculativeDecoder.buildParameters`
    // at `MTPSpeculativeDecoder.swift:129`) — see the matching comment on the speculative arm
    // below for why a mismatch here is a real, previously-measured geometry divergence, not a
    // hypothetical one. HONESTLY: the fixed startup prompt this gate actually runs
    // (`ScalarServingModelLoadConfiguration.defaultStartupMessages`, ~15 tokens) is far below
    // EITHER chunk size, so this line has NO observable effect on today's startup run. It keeps
    // this gate's arms calibrated against what production actually does, for the day a longer
    // startup prompt (or a differently-sized chunk boundary) makes it observable.
    parameters.prefillStepSize = MLXDecoder.defaultPrefillChunkSize
    // Built through the SAME `cacheFactory` production hands the real decoders, not a bare
    // `mainModel.newCache(parameters:)` call — see `verifyInCheckpointMTPStartupReadiness`'s doc
    // comment on `cacheFactory` for why.
    let cache = cacheFactory()
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
    blockSize: Int,
    cacheFactory: () -> [KVCache]
) throws -> InCheckpointMTPGreedyDecodeResult {
    var parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
    // Same reasoning and same HONESTLY caveat as `runInCheckpointMTPScalarReference`'s identical
    // line above -- named constant, never a second `2048` literal.
    parameters.prefillStepSize = MLXDecoder.defaultPrefillChunkSize
    // Same `cacheFactory` reasoning as the scalar reference arm above -- both arms must be blind
    // to the operator's KV-cache tier selection in the SAME way production is, not differently.
    var iterator = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray(promptTokens.map { Int32($0) })),
        mainModel: mainModel,
        drafter: drafter,
        mainCache: cacheFactory(),
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
