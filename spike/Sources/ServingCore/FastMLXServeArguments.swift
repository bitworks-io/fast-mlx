import Foundation

public enum FastMLXServeBackend: Equatable, Sendable {
    case scripted
    // `memoryLimitBytes`/`cacheLimitBytes` are `Int?`: an omitted `--memory-limit-bytes` or
    // `--cache-limit-bytes` reaches here as `nil` (the operator-budget-envelope-shape decision) —
    // the sizer's derived figures are what actually apply; the flags were never load-bearing
    // requirements, only an optional tighter operator ask. `maxReservedKVBytes` is unaffected and
    // stays a required `Int` on the continuous routes.
    case scalar(
        modelDirectory: URL,
        memoryLimitBytes: Int?,
        cacheLimitBytes: Int?)
    case continuousBatchNoSpec(
        modelDirectory: URL,
        memoryLimitBytes: Int?,
        cacheLimitBytes: Int?,
        maxReservedKVBytes: Int)
    case continuousDynamicPLD(
        modelDirectory: URL,
        memoryLimitBytes: Int?,
        cacheLimitBytes: Int?,
        maxReservedKVBytes: Int)
}

public enum FastMLXServeHostUse: String, Equatable, Sendable {
    case shared
    case dedicatedServing = "dedicated-serving"
}

/// `--default-sampling`: whether a request that omits sampling parameters (notably `temperature`)
/// decodes greedy argmax (`off`, today's behavior, and the default) or is resolved against the
/// served checkpoint's own `generation_config.json` sampling subset
/// (`GenerationConfigSamplingDefaults.load(contentsOf:)`, `generation-config`). `generation-config`
/// is wired into the SCALAR serve route only (`loadScalarServingModel`,
/// `spike/Sources/SpikeServingAdapters/MLXScalarServing.swift`), fail-closed at load -- a
/// missing/unparseable/non-sampling artifact, or a resolved `.compiledFP16` decoder strategy,
/// refuses to start rather than silently falling back to greedy. See the refusals below, which
/// exist because turning this on converts param-less traffic into SAMPLED traffic, which every
/// OTHER route either cannot honor or has not proven safe for.
public enum FastMLXServeDefaultSampling: String, Equatable, Sendable {
    case off
    case generationConfig = "generation-config"
}

public enum FastMLXExactMTPSelection: String, Equatable, Sendable {
    case qwen35_9BDepth1 = "qwen35-9b-depth1"
    case qwen38_27BMXFP8Depth1 = "qwen38-27b-mxfp8-depth1"
    case qwen38_27B4BitDepth1 = "qwen38-27b-4bit-depth1"
}

/// MLX-free mirror of `MLXLLM`'s own in-checkpoint MTP namespace-selection enum (the vendored
/// MTP drafter facade module) -- `ServingCore` has zero dependencies (see this target's
/// entry in `Package.swift`), so it cannot import MLXLLM and name that type directly.
/// `SpikeServingAdapters` maps this to the runtime enum with an exhaustive switch, mirroring
/// `FastMLXExactMTPSelection` -> `Qwen35ExactMTPRuntimeSelection`
/// (`exactMTPRuntimeSelection`, `FastMLXServe.swift:975`).
public enum FastMLXInCheckpointMTPNamespace: String, Equatable, Sendable, CaseIterable {
    /// Raw MTP source keys prefixed `mtp.` -- the official BF16 artifact's layout.
    case official
    /// Raw MTP source keys prefixed `language_model.mtp.` -- the converted/quantized artifact's
    /// layout.
    case converted
}

/// Selects which in-checkpoint Qwen4-Exp (Flash Next, `qwen4_exp`) MTP drafter artifact
/// `loadScalarServingModel` should load from the served target's own checkpoint directory and
/// gate at startup. Mirrors `FastMLXExactMTPSelection`'s shape -- a pinned, per-artifact
/// deployment-policy row joined to the `MLXLLM` runtime layout enum by a mapping function in
/// `SpikeServingAdapters` -- because a checkpoint's namespace is a LAYOUT fact (belongs in the
/// vendored module) while which artifact to serve is DEPLOYMENT POLICY (belongs here); see
/// `docs/task-inbox/2026-09-07-qwen4exp-mtp-serving-wiring-DECISION.md`.
///
/// ONLY the converted 4-bit artifact is serve-eligible. `Qwen/Qwen3.8-Flash-Next` (the official
/// BF16 artifact) is 360,023,351,514 bytes and fits no host this project owns -- the largest is
/// 256 GB (~178.7 GiB at the 75% shared ceiling). Because this drafter loads FROM
/// INSIDE the target checkpoint's own shard set, not a separate drafter directory, selecting the
/// official artifact would mean serving the 360 GB target itself, not merely a larger drafter.
/// That artifact stays loader-level only -- already tested inside the vendored in-checkpoint MTP
/// drafter loader's own `mtpSourceKeys(in:namespace:)` -- and is deliberately not a case here.
public enum FastMLXInCheckpointMTPSelection: String, Equatable, Sendable, CaseIterable {
    case converted4Bit = "converted-4bit"

    /// The checkpoint layout this selection's artifact carries.
    public var namespace: FastMLXInCheckpointMTPNamespace {
        switch self {
        case .converted4Bit: .converted
        }
    }

    /// The MTP source-key count `Vontra/Qwen3.8-Flash-Next-MLX-oQ4-MTP`'s own index declares
    /// under the `converted` namespace's prefix, independently verified (35 files /
    /// 113,348,682,948 bytes, `revision_match=true` -- see the decision doc above).
    public var expectedSourceKeyCount: Int {
        switch self {
        case .converted4Bit: 76
        }
    }

    /// The pinned upstream revision this artifact's `expectedSourceKeyCount` was verified against.
    public var revision: String {
        switch self {
        case .converted4Bit: "43a82b3f0ff64fa417fd09ca046580f08d19b0d6"
        }
    }
}

public enum FastMLXServeArgumentError:
    Error, Equatable, CustomStringConvertible, Sendable
{
    case unknownArgument(String)
    case duplicateOption(String)
    case missingValue(String)
    case invalidPort
    case invalidPositiveInteger(String)
    case missingBackendMode
    case conflictingBackendModes
    case missingRequiredOption(String)
    case invalidModelIdentifier
    case modelPathMustBeAbsolute
    case evidencePathMustBeAbsolute
    case cacheLimitExceedsMemoryLimit
    case reservedKVLimitExceedsMemoryLimit
    case optionRequiresContinuousBatchMode(String)
    case quantCandidatesWithModelPath
    case quantCandidateMustBeAbsolute
    case quantReliabilityPathMustBeAbsolute
    case kvQuantWithScripted
    case allowHybridWithScripted
    case autoQuantWithCandidates
    case autoQuantRequiresPickOnly
    case invalidAutoQuantBase
    case mtpDrafterPathMustBeAbsolute
    case mtpDrafterRequiresExactQwen35MTP
    case ngramOffloadPlanMustBeAbsolute
    case ngramOffloadPlanWithContinuousBatch
    case ngramOffloadPlanWithExactQwen35MTP
    case ngramOffloadPlanWithQuantPickOnly
    case ngramOffloadPlanWithQuantCandidates
    case invalidExactMTPSelection
    case exactMTPSelectionRequiresExactQwen35MTP
    case exactQwen35MTPWithScripted
    case exactQwen35MTPWithContinuousBatch
    case exactQwen35MTPWithQuantSource
    case dynamicPLDWithHybridQwen35
    case invalidHostUse
    case osServiceReserveRequiresDedicatedServing
    case invalidCompletionLimitPolicy
    case defaultCompletionTokensExceedsMaximumCompletionTokens
    /// `--qwen4exp-mtp` without `--ngram-offload-plan`: the marker-family admission gate
    /// (`scalarServingMarkerFamilyAdmissionError` in `SpikeServingAdapters/MLXScalarServing.swift`)
    /// admits a marker-classified family only when `offloadedNGramPlanResolved` is true, so this
    /// pairing would otherwise spend minutes loading weights and then throw a confusing
    /// `unprovenServingFamily` error deep in the load path. Refuse at parse time instead.
    case qwen4ExpMTPRequiresNGramOffloadPlan
    /// `--qwen4exp-mtp` with `--scripted`: the transport-only scripted backend loads no model, so
    /// the flag would be silently dropped. `hasLoadedModelOptions` does NOT include this flag (it
    /// only requires `--ngram-offload-plan`, which is not itself a "loaded model option" flag
    /// either), so the existing `conflictingBackendModes` check does not cover this combination —
    /// an explicit refusal is required.
    case qwen4ExpMTPWithScripted
    /// `--ngram-offload-plan` with `--scripted`: the flag is consumed only at the scalar-load call
    /// site (`loadScalarServingBackend` → `ScalarServingModelLoadConfiguration`); the transport-only
    /// scripted backend loads no model and never reaches that seam, so the plan would be silently
    /// dropped. `hasLoadedModelOptions` deliberately does NOT include this flag (the same set the
    /// `qwen4ExpMTPWithScripted` doc comment above describes), so the existing
    /// `conflictingBackendModes` check does not cover this combination — an explicit refusal is
    /// required.
    case ngramOffloadPlanWithScripted
    /// `--chat-template` was supplied but the value does not begin with `/` — mirrors
    /// `ngramOffloadPlanMustBeAbsolute`'s guard exactly (same shape, same rationale: an absolute
    /// local path is required so the resolved file is unambiguous regardless of the process's
    /// working directory).
    case chatTemplateMustBeAbsolute
    /// `--chat-template` is consumed only at the scalar-load call site
    /// (`ScalarServingModelLoadConfiguration.chatTemplateOverrideURL`); the transport-only
    /// scripted backend loads no model, so the flag would be silently dropped.
    case chatTemplateWithScripted
    /// `--chat-template` is consumed only at the scalar-load call site; neither continuous route
    /// constructs its tokenizer through that seam, so the flag would be silently dropped.
    case chatTemplateWithContinuousBatch
    /// `--chat-template` is consumed only at the scalar-load call site; the exact Qwen3.5 MTP
    /// composition constructs its own tokenizer separately and does not reach that seam, so the
    /// flag would be silently dropped.
    case chatTemplateWithExactQwen35MTP
    /// `--quant-pick-only` returns early below without threading `chatTemplateURL` at all — no
    /// model is loaded on that dry-run path, so the flag would be silently dropped.
    case chatTemplateWithQuantPickOnly
    /// `--fit-check-only` and `--quant-pick-only` are two different dry runs -- a single-model
    /// fit-check vs. an auto-pick across several `--quant-candidates` directories -- so combining
    /// them is ambiguous about which dry run the operator wants. Unlike `--chat-template` and the
    /// other flags `--quant-pick-only`'s early return drops, `--fit-check-only` is NOT threaded
    /// through that early-return construction at all (see the parser's own guard just above the
    /// early return): it stays on the full loaded-model construction path so it reaches the real
    /// `resolveServingLimits` call sites unchanged, so this is a genuine ambiguity refusal, not a
    /// silent-drop guard.
    case fitCheckOnlyWithQuantPickOnly
    /// `--force` exists to proceed past a RED fit-check verdict; `--fit-check-only` exists to LEARN
    /// that verdict. Combining them would let a red host report a `--fit-check-only` "success" by
    /// suppressing the very signal the dry run exists to surface -- fail closed instead.
    case fitCheckOnlyWithForce
    /// `--fit-check-only` reports the verdict `resolveServingLimits` computes for a loaded model;
    /// the transport-only `--scripted` backend loads no model and has no model directory to check,
    /// so the flag would have nothing to report.
    case fitCheckOnlyWithScripted
    /// `--offload-plan-check-only` and `--fit-check-only` are two different dry runs stopping at
    /// two different, non-confusable points: "the declared budget arithmetic fits" versus "this
    /// host's offload artifacts resolve and verify". Composing them is refused rather than given a
    /// silent precedence, mirroring `fitCheckOnlyWithQuantPickOnly`'s identical rationale.
    case offloadPlanCheckOnlyWithFitCheckOnly
    /// `--force` exists to proceed past a refused verdict; `--offload-plan-check-only` exists to
    /// LEARN that verdict. Combining them would let a host with unresolvable offload artifacts
    /// report an `--offload-plan-check-only` "success" by suppressing the very refusal the dry run
    /// exists to surface -- mirrors `fitCheckOnlyWithForce`'s identical rationale exactly.
    case offloadPlanCheckOnlyWithForce
    /// `--offload-plan-check-only` verifies the offloaded n-gram plan the SAME way the real load
    /// resolves it (see `qwen4ExpMTPRequiresNGramOffloadPlan`'s identical shape), so it is
    /// meaningless without a plan to check. This requirement also gives the flag TRANSITIVE
    /// coverage against continuous batching, `--exact-qwen35-mtp`, `--quant-pick-only`,
    /// `--quant-candidates`, and `--scripted`: each of those is already refused above whenever
    /// `--ngram-offload-plan` is present, so a separate `--offload-plan-check-only`-specific
    /// refusal for any of them would be unreachable dead code.
    case offloadPlanCheckOnlyRequiresNGramOffloadPlan
    /// `--qwen4exp-sampled-mtp` without `--qwen4exp-mtp`: the sampled block-decision provider this
    /// flag constructs plugs into the in-checkpoint MTP drafter's own iterator
    /// (`MTPSpeculativeDecoder`'s `sampledBlockDecisionsEnabled`), which only `--qwen4exp-mtp`
    /// loads. Without that drafter there is no iterator to hand the provider to, so the flag would
    /// be silently meaningless rather than merely unused. Refuse at parse time instead, mirroring
    /// `mtpDrafterRequiresExactQwen35MTP`'s identical "valued flag requires the feature it
    /// modifies" shape. `--qwen4exp-mtp` itself already requires `--ngram-offload-plan` (see
    /// `qwen4ExpMTPRequiresNGramOffloadPlan`'s doc comment), so this requirement gives
    /// `--qwen4exp-sampled-mtp` the SAME transitive coverage against continuous batching,
    /// `--exact-qwen35-mtp`, `--quant-pick-only`, `--quant-candidates`, and `--scripted` that
    /// `--qwen4exp-mtp` already gets — a separate `--qwen4exp-sampled-mtp`-specific refusal for any
    /// of those would be unreachable dead code.
    case sampledMTPRequiresInCheckpointMTP
    case invalidDefaultSampling
    /// `--default-sampling generation-config` resolves artifact-sourced sampling defaults for a
    /// param-less request; `--scripted` is a transport-only dry run that never constructs a
    /// sampling request at all, so the flag would be silently inert rather than merely unused.
    case defaultSamplingWithScripted
    /// `--default-sampling generation-config` would apply to a served request; `--quant-pick-only`
    /// is a dry-run return path that resolves a pick and exits without ever serving a request, so
    /// the flag would be silently inert there too.
    case defaultSamplingWithQuantPickOnly
    /// `ContinuousServingBackend` rejects `.sampled` with HTTP 400 (see its own admission check),
    /// so turning param-less traffic into sampled traffic on a continuous route would convert
    /// every affected request into a hard failure rather than silently ignoring the flag -- refuse
    /// at parse time instead, mirroring `chatTemplateWithContinuousBatch` /
    /// `ngramOffloadPlanWithContinuousBatch`'s identical "the flag would be silently
    /// dropped/ignored" shape.
    case defaultSamplingWithContinuousBatch
    /// The exact Qwen3.5 MTP route falls back to scalar decoding on any non-greedy sampling
    /// (`ExactQwen35MTPServingAdmission` -> `.scalarFallback(.sampledGeneration)`), so enabling
    /// artifact-sourced defaults there would silently drop every param-less request off the MTP
    /// path onto the scalar fallback instead of accelerating it.
    case defaultSamplingWithExactQwen35MTP
    /// `--qwen4exp-mtp` without `--qwen4exp-sampled-mtp`: the in-checkpoint MTP drafter only
    /// accelerates GREEDY requests without the sampled block-decision provider
    /// (`--qwen4exp-sampled-mtp`). Turning on artifact-sourced defaults would move every
    /// param-less request from accelerated greedy MTP to unaccelerated scalar decode -- a
    /// throughput cliff, not a silent drop -- so this combination is refused unless
    /// `--qwen4exp-sampled-mtp` is also present.
    case defaultSamplingWithInCheckpointMTPRequiresSampledMTP

    public var description: String {
        switch self {
        case .unknownArgument(let argument):
            "Unknown argument: \(argument)"
        case .duplicateOption(let option):
            "\(option) may be specified only once"
        case .missingValue(let option):
            "\(option) requires a value"
        case .invalidPort:
            "--port must be an integer from 0 through 65535"
        case .invalidPositiveInteger(let option):
            "\(option) must be a positive integer"
        case .missingBackendMode:
            "Choose --scripted or provide --model-path with explicit model limits"
        case .conflictingBackendModes:
            "--scripted cannot be combined with loaded-model options"
        case .missingRequiredOption(let option):
            "Loaded model serving requires \(option)"
        case .invalidModelIdentifier:
            "--model must be a non-empty identifier"
        case .modelPathMustBeAbsolute:
            "--model-path must be an absolute local path"
        case .evidencePathMustBeAbsolute:
            "--evidence-path must be an absolute local path"
        case .cacheLimitExceedsMemoryLimit:
            "--cache-limit-bytes cannot exceed --memory-limit-bytes"
        case .reservedKVLimitExceedsMemoryLimit:
            "--max-reserved-kv-bytes cannot exceed --memory-limit-bytes"
        case .optionRequiresContinuousBatchMode(let option):
            "\(option) requires --continuous-batch-no-spec"
        case .quantCandidatesWithModelPath:
            "--quant-candidates cannot be combined with --model-path"
        case .quantCandidateMustBeAbsolute:
            "--quant-candidates entries must be non-empty absolute local paths, comma-separated"
        case .quantReliabilityPathMustBeAbsolute:
            "--quant-reliability must be an absolute local path"
        case .kvQuantWithScripted:
            "--kv-quant applies to a loaded model and cannot be combined with --scripted"
        case .allowHybridWithScripted:
            "--allow-hybrid-qwen35 admits a model onto the continuous route and cannot be "
                + "combined with --scripted"
        case .autoQuantWithCandidates:
            "--auto-quant and --quant-candidates are alternative quant sources; use one, not both"
        case .autoQuantRequiresPickOnly:
            "--auto-quant currently requires --quant-pick-only (the network probe/download half is "
                + "not built yet; it enumerates candidate repo names offline)"
        case .invalidAutoQuantBase:
            "--auto-quant requires a non-empty base repository identifier"
        case .mtpDrafterPathMustBeAbsolute:
            "--mtp-drafter-path must be an absolute local path"
        case .mtpDrafterRequiresExactQwen35MTP:
            "--mtp-drafter-path requires --exact-qwen35-mtp"
        case .ngramOffloadPlanMustBeAbsolute:
            "--ngram-offload-plan must be an absolute local path"
        case .ngramOffloadPlanWithContinuousBatch:
            "--ngram-offload-plan is not supported with continuous batching"
        case .ngramOffloadPlanWithExactQwen35MTP:
            "--ngram-offload-plan is not supported with --exact-qwen35-mtp"
        case .ngramOffloadPlanWithQuantPickOnly:
            "--ngram-offload-plan is not supported with --quant-pick-only"
        case .ngramOffloadPlanWithQuantCandidates:
            "--ngram-offload-plan is sealed against one specific artifact and cannot be combined "
                + "with --quant-candidates auto-pick across several candidate directories; pass the "
                + "single sealed model directory explicitly via --model-path instead"
        case .invalidExactMTPSelection:
            "--exact-mtp-selection must be qwen35-9b-depth1, qwen38-27b-mxfp8-depth1, "
                + "or qwen38-27b-4bit-depth1"
        case .exactMTPSelectionRequiresExactQwen35MTP:
            "--exact-mtp-selection requires --exact-qwen35-mtp"
        case .exactQwen35MTPWithScripted:
            "--exact-qwen35-mtp loads a model and cannot be combined with --scripted"
        case .exactQwen35MTPWithContinuousBatch:
            "--exact-qwen35-mtp is a scalar-fallback exact route and cannot be combined with "
                + "a continuous serving mode"
        case .exactQwen35MTPWithQuantSource:
            "--exact-qwen35-mtp requires explicit local --model-path/--mtp-drafter-path snapshots "
                + "and cannot be combined with quant selection flags"
        case .dynamicPLDWithHybridQwen35:
            "--continuous-dynamic-pld is admitted only for the exact dense Qwen3 policy and "
                + "cannot be combined with --allow-hybrid-qwen35"
        case .invalidHostUse:
            "--host-use must be shared or dedicated-serving"
        case .osServiceReserveRequiresDedicatedServing:
            "--os-service-reserve-bytes requires --host-use dedicated-serving"
        case .invalidCompletionLimitPolicy:
            "--completion-limit-policy must be reject or clamp"
        case .defaultCompletionTokensExceedsMaximumCompletionTokens:
            "--default-completion-tokens cannot exceed an explicit --max-completion-tokens"
        case .qwen4ExpMTPRequiresNGramOffloadPlan:
            "--qwen4exp-mtp requires --ngram-offload-plan"
        case .qwen4ExpMTPWithScripted:
            "--qwen4exp-mtp loads a model and cannot be combined with --scripted"
        case .ngramOffloadPlanWithScripted:
            "--ngram-offload-plan loads a model and cannot be combined with --scripted"
        case .chatTemplateMustBeAbsolute:
            "--chat-template must be an absolute local path"
        case .chatTemplateWithScripted:
            "--chat-template loads a model and cannot be combined with --scripted"
        case .chatTemplateWithContinuousBatch:
            "--chat-template is not supported with continuous batching"
        case .chatTemplateWithExactQwen35MTP:
            "--chat-template is not supported with --exact-qwen35-mtp"
        case .chatTemplateWithQuantPickOnly:
            "--chat-template is not supported with --quant-pick-only"
        case .fitCheckOnlyWithQuantPickOnly:
            "--fit-check-only and --quant-pick-only are alternative dry runs; use one, not both"
        case .fitCheckOnlyWithForce:
            "--fit-check-only cannot be combined with --force, which would suppress the verdict "
                + "the dry run exists to learn"
        case .fitCheckOnlyWithScripted:
            "--fit-check-only reports a loaded-model fit verdict and cannot be combined with "
                + "--scripted"
        case .offloadPlanCheckOnlyWithFitCheckOnly:
            "--offload-plan-check-only and --fit-check-only are two different dry runs with "
                + "different stop points; use one, not both"
        case .offloadPlanCheckOnlyWithForce:
            "--offload-plan-check-only cannot be combined with --force, which would suppress the "
                + "verdict the dry run exists to learn"
        case .offloadPlanCheckOnlyRequiresNGramOffloadPlan:
            "--offload-plan-check-only requires --ngram-offload-plan"
        case .sampledMTPRequiresInCheckpointMTP:
            "--qwen4exp-sampled-mtp requires --qwen4exp-mtp"
        case .invalidDefaultSampling:
            "--default-sampling must be off or generation-config"
        case .defaultSamplingWithScripted:
            "--default-sampling generation-config resolves artifact-sourced sampling defaults for "
                + "a served request and cannot be combined with --scripted, which loads no model "
                + "and serves nothing"
        case .defaultSamplingWithQuantPickOnly:
            "--default-sampling generation-config resolves artifact-sourced sampling defaults for "
                + "a served request and cannot be combined with --quant-pick-only, whose dry-run "
                + "return path exits without ever serving one"
        case .defaultSamplingWithContinuousBatch:
            "--default-sampling generation-config is not supported with continuous batching: "
                + "ContinuousServingBackend rejects a sampled request with HTTP 400"
        case .defaultSamplingWithExactQwen35MTP:
            "--default-sampling generation-config is not supported with --exact-qwen35-mtp: that "
                + "route falls back to scalar decoding on any non-greedy sampling, so every "
                + "param-less request would silently drop off the MTP path"
        case .defaultSamplingWithInCheckpointMTPRequiresSampledMTP:
            "--default-sampling generation-config requires --qwen4exp-sampled-mtp when "
                + "--qwen4exp-mtp is enabled: without it every param-less request would move from "
                + "accelerated greedy MTP to unaccelerated scalar decode"
        }
    }
}

/// Operator-facing announce line for `FastMLXServe.main`'s top-level catch of
/// `FastMLXServeArgumentError`. Without this, an argument-validation refusal thrown by
/// `FastMLXServeArguments.parse` — the very FIRST statement in `run()`, so this is the earliest
/// possible refusal in the whole process — survives to the top level unwrapped and traps via
/// Swift's top-level fatalError (exit 133, doubled message) instead of exiting cleanly with
/// exit(2); see the sibling `catch let error as ScalarServingModelLoadError` arm in
/// `FastMLXServe.swift`, whose comment names this exact failure mode, and
/// `scalarServingModelLoadRefusalAnnounceLine` in
/// `SpikeServingAdapters/MLXScalarServing.swift`, whose conventions this mirrors.
///
/// Declared HERE, next to `FastMLXServeArgumentError` itself, rather than in `SpikeServingAdapters`
/// alongside `scalarServingModelLoadRefusalAnnounceLine`: `FastMLXServeArgumentError` already lives
/// in `ServingCore` and this function names no type outside it, so — unlike that sibling, whose
/// error type lives in `SpikeServingAdapters` — nothing forces this one out of `ServingCore`.
///
/// Renders the WHOLE 94-case `FastMLXServeArgumentError` type with one honest generic line rather
/// than per-case bespoke `reason=` tokens: every case's own `description` already names the
/// offending flag and the concrete violation (e.g. "--chat-template is not supported with
/// continuous batching"), so a fixed `reason=invalid_arguments` prefix plus that description in
/// `detail=` is accurate and complete on its own — mirroring
/// `scalarServingModelLoadRefusalAnnounceLine`'s `default` branch, which renders its own remaining
/// ~28 cases the same way rather than inventing structure the description doesn't need.
public func fastMLXServeArgumentRefusalAnnounceLine(_ error: FastMLXServeArgumentError) -> String {
    "fastmlx-serve configuration=refused reason=invalid_arguments detail=\(error.description)"
}

public struct FastMLXServeArguments: Equatable, Sendable {
    public static let usage = """
        Usage:
          fastmlx-serve --scripted [--host HOST] [--port PORT] [--model MODEL]
          fastmlx-serve [--continuous-batch-no-spec | --continuous-dynamic-pld]
            --model-path PATH --model MODEL
            --memory-limit-bytes N --cache-limit-bytes N
            [--max-reserved-kv-bytes N]
            [--host HOST] [--port PORT] [--evidence-path PATH]

          --scripted                  Transport-only backend; no model is loaded.
          --continuous-batch-no-spec  Explicit dense continuous-batch route.
          --continuous-dynamic-pld    Opt-in dense Qwen3 adaptive route: solo PLD,
                                      then no-spec batching for compatible concurrency.
          --model-path PATH           Absolute local source-locked model directory.
          --model MODEL               Exact OpenAI request model identifier.
          --memory-limit-bytes N      Explicit positive MLX memory limit.
          --cache-limit-bytes N       Explicit positive MLX cache limit.
          --max-reserved-kv-bytes N   Required continuous-route aggregate KV cap.
          --context N                 Requested served context; a pre-load fit-check
                                      caps it to the host's ceiling and, when the
                                      model+context fits, derives the MLX memory/
                                      cache/reserved-KV limits from the sizer instead
                                      of the provided values.
          --max-prefill-tokens N      Host prefill-memory safety bound: reject prompts
                                      longer than N tokens at admission so a prompt this
                                      host cannot prefill fails the request instead of
                                      crashing the process. Independent of --context and
                                      the model window; set per host (the safe value
                                      scales with host memory). Unset disables it.
          --plan-concurrency N        Compute the fit-check verdict for N concurrent
                                      decode streams (per-stream KV scales ×N) instead
                                      of the single-stream default. Opt-in: the stricter
                                      verdict can cap the served context or refuse a set
                                      that fits at concurrency 1. Default (unset) is 1.
          --force                     Serve even when the fit-check verdict is red.
          --quant-candidates DIRS     Comma-separated absolute local checkpoint dirs
                                      (different quants of a model); the pre-load fit
                                      check auto-picks the best fit for the host and
                                      loads it. Replaces --model-path. A red-only set
                                      refuses (no --force override in this mode yet).
          --quant-pick-only           Dry-run: with --quant-candidates, print the
                                      machine-readable winner line to stdout and exit
                                      WITHOUT loading a model (exit 2 if none fits).
                                      Needs only --quant-candidates (+ optional
                                      --context); no runtime limits required.
          --fit-check-only           Dry-run: resolve the fit-check verdict for a
                                      single --model-path model directory via the
                                      exact same resolveServingLimits call the real
                                      serve makes (memory/cache limits, admitted max
                                      context, verdict), print it, and exit WITHOUT
                                      loading weights. Mutually exclusive with
                                      --quant-pick-only and --force (which would
                                      suppress the very verdict this exists to
                                      learn); not supported with --scripted (no
                                      model directory to check). Composes with
                                      --ngram-offload-plan + --qwen4exp-mtp to
                                      report the offloaded verdict -- the
                                      motivating production-cutover use case.
          --offload-plan-check-only   Dry-run: resolve the --ngram-offload-plan on THIS
                                      host and run the same pre-load verification the
                                      real load runs (row-store geometry against the
                                      chunk seal, plus the seal's chunk digests under
                                      the plan's chunkVerification policy), print the
                                      measured identity/count attestation, and exit
                                      WITHOUT loading weights or constructing a model.
                                      Answers "is this host provisioned?", which
                                      --fit-check-only does NOT (it reads only the
                                      plan's declared limits). Requires
                                      --ngram-offload-plan; mutually exclusive with
                                      --fit-check-only (a different stop point) and
                                      --force (which would suppress the very refusal
                                      this exists to surface).
          --quant-reliability PATH    With --quant-pick-only, overlay measured tool-call
                                      reliability (a quant-reliability/v1 artifact) onto
                                      the announce, joined by quant bits. Advisory: never
                                      changes the pick; a bad artifact only skips the
                                      overlay. Absolute local path.
          --kv-quant TIER             Requested KV-cache precision tier
                                      (fp16|int8|turbo4|tq2_5|tq3_5). ADVISORY ONLY:
                                      the serving runtime stores KV in bf16 (the
                                      model's mandated compute dtype); any tier only
                                      drives a sizing-only preview (against a 2-byte
                                      unquantized baseline) of the context ceiling it
                                      would buy and is NOT applied. Unknown tier fails
                                      closed.
          --tier TIER                 Serve dial (transparent|balanced|maxfit)
                                      for --quant-candidates auto-pick:
                                        transparent  fp16 KV only, never cap context
                                        balanced     escalate KV to hold full context,
                                                     refuse rather than cap
                                        maxfit       escalate KV AND cap context to fit
                                      Currently applied on the --quant-pick-only
                                      dry-run (a plan; loads nothing); enforced-serve
                                      KV-tier wiring is gated on runtime int8 KV.
                                      An unknown tier fails closed.
          --auto-quant BASE           OFFLINE-enumerate the HF quant variants of a base
                                      repository id (e.g. mlx-community/Qwen3-8B) as pick
                                      candidates, instead of explicit --quant-candidates
                                      dirs. Requires --quant-pick-only and prints the
                                      candidate repo names (the network probe/download
                                      half is not built yet). Mutually exclusive with
                                      --quant-candidates.
          --prefer MODE               Quant auto-pick ranking axis (context|quality)
                                      for --quant-candidates. context (default) keeps
                                      served context primary; quality hoists quant bits
                                      to primary, picking the highest-fidelity build
                                      that fits even when a lower-bit build would serve
                                      more context. Never selects a red candidate.
                                      An unknown mode fails closed.
          --allow-hybrid-qwen35       Admit the qwen3_5 hybrid architecture onto the
                                      continuous-batch route instead of silently falling
                                      back to scalar serving. Opt-in (default off);
                                      continuous-only, so it cannot be combined with
                                      --scripted.
          --exact-qwen35-mtp          Explicitly compose the reviewed exact Qwen3.5 MTP
                                      target/drafter pair. Default off; target remains
                                      --model-path and scalar fallback loads first.
          --exact-mtp-selection SELECTION
                                      Reviewed exact-MTP artifact lock
                                      (qwen35-9b-depth1|qwen38-27b-mxfp8-depth1|
                                      qwen38-27b-4bit-depth1).
                                      Defaults to qwen35-9b-depth1 for compatibility.
          --mtp-drafter-path PATH     Absolute local drafter snapshot directory for
                                      --exact-qwen35-mtp.
          --ngram-offload-plan PATH   Absolute local path to an offloaded n-gram serving
                                      plan. Selects the offloaded load path for a
                                      qwen4_exp checkpoint; omitted means the default
                                      load path.
          --qwen4exp-mtp              Load the in-checkpoint Qwen4-Exp (Flash Next)
                                      converted 4-bit MTP drafter from the served
                                      target's own checkpoint and gate it at startup.
                                      Requires --ngram-offload-plan (the marker-family
                                      admission gate only admits this family when an
                                      offloaded plan resolved). A bare flag: only one
                                      artifact is serve-eligible today; a second
                                      artifact would add a valued
                                      --qwen4exp-mtp-selection flag mirroring
                                      --exact-qwen35-mtp / --exact-mtp-selection.
          --qwen4exp-sampled-mtp      Opt-in: let the in-checkpoint Qwen4-Exp MTP
                                      drafter propose SAMPLED (temperature != 0)
                                      block decisions instead of only accelerating
                                      greedy requests. Requires --qwen4exp-mtp.
                                      Default off. Turning this on changes the
                                      sampling distribution machinery for every
                                      eligible request; the measured 1.31-1.40x
                                      speedup was obtained at topP=1, topK=0 and
                                      does NOT transfer to a truncated
                                      configuration (under truncation the draft and
                                      target top-k sets can differ, and where they
                                      are disjoint per-block acceptance is exactly
                                      zero and speculation is pure overhead) --
                                      default-on would ship an unmeasured change.
          --chat-template PATH        Absolute local path to a chat-template file that overrides
                                      the served checkpoint's own resolved template for BOTH
                                      rendering and the boot attestation probe (the same resolved
                                      text feeds both, so the attestation can never lie about what
                                      is actually rendered). Omitted (default) preserves today's
                                      resolution from the model directory unchanged. Missing or
                                      unreadable refuses to start, before the weight load, rather
                                      than silently falling back to the checkpoint's own template.
                                      Applies only to the loaded scalar-serve route; not supported
                                      with --scripted, continuous batching, --exact-qwen35-mtp, or
                                      --quant-pick-only.
          --default-sampling MODE     Whether a request omitting sampling params (default:
                                      off) decodes greedy argmax (off) or resolves the
                                      served checkpoint's own generation_config.json
                                      sampling subset (generation-config). Wired into
                                      the scalar serve route only; fail-closed at load
                                      (a missing/invalid generation_config.json, or a
                                      compiled-fp16 decoder strategy, refuses to
                                      start rather than falling back to greedy).
                                      Not supported with --scripted, --quant-pick-only,
                                      continuous batching, or --exact-qwen35-mtp; with
                                      --qwen4exp-mtp also requires
                                      --qwen4exp-sampled-mtp (otherwise every
                                      param-less request would move from accelerated
                                      greedy MTP to unaccelerated scalar decode).
          --host HOST                 Bind host (default: 127.0.0.1).
          --host-use VALUE            Operator host-use intent (shared|dedicated-serving).
                                      Omit to keep default policy provenance distinct
                                      from an explicit shared assertion.
          --os-service-reserve-bytes N
                                      Required nonzero OS/service reserve for an explicit
                                      dedicated-serving host classification.
          --port PORT                 Bind port (default: 8080; 0 is ephemeral).
          --default-completion-tokens N
                                      Budget used when a request omits max tokens
                                      (default: 4096; safely reduced for small contexts).
          --max-completion-tokens N   Optional operator ceiling. When omitted, the
                                      model/host-fit context determines the maximum.
          --max-non-streaming-completion-tokens N
                                      In-memory non-streaming safety cap (default: 16384).
                                      Larger admitted budgets require stream=true.
          --max-request-body-bytes N  HTTP request-body safety cap. Loaded-model default is
                                      derived from admitted context (64 bytes/token, bounded
                                      from 1 MiB through 64 MiB); explicit values override it.
          --max-non-streaming-response-bytes N
                                      Exact serialized JSON response cap (default: 16 MiB).
                                      Streaming is not subject to this aggregate-response cap.
          --completion-limit-policy MODE
                                      Explicit over-limit behavior: reject|clamp
                                      (default: reject).
          --evidence-path PATH        Fresh append-only canonical evidence output.
          --help                      Show this help.

        Set FASTMLX_API_KEY to require Bearer authentication. A non-loopback host
        is rejected unless that environment variable is non-empty.
        """

    public let backend: FastMLXServeBackend?
    public let host: String
    public let requestedHostUse: FastMLXServeHostUse?
    public let osServiceReserveBytes: Int?
    public let port: Int
    public let model: String
    public let evidencePath: URL?
    public let showHelp: Bool
    /// Compatibility value for the maximum flag. `maximumCompletionTokensWasExplicit` determines
    /// whether it narrows the model/fit-derived maximum or is merely the historical parser default.
    public let maximumCompletionTokens: Int
    public let maximumCompletionTokensWasExplicit: Bool
    public let defaultCompletionTokens: Int
    public let defaultCompletionTokensWasExplicit: Bool
    public let maximumNonStreamingCompletionTokens: Int
    public let maximumRequestBodyBytes: Int?
    public let maximumNonStreamingResponseBytes: Int
    public let completionLimitPolicy: ServingCompletionLimitPolicy
    /// Operator-requested served context (`--context N`); `nil` uses the sizer's effective default.
    /// Consumed by the pre-load fit-check, not by backend selection.
    public let requestedContext: Int?
    /// Host prefill-memory safety bound (`--max-prefill-tokens N`); `nil` disables it (the default).
    /// Rejects prompts longer than N tokens at admission so a prompt this host cannot prefill fails
    /// the request instead of crashing the process. Independent of `--context`/the model window; an
    /// interim stopgap the fit planner's transient sizer is expected to supersede. Set it per host —
    /// the safe value scales with host memory. See the incident record for the prod derivation.
    public let maxPrefillTokens: Int?
    /// `--force`: proceed past a red fit-check verdict instead of failing closed.
    public let forceServe: Bool
    /// `--quant-candidates`: several already-downloaded local checkpoint directories (different quants
    /// of a model). Empty on the single-model path. When non-empty the pre-load quant auto-pick decides
    /// which directory loads; the backend's `modelDirectory` is seeded with the first entry as a
    /// placeholder the preflight substitutes. Local-dirs-only by design (HF repo-name enumeration is a
    /// deferred policy decision — see docs/task-inbox/2026-08-18-quant-auto-pick-policy.md).
    public let quantCandidateDirectories: [URL]
    /// `--quant-pick-only`: resolve which quant candidate would load for this host and print the
    /// machine-readable winner line, then exit — NO model is loaded. Its own early-return mode
    /// (`backend == nil`), so it requires only `--quant-candidates` (+ optional `--context`), never
    /// the runtime load limits. A dry-run scripting primitive; changes no default serve behavior.
    public let quantPickOnly: Bool
    /// `--quant-reliability`: an off-box `quant-reliability/v1` artifact whose measured tool-call
    /// reliability is overlaid (display-only, joined by quant bits) onto the `--quant-pick-only`
    /// announce. Advisory: it never changes which quant is picked, and a bad artifact only skips the
    /// overlay — it does not fail the pick. Absolute local path only, mirroring `--evidence-path`.
    public let quantReliabilityPath: URL?
    /// `--kv-quant TIER`: an operator-requested KV-cache precision tier (`fp16`/`int8`/`turbo4`/…).
    /// The RAW string is carried here deliberately: `ServingCore` stays free of a `HarnessCore`
    /// dependency, so tier validation and the sizing preview both happen in `HarnessCore`
    /// (`KVQuantAdvisory`) at the serve call site. The serving runtime stores KV in **bf16** — the
    /// compute dtype the model mandates (its configuration decoder rejects any non-bfloat16 checkpoint),
    /// and the `KVCacheSimple` store allocates in the incoming K/V dtype with no fp16 cast. A tier only
    /// drives a sizing-only advisory preview (against a 2-byte unquantized baseline), never the
    /// enforced verdict; the runtime dtype is unaffected by the requested tier.
    public let kvQuantTier: String?
    /// `--tier TIER`: the operator-intent serve dial (`transparent`/`balanced`/`maxfit`). The RAW
    /// string is carried here for the same dependency-boundary reason as `kvQuantTier`: `ServingCore`
    /// stays free of a `HarnessCore` dependency, so the tier is validated (`ServeTier(rawValue:)`) and
    /// resolved into a `ServingPolicy` in `HarnessCore` at the serve call site. Consumed by the
    /// pre-load quant auto-pick (KV-tier escalation + context-capping stance), never by backend
    /// selection. `nil` (unset) preserves today's fp16 + cap-and-proceed behavior byte-for-byte.
    public let serveTier: String?
    /// `--plan-concurrency N`: the number of concurrent decode streams the operator will actually run,
    /// used to compute a STRICTER, concurrency-aware fit-check verdict (per-stream KV scales ×N). `nil`
    /// (the default) computes the verdict at concurrency 1 — byte-identical to the shipped behavior.
    /// Consumed only by the pre-load fit-check / quant auto-pick, never by backend selection. See
    /// docs/task-inbox/2026-08-18-fit-check-concurrency-kv-undercount.md (option 2).
    public let planConcurrency: Int?
    /// `--prefer MODE`: the quant auto-pick's ranking axis (`context`/`quality`). The RAW string is
    /// carried here for the same dependency-boundary reason as `serveTier`/`kvQuantTier`: `ServingCore`
    /// stays free of a `HarnessCore` dependency, so the value is validated (`QuantPickPreference`) and
    /// consumed by the pre-load quant auto-pick in `HarnessCore` at the serve call site, never by
    /// backend selection. `nil` (unset) preserves today's context-first pick behavior byte-for-byte.
    public let preferMode: String?
    /// `--auto-quant BASE`: a base repository identifier whose HF quant variants are ENUMERATED
    /// (offline, network-free) as pick candidates — the alternative to explicit local
    /// `--quant-candidates` dirs. The RAW string is carried here for the same dependency-boundary
    /// reason as `serveTier`/`preferMode`: `ServingCore` stays free of a `HarnessCore` dependency, so
    /// `HarnessCore.QuantCandidateSourcer.enumerate` expands it at the serve call site. Because the
    /// network probe/download half is not built, it is only valid under `--quant-pick-only` (enumerate
    /// and exit); the parser fails closed on any other combination. `nil` (unset) is today's behavior.
    public let autoQuantBase: String?
    /// `--allow-hybrid-qwen35`: opt-in admission of the qwen3_5 hybrid architecture (alternating
    /// GatedDeltaNet-linear / full-attention layers, `.recurrentState` cache) onto the continuous-batch
    /// serve route. Default `false` preserves today's behavior — the continuous proof rejects the hybrid
    /// family and the executable silently falls back to scalar serving. When `true`, the flag is threaded
    /// into `ContinuousServingModelLoadConfiguration` → the `DenseContinuousBatchModelProof.verifying`
    /// call (which then carries qwen3_5 through instead of throwing `unsupportedModelFamily`) and relaxes
    /// the continuous cache-layout validator to admit `.recurrentState`. Continuous-only: rejected when
    /// combined with `--scripted` (which loads no model). See
    /// docs/task-inbox/2026-08-20-hybrid-continuous-serve-path-admission.md.
    public let allowHybridQwen35: Bool
    /// `--exact-qwen35-mtp`: explicit opt-in to compose the reviewed exact Qwen3.5 MTP pair on top
    /// of a separately loaded scalar fallback. Default `false`; the target stays the ordinary
    /// `--model-path` scalar load, while `mtpDrafterDirectory` carries the separate local drafter
    /// snapshot when enabled.
    public let exactQwen35MTP: Bool
    /// `--exact-mtp-selection`: reviewed exact-MTP artifact lock to load when exact MTP is enabled.
    /// Defaults to the original 9B lock so existing invocations stay compatible.
    public let exactMTPSelection: FastMLXExactMTPSelection
    public let mtpDrafterDirectory: URL?
    /// `--ngram-offload-plan`: an absolute local path to an offloaded n-gram serving plan. Selects
    /// the offloaded load path for a `qwen4_exp` checkpoint at the scalar load call site; `nil`
    /// (the default) preserves today's load path. Fail-closed at the parser: not supported combined
    /// with continuous batching or `--exact-qwen35-mtp`, because neither route reaches the scalar-load
    /// seam that consumes it — accepting the flag there would silently ignore an operator's request.
    public let ngramOffloadPlanURL: URL?
    /// `--qwen4exp-mtp`: opt-in to load the in-checkpoint Qwen4-Exp (Flash Next, `qwen4_exp`) MTP
    /// drafter from the served target's own checkpoint and gate it at startup
    /// (`loadScalarServingModel`, which already implements the gate). `nil` (the default) preserves
    /// today's load path. A BARE flag rather than a valued one: only the converted 4-bit artifact
    /// (`FastMLXInCheckpointMTPSelection.converted4Bit`) is serve-eligible today, so there is
    /// exactly one case to select. A future second artifact would add a valued
    /// `--qwen4exp-mtp-selection` flag, mirroring how `--exact-qwen35-mtp` (bare, opt-in) pairs
    /// with `--exact-mtp-selection` (valued, picks among several reviewed locks). Fail-closed at
    /// the parser: requires `--ngram-offload-plan` (see `qwen4ExpMTPRequiresNGramOffloadPlan`) and
    /// cannot be combined with `--scripted` (see `qwen4ExpMTPWithScripted`); every other conflicting
    /// mode (continuous batching, `--exact-qwen35-mtp`, `--quant-pick-only`, `--quant-candidates`)
    /// is covered TRANSITIVELY through the `--ngram-offload-plan` requirement, which already
    /// refuses all four combinations itself.
    public let inCheckpointMTPSelection: FastMLXInCheckpointMTPSelection?
    /// `--qwen4exp-sampled-mtp`: opt-in to let the in-checkpoint Qwen4-Exp MTP drafter propose
    /// SAMPLED (temperature != 0) block decisions through a `SampledMTPBlockRuntimeDeciding`
    /// provider, instead of accelerating only greedy requests (today's behavior). Default `false`
    /// preserves today's behavior byte-for-byte: `MTPSpeculativeDecoder.prefill` passes `nil` for
    /// `sampledBlockDecisionProvider`, so `MTPSpeculativeTokenIterator`'s `providerIsEligible` stays
    /// unconditionally false and no sampled request is ever routed through the provider machinery.
    /// Requires `inCheckpointMTPSelection` to be non-nil (see
    /// `FastMLXServeArgumentError.sampledMTPRequiresInCheckpointMTP`) -- there is no drafter
    /// iterator to hand the provider to otherwise.
    ///
    /// THIS IS AN UNMEASURED-DISTRIBUTION-CHANGE FLAG, not a free accelerator: turning it on swaps
    /// the drafter's sampler and every acceptance/residual/bonus uniform for the provider's own
    /// (`MTPSpeculativeTokenIterator.swift:211-212`) for every eligible request. The measured
    /// 1.31-1.40x sampled-MTP speedup was obtained at `topP=1, topK=0` (untruncated) and does NOT
    /// transfer to a truncated configuration: under truncation the draft and target top-k sets can
    /// differ, and where they are disjoint per-block acceptance is exactly zero and speculation is
    /// pure overhead. Shipping this default-on would ship that unmeasured regime as the default,
    /// which is why it stays opt-in.
    public let sampledMTPBlockDecisionsEnabled: Bool
    /// The operator's raw `--memory-limit-bytes` payload, threaded independently of `backend` so it
    /// reaches every consumer that plans against the host envelope — including `--quant-pick-only`,
    /// whose early-return carries no `backend` case at all (`backend == nil`) and would otherwise
    /// silently drop an operator budget the pre-load quant pick should have planned against. `nil`
    /// when the flag was omitted; omission IS the signal that no operator budget should bind (see
    /// docs/task-inbox/2026-09-06-operator-budget-envelope-shape-DECISION.md).
    public let memoryLimitBytes: Int?
    /// `--chat-template`: an absolute local path to a chat-template file that overrides the served
    /// checkpoint's own resolved template. `nil` (the default) preserves today's resolution from
    /// the model directory (`chat_template.jinja`, falling back to `chat_template.json`, then
    /// `tokenizer_config.json`) unchanged, byte-for-byte. Consumed only at the scalar-load call
    /// site (`ScalarServingModelLoadConfiguration.chatTemplateOverrideURL`,
    /// `SpikeServingAdapters`), mirroring `ngramOffloadPlanURL`'s dependency-boundary idiom
    /// exactly: `ServingCore` carries the validated `URL?`, never the file's contents, and the
    /// load-time seam re-validates and reads it. Fail-closed at the parser against every mode
    /// that would silently drop the flag rather than merely leave it unused — `--scripted`
    /// (`chatTemplateWithScripted`), either continuous route (`chatTemplateWithContinuousBatch`),
    /// `--exact-qwen35-mtp` (`chatTemplateWithExactQwen35MTP`), and `--quant-pick-only`
    /// (`chatTemplateWithQuantPickOnly`) — none of those routes reach the scalar-load seam that
    /// consumes this field. `--quant-candidates` (the loaded, non-pick-only auto-pick route) is
    /// deliberately NOT refused: the resolved winning directory still loads through the same
    /// scalar-load seam, so the override applies normally there.
    public let chatTemplateURL: URL?
    /// `--fit-check-only`: report the `resolveServingLimits` verdict for the resolved model
    /// directory and exit before the load that follows it, instead of skipping that computation.
    /// Default `false` preserves today's behavior byte-for-byte. Deliberately NOT threaded through
    /// `--quant-pick-only`'s early-return construction (see `fitCheckOnlyWithQuantPickOnly`'s doc
    /// comment): this flag stays on the FULL loaded-model construction path all the way through
    /// `--model-path`/`--model` resolution, so the seam at the real `resolveServingLimits` call
    /// sites in `FastMLXServe.swift` is the ONLY place that computes the reported verdict -- a
    /// second, parallel fit computation here could drift from what the real serve uses and make
    /// the flag misleading.
    public let fitCheckOnly: Bool
    /// `--offload-plan-check-only`: resolve and verify the offloaded n-gram plan's row store on
    /// this host at the scalar-load seam
    /// (`ScalarServingModelLoadConfiguration.offloadPlanCheckOnly`), then stop BEFORE any weight
    /// load or model construction, instead of skipping that computation. Default `false` preserves
    /// today's behavior byte-for-byte. Requires `ngramOffloadPlanURL` to be non-nil (see
    /// `offloadPlanCheckOnlyRequiresNGramOffloadPlan`) and is refused alongside `--fit-check-only`
    /// or `--force` (see those cases' own doc comments) -- otherwise threaded exactly like
    /// `fitCheckOnly`, on the FULL loaded-model construction path, never through
    /// `--quant-pick-only`'s early return.
    public let offloadPlanCheckOnly: Bool
    /// `--default-sampling`: `off` (the default) preserves today's behavior byte-for-byte -- a
    /// request omitting sampling parameters decodes greedy argmax. `generationConfig` opts into
    /// resolving the served checkpoint's own `generation_config.json` sampling subset for such a
    /// request instead, for the SCALAR serve route only
    /// (`ScalarServingModelLoadConfiguration.defaultSampling`,
    /// `spike/Sources/SpikeServingAdapters/MLXScalarServing.swift`). Fail-closed at LOAD, not
    /// per-request: `GenerationConfigSamplingDefaults.load(contentsOf:)` runs against
    /// `generation_config.json` inside the model directory before any weight load, and a
    /// missing/unparseable file, `do_sample: false`, absent/zero `temperature`, or an out-of-range
    /// sampling field refuses to start rather than silently falling back to greedy. Also refused at
    /// load when the resolved scalar decoder strategy is `.compiledFP16`
    /// (`scalarServingDefaultSamplingDecoderStrategyError`) -- that decoder does not opt into
    /// `Decoder.supportsSampling`, so admitting the combination would boot healthy on the greedy
    /// startup probe and then fail every real sampled request mid-stream with an opaque backend
    /// error instead. Refused here at PARSE TIME (this type) against every OTHER route that would
    /// otherwise silently ignore this flag or regress accepted behavior when it converts
    /// param-less traffic into sampled traffic: `--scripted` and `--quant-pick-only` never serve a
    /// request at all (`defaultSamplingWithScripted` / `defaultSamplingWithQuantPickOnly`);
    /// continuous batching's `ContinuousServingBackend` rejects `.sampled` with HTTP 400
    /// (`defaultSamplingWithContinuousBatch`); `--exact-qwen35-mtp` falls back to scalar decoding
    /// on any non-greedy sampling (`defaultSamplingWithExactQwen35MTP`); and `--qwen4exp-mtp`
    /// without `--qwen4exp-sampled-mtp` would move every param-less request from accelerated
    /// greedy MTP to unaccelerated scalar decode (`defaultSamplingWithInCheckpointMTPRequiresSampledMTP`).
    public let defaultSampling: FastMLXServeDefaultSampling

    private init(
        backend: FastMLXServeBackend?,
        host: String,
        requestedHostUse: FastMLXServeHostUse? = nil,
        osServiceReserveBytes: Int? = nil,
        port: Int,
        model: String,
        evidencePath: URL?,
        showHelp: Bool,
        maximumCompletionTokens: Int = OpenAIChatRequestLimits.productionDefault
            .maximumCompletionTokens,
        maximumCompletionTokensWasExplicit: Bool = false,
        defaultCompletionTokens: Int = 4_096,
        defaultCompletionTokensWasExplicit: Bool = false,
        maximumNonStreamingCompletionTokens: Int = 16_384,
        maximumRequestBodyBytes: Int? = nil,
        maximumNonStreamingResponseBytes: Int = 16 * 1_048_576,
        completionLimitPolicy: ServingCompletionLimitPolicy = .reject,
        requestedContext: Int? = nil,
        maxPrefillTokens: Int? = nil,
        forceServe: Bool = false,
        quantCandidateDirectories: [URL] = [],
        quantPickOnly: Bool = false,
        quantReliabilityPath: URL? = nil,
        kvQuantTier: String? = nil,
        serveTier: String? = nil,
        planConcurrency: Int? = nil,
        preferMode: String? = nil,
        autoQuantBase: String? = nil,
        allowHybridQwen35: Bool = false,
        exactQwen35MTP: Bool = false,
        exactMTPSelection: FastMLXExactMTPSelection = .qwen35_9BDepth1,
        mtpDrafterDirectory: URL? = nil,
        ngramOffloadPlanURL: URL? = nil,
        inCheckpointMTPSelection: FastMLXInCheckpointMTPSelection? = nil,
        sampledMTPBlockDecisionsEnabled: Bool = false,
        memoryLimitBytes: Int? = nil,
        chatTemplateURL: URL? = nil,
        fitCheckOnly: Bool = false,
        offloadPlanCheckOnly: Bool = false,
        defaultSampling: FastMLXServeDefaultSampling = .off
    ) {
        self.backend = backend
        self.host = host
        self.requestedHostUse = requestedHostUse
        self.osServiceReserveBytes = osServiceReserveBytes
        self.port = port
        self.model = model
        self.evidencePath = evidencePath
        self.showHelp = showHelp
        self.maximumCompletionTokens = maximumCompletionTokens
        self.maximumCompletionTokensWasExplicit = maximumCompletionTokensWasExplicit
        self.defaultCompletionTokens = defaultCompletionTokens
        self.defaultCompletionTokensWasExplicit = defaultCompletionTokensWasExplicit
        self.maximumNonStreamingCompletionTokens = maximumNonStreamingCompletionTokens
        self.maximumRequestBodyBytes = maximumRequestBodyBytes
        self.maximumNonStreamingResponseBytes = maximumNonStreamingResponseBytes
        self.completionLimitPolicy = completionLimitPolicy
        self.requestedContext = requestedContext
        self.maxPrefillTokens = maxPrefillTokens
        self.forceServe = forceServe
        self.quantCandidateDirectories = quantCandidateDirectories
        self.quantPickOnly = quantPickOnly
        self.quantReliabilityPath = quantReliabilityPath
        self.kvQuantTier = kvQuantTier
        self.serveTier = serveTier
        self.planConcurrency = planConcurrency
        self.preferMode = preferMode
        self.autoQuantBase = autoQuantBase
        self.allowHybridQwen35 = allowHybridQwen35
        self.exactQwen35MTP = exactQwen35MTP
        self.exactMTPSelection = exactMTPSelection
        self.mtpDrafterDirectory = mtpDrafterDirectory
        self.ngramOffloadPlanURL = ngramOffloadPlanURL
        self.inCheckpointMTPSelection = inCheckpointMTPSelection
        self.sampledMTPBlockDecisionsEnabled = sampledMTPBlockDecisionsEnabled
        self.memoryLimitBytes = memoryLimitBytes
        self.chatTemplateURL = chatTemplateURL
        self.fitCheckOnly = fitCheckOnly
        self.offloadPlanCheckOnly = offloadPlanCheckOnly
        self.defaultSampling = defaultSampling
    }

    public static func parse<S: Sequence>(
        _ rawArguments: S
    ) throws -> FastMLXServeArguments where S.Element == String {
        let arguments = Array(rawArguments)
        var seen: Set<String> = []
        var scripted = false
        var continuousBatchNoSpec = false
        var continuousDynamicPLD = false
        var showHelp = false
        var maximumCompletionTokens = OpenAIChatRequestLimits.productionDefault
            .maximumCompletionTokens
        var maximumCompletionTokensWasExplicit = false
        var defaultCompletionTokens = 4_096
        var defaultCompletionTokensWasExplicit = false
        var maximumNonStreamingCompletionTokens = 16_384
        var maximumRequestBodyBytes: Int?
        var maximumNonStreamingResponseBytes = 16 * 1_048_576
        var completionLimitPolicy = ServingCompletionLimitPolicy.reject
        var host = "127.0.0.1"
        var requestedHostUse: FastMLXServeHostUse?
        var osServiceReserveBytes: Int?
        var port = 8_080
        var model: String?
        var modelPath: String?
        var memoryLimitBytes: Int?
        var cacheLimitBytes: Int?
        var maxReservedKVBytes: Int?
        var requestedContext: Int?
        var maxPrefillTokens: Int?
        var forceServe = false
        var quantCandidateDirs: [URL] = []
        var quantPickOnly = false
        var quantReliabilityPath: URL?
        var kvQuantTier: String?
        var serveTier: String?
        var planConcurrency: Int?
        var preferMode: String?
        var autoQuantBase: String?
        var allowHybridQwen35 = false
        var exactQwen35MTP = false
        var exactMTPSelection = FastMLXExactMTPSelection.qwen35_9BDepth1
        var exactMTPSelectionWasExplicit = false
        var mtpDrafterDirectory: URL?
        var ngramOffloadPlanURL: URL?
        var qwen4ExpMTP = false
        var qwen4ExpSampledMTP = false
        var evidencePath: URL?
        var chatTemplateURL: URL?
        var fitCheckOnly = false
        var offloadPlanCheckOnly = false
        var defaultSampling = FastMLXServeDefaultSampling.off

        var index = 0
        while index < arguments.count {
            let argument = arguments[index]
            guard Self.supportedOptions.contains(argument) else {
                throw FastMLXServeArgumentError.unknownArgument(argument)
            }
            guard seen.insert(argument).inserted else {
                throw FastMLXServeArgumentError.duplicateOption(argument)
            }

            switch argument {
            case "--scripted":
                scripted = true
            case "--continuous-batch-no-spec":
                continuousBatchNoSpec = true
            case "--continuous-dynamic-pld":
                continuousDynamicPLD = true
            case "--help", "-h":
                showHelp = true
            case "--host":
                index += 1
                host = try value(at: index, in: arguments, for: argument)
            case "--host-use":
                index += 1
                let rawHostUse = try value(at: index, in: arguments, for: argument)
                guard let hostUse = FastMLXServeHostUse(rawValue: rawHostUse) else {
                    throw FastMLXServeArgumentError.invalidHostUse
                }
                requestedHostUse = hostUse
            case "--os-service-reserve-bytes":
                index += 1
                osServiceReserveBytes = try positiveInteger(
                    try value(at: index, in: arguments, for: argument),
                    option: argument)
            case "--port":
                index += 1
                let rawPort = try value(
                    at: index, in: arguments, for: argument)
                guard let parsedPort = Int(rawPort),
                    (0...65_535).contains(parsedPort)
                else {
                    throw FastMLXServeArgumentError.invalidPort
                }
                port = parsedPort
            case "--max-completion-tokens":
                index += 1
                maximumCompletionTokens = try strictPositiveInteger(
                    try value(at: index, in: arguments, for: argument),
                    option: argument)
                maximumCompletionTokensWasExplicit = true
            case "--default-completion-tokens":
                index += 1
                defaultCompletionTokens = try strictPositiveInteger(
                    try value(at: index, in: arguments, for: argument),
                    option: argument)
                defaultCompletionTokensWasExplicit = true
            case "--max-non-streaming-completion-tokens":
                index += 1
                maximumNonStreamingCompletionTokens = try strictPositiveInteger(
                    try value(at: index, in: arguments, for: argument),
                    option: argument)
            case "--max-request-body-bytes":
                index += 1
                maximumRequestBodyBytes = try strictPositiveInteger(
                    try value(at: index, in: arguments, for: argument),
                    option: argument)
            case "--max-non-streaming-response-bytes":
                index += 1
                maximumNonStreamingResponseBytes = try strictPositiveInteger(
                    try value(at: index, in: arguments, for: argument),
                    option: argument)
            case "--completion-limit-policy":
                index += 1
                let rawPolicy = try value(at: index, in: arguments, for: argument)
                guard let policy = ServingCompletionLimitPolicy(rawValue: rawPolicy) else {
                    throw FastMLXServeArgumentError.invalidCompletionLimitPolicy
                }
                completionLimitPolicy = policy
            case "--model":
                index += 1
                model = try value(at: index, in: arguments, for: argument)
            case "--model-path":
                index += 1
                modelPath = try value(
                    at: index, in: arguments, for: argument)
            case "--evidence-path":
                index += 1
                let path = try value(
                    at: index, in: arguments, for: argument)
                guard path.hasPrefix("/") else {
                    throw FastMLXServeArgumentError
                        .evidencePathMustBeAbsolute
                }
                evidencePath = URL(fileURLWithPath: path)
            case "--memory-limit-bytes":
                index += 1
                memoryLimitBytes = try positiveInteger(
                    try value(at: index, in: arguments, for: argument),
                    option: argument)
            case "--cache-limit-bytes":
                index += 1
                cacheLimitBytes = try positiveInteger(
                    try value(at: index, in: arguments, for: argument),
                    option: argument)
            case "--max-reserved-kv-bytes":
                index += 1
                maxReservedKVBytes = try positiveInteger(
                    try value(at: index, in: arguments, for: argument),
                    option: argument)
            case "--context":
                index += 1
                requestedContext = try positiveInteger(
                    try value(at: index, in: arguments, for: argument),
                    option: argument)
            case "--max-prefill-tokens":
                index += 1
                maxPrefillTokens = try positiveInteger(
                    try value(at: index, in: arguments, for: argument),
                    option: argument)
            case "--plan-concurrency":
                index += 1
                planConcurrency = try positiveInteger(
                    try value(at: index, in: arguments, for: argument),
                    option: argument)
            case "--force":
                forceServe = true
            case "--quant-candidates":
                index += 1
                let raw = try value(at: index, in: arguments, for: argument)
                // Split on comma; every entry must be a non-empty absolute local path. A trailing/
                // leading/doubled comma yields an empty entry → fail closed (never silently drop it).
                let entries = raw.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
                var dirs: [URL] = []
                for entry in entries {
                    guard entry.hasPrefix("/") else {
                        throw FastMLXServeArgumentError.quantCandidateMustBeAbsolute
                    }
                    dirs.append(URL(fileURLWithPath: entry, isDirectory: true))
                }
                quantCandidateDirs = dirs
            case "--quant-pick-only":
                quantPickOnly = true
            case "--quant-reliability":
                index += 1
                let path = try value(at: index, in: arguments, for: argument)
                guard path.hasPrefix("/") else {
                    throw FastMLXServeArgumentError.quantReliabilityPathMustBeAbsolute
                }
                quantReliabilityPath = URL(fileURLWithPath: path)
            case "--kv-quant":
                index += 1
                // Carry the raw tier string; HarnessCore's KVQuantAdvisory validates it (fail-closed
                // on an unknown tier) so ServingCore stays free of a HarnessCore dependency.
                kvQuantTier = try value(at: index, in: arguments, for: argument)
            case "--tier":
                index += 1
                // Carry the raw serve-dial string; HarnessCore's ServeTier validates it (fail-closed
                // on an unknown tier) at the serve call site, same dependency-boundary idiom as --kv-quant.
                serveTier = try value(at: index, in: arguments, for: argument)
            case "--prefer":
                index += 1
                // Carry the raw ranking-axis string; HarnessCore's QuantPickPreference validates it
                // (fail-closed on an unknown mode) at the serve call site, same idiom as --tier.
                preferMode = try value(at: index, in: arguments, for: argument)
            case "--auto-quant":
                index += 1
                // Carry the raw base repo id; HarnessCore's QuantCandidateSourcer enumerates its quant
                // variants at the serve call site (offline), same dependency-boundary idiom as --prefer.
                autoQuantBase = try value(at: index, in: arguments, for: argument)
            case "--allow-hybrid-qwen35":
                allowHybridQwen35 = true
            case "--exact-qwen35-mtp":
                exactQwen35MTP = true
            case "--qwen4exp-mtp":
                qwen4ExpMTP = true
            case "--qwen4exp-sampled-mtp":
                qwen4ExpSampledMTP = true
            case "--fit-check-only":
                fitCheckOnly = true
            case "--offload-plan-check-only":
                offloadPlanCheckOnly = true
            case "--exact-mtp-selection":
                index += 1
                let rawSelection = try value(at: index, in: arguments, for: argument)
                guard let parsedSelection = FastMLXExactMTPSelection(rawValue: rawSelection) else {
                    throw FastMLXServeArgumentError.invalidExactMTPSelection
                }
                exactMTPSelection = parsedSelection
                exactMTPSelectionWasExplicit = true
            case "--mtp-drafter-path":
                index += 1
                let path = try value(at: index, in: arguments, for: argument)
                guard path.hasPrefix("/") else {
                    throw FastMLXServeArgumentError.mtpDrafterPathMustBeAbsolute
                }
                mtpDrafterDirectory = URL(fileURLWithPath: path, isDirectory: true)
            case "--ngram-offload-plan":
                index += 1
                let path = try value(at: index, in: arguments, for: argument)
                guard path.hasPrefix("/") else {
                    throw FastMLXServeArgumentError.ngramOffloadPlanMustBeAbsolute
                }
                ngramOffloadPlanURL = URL(fileURLWithPath: path)
            case "--chat-template":
                index += 1
                let path = try value(at: index, in: arguments, for: argument)
                guard path.hasPrefix("/") else {
                    throw FastMLXServeArgumentError.chatTemplateMustBeAbsolute
                }
                chatTemplateURL = URL(fileURLWithPath: path)
            case "--default-sampling":
                index += 1
                let rawDefaultSampling = try value(at: index, in: arguments, for: argument)
                guard
                    let parsedDefaultSampling = FastMLXServeDefaultSampling(
                        rawValue: rawDefaultSampling)
                else {
                    throw FastMLXServeArgumentError.invalidDefaultSampling
                }
                defaultSampling = parsedDefaultSampling
            default:
                preconditionFailure("supported option was not handled")
            }
            index += 1
        }

        if defaultCompletionTokensWasExplicit,
            maximumCompletionTokensWasExplicit,
            defaultCompletionTokens > maximumCompletionTokens
        {
            throw FastMLXServeArgumentError
                .defaultCompletionTokensExceedsMaximumCompletionTokens
        }

        if showHelp {
            return FastMLXServeArguments(
                backend: nil,
                host: host,
                requestedHostUse: requestedHostUse,
                osServiceReserveBytes: osServiceReserveBytes,
                port: port,
                model: model ?? "fastmlx-scripted",
                evidencePath: evidencePath,
                showHelp: true,
                maximumCompletionTokens: maximumCompletionTokens,
                maximumCompletionTokensWasExplicit: maximumCompletionTokensWasExplicit,
                defaultCompletionTokens: defaultCompletionTokens,
                defaultCompletionTokensWasExplicit: defaultCompletionTokensWasExplicit,
                maximumNonStreamingCompletionTokens: maximumNonStreamingCompletionTokens,
                maximumRequestBodyBytes: maximumRequestBodyBytes,
                maximumNonStreamingResponseBytes: maximumNonStreamingResponseBytes,
                completionLimitPolicy: completionLimitPolicy)
        }

        if requestedHostUse == .dedicatedServing {
            guard osServiceReserveBytes != nil else {
                throw FastMLXServeArgumentError.missingRequiredOption(
                    "--os-service-reserve-bytes")
            }
        } else if osServiceReserveBytes != nil {
            throw FastMLXServeArgumentError.osServiceReserveRequiresDedicatedServing
        }

        if mtpDrafterDirectory != nil, !exactQwen35MTP {
            throw FastMLXServeArgumentError.mtpDrafterRequiresExactQwen35MTP
        }
        if exactMTPSelectionWasExplicit, !exactQwen35MTP {
            throw FastMLXServeArgumentError.exactMTPSelectionRequiresExactQwen35MTP
        }

        // --fit-check-only refusals that do not depend on ordering against anything else below:
        // (1) --force suppresses the very verdict --fit-check-only exists to learn -- a red host
        // would otherwise be able to report a --fit-check-only "success" merely because --force
        // stopped the refusal that verdict would normally trigger; (2) --quant-pick-only is a
        // DIFFERENT dry run (auto-pick across --quant-candidates directories rather than a
        // single-model fit-check), so combining the two is ambiguous about which one the operator
        // wants. The --scripted refusal lives further below, grouped alongside its sibling
        // loaded-model-only flags (--qwen4exp-mtp, --ngram-offload-plan, --chat-template) that are
        // each refused there by an explicit, unconditional check rather than through
        // `hasLoadedModelOptions` -- see that check's own doc comment for why it is not folded in
        // here as a third simple pair-check.
        if fitCheckOnly, forceServe {
            throw FastMLXServeArgumentError.fitCheckOnlyWithForce
        }
        if fitCheckOnly, quantPickOnly {
            throw FastMLXServeArgumentError.fitCheckOnlyWithQuantPickOnly
        }

        // --offload-plan-check-only's own refusals, grouped here mirroring --fit-check-only's
        // block immediately above: (1) --fit-check-only is a DIFFERENT dry run with a different
        // stop point -- composing them would silently pick one stop point over the other; (2)
        // --force suppresses the very refusal --offload-plan-check-only exists to prove absent,
        // exactly like --fit-check-only's identical refusal above. The "requires
        // --ngram-offload-plan" refusal lives further below, grouped alongside its sibling
        // --qwen4exp-mtp requirement -- see offloadPlanCheckOnlyRequiresNGramOffloadPlan's own
        // doc comment for why that placement gives it transitive coverage instead of needing a
        // third check here.
        if offloadPlanCheckOnly, fitCheckOnly {
            throw FastMLXServeArgumentError.offloadPlanCheckOnlyWithFitCheckOnly
        }
        if offloadPlanCheckOnly, forceServe {
            throw FastMLXServeArgumentError.offloadPlanCheckOnlyWithForce
        }

        if continuousBatchNoSpec, continuousDynamicPLD {
            throw FastMLXServeArgumentError.conflictingBackendModes
        }
        let continuousModeSelected = continuousBatchNoSpec || continuousDynamicPLD

        // --ngram-offload-plan is consumed only at the scalar-load call site (`loadScalarServingBackend`
        // → `ScalarServingModelLoadConfiguration`). Neither the continuous-batch routes nor the exact
        // Qwen3.5 MTP composition reach that seam, so silently accepting the flag there would drop an
        // operator's explicit request on the floor — fail closed instead.
        if ngramOffloadPlanURL != nil, continuousModeSelected {
            throw FastMLXServeArgumentError.ngramOffloadPlanWithContinuousBatch
        }
        if ngramOffloadPlanURL != nil, exactQwen35MTP {
            throw FastMLXServeArgumentError.ngramOffloadPlanWithExactQwen35MTP
        }
        // --quant-pick-only returns early below WITHOUT threading this field, so the flag would be
        // silently dropped rather than merely unused. `--mtp-drafter-path` avoids the same trap only
        // transitively (it requires --exact-qwen35-mtp, which already conflicts with a quant source);
        // this flag has no such transitive block, so it needs an explicit one.
        if ngramOffloadPlanURL != nil, quantPickOnly {
            throw FastMLXServeArgumentError.ngramOffloadPlanWithQuantPickOnly
        }
        // The plan is sealed against ONE specific on-disk artifact, so pairing it with an auto-pick
        // across several candidate directories is semantically incoherent — the winning candidate
        // need not be the artifact the plan was sealed against. It is also unreachable in practice:
        // `resolveServedDirectory` runs the quant auto-pick and can throw `FitCheckRefusal` on the
        // unadjusted full-resident figure BEFORE `resolveServingLimits` and its offload-aware fit
        // adjustment ever run, so the very refusal this flag exists to eliminate would still fire.
        // Fail closed and point the operator at the single sealed directory via --model-path instead.
        if ngramOffloadPlanURL != nil, !quantCandidateDirs.isEmpty {
            throw FastMLXServeArgumentError.ngramOffloadPlanWithQuantCandidates
        }

        // --chat-template overrides the resolved chat template only at the scalar-load call site
        // (`loadScalarServingBackend` → `ScalarServingModelLoadConfiguration.chatTemplateOverrideURL`).
        // Neither continuous-batch route nor the exact Qwen3.5 MTP composition constructs its
        // tokenizer through that seam, so silently accepting the flag there would drop an
        // operator's explicit override request on the floor — fail closed, mirroring
        // --ngram-offload-plan's identical continuous/exact-MTP rationale immediately above.
        // Unlike --ngram-offload-plan, no --quant-candidates refusal is needed: a template
        // override is orthogonal to which candidate directory wins the auto-pick, and the
        // resolved winner still loads through the same scalar-load seam this flag targets.
        if chatTemplateURL != nil, continuousModeSelected {
            throw FastMLXServeArgumentError.chatTemplateWithContinuousBatch
        }
        if chatTemplateURL != nil, exactQwen35MTP {
            throw FastMLXServeArgumentError.chatTemplateWithExactQwen35MTP
        }
        // --quant-pick-only returns early below WITHOUT threading this field, the same silent-drop
        // trap --ngram-offload-plan's own guard above documents.
        if chatTemplateURL != nil, quantPickOnly {
            throw FastMLXServeArgumentError.chatTemplateWithQuantPickOnly
        }

        // --qwen4exp-mtp requires --ngram-offload-plan (the marker-family admission gate only
        // admits this family when an offloaded plan resolved — see the error case doc comment).
        // Placed AFTER the --ngram-offload-plan refusal block above so that, when both flags plus
        // a conflicting mode are given, the more specific ngram error fires first. This also gives
        // --qwen4exp-mtp TRANSITIVE coverage against continuous batching, --exact-qwen35-mtp,
        // --quant-pick-only, and --quant-candidates: each of those four is already refused above
        // whenever --ngram-offload-plan is present, so a separate --qwen4exp-mtp-specific refusal
        // for any of them would be unreachable dead code. This mirrors the existing
        // --mtp-drafter-path comment describing the same transitive-coverage trap.
        if qwen4ExpMTP, ngramOffloadPlanURL == nil {
            throw FastMLXServeArgumentError.qwen4ExpMTPRequiresNGramOffloadPlan
        }

        // --qwen4exp-sampled-mtp plugs a sampled block-decision provider into the in-checkpoint MTP
        // drafter's own iterator, which only --qwen4exp-mtp loads -- see
        // sampledMTPRequiresInCheckpointMTP's doc comment for the transitive coverage this
        // requirement buys against every other conflicting mode (the same shape as the
        // --qwen4exp-mtp / --ngram-offload-plan requirement immediately above).
        if qwen4ExpSampledMTP, !qwen4ExpMTP {
            throw FastMLXServeArgumentError.sampledMTPRequiresInCheckpointMTP
        }

        // --offload-plan-check-only verifies the offloaded n-gram plan the SAME way the real load
        // resolves it, so it is meaningless without a plan to check. Mirrors
        // qwen4ExpMTPRequiresNGramOffloadPlan's identical placement/shape immediately above -- see
        // offloadPlanCheckOnlyRequiresNGramOffloadPlan's own doc comment for the transitive
        // coverage this requirement buys against every other conflicting mode.
        if offloadPlanCheckOnly, ngramOffloadPlanURL == nil {
            throw FastMLXServeArgumentError.offloadPlanCheckOnlyRequiresNGramOffloadPlan
        }

        // --auto-quant is an OFFLINE enumerate-only quant source (its network probe/download half is
        // not built): mutually exclusive with the local --quant-candidates source, and usable only
        // under --quant-pick-only (enumerate the candidate repo names and exit). Fail closed on any
        // other combination rather than pretending it can load a model.
        if let base = autoQuantBase {
            guard quantCandidateDirs.isEmpty else {
                throw FastMLXServeArgumentError.autoQuantWithCandidates
            }
            guard quantPickOnly else {
                throw FastMLXServeArgumentError.autoQuantRequiresPickOnly
            }
            guard !base.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw FastMLXServeArgumentError.invalidAutoQuantBase
            }
        }

        if exactQwen35MTP {
            if scripted {
                throw FastMLXServeArgumentError.exactQwen35MTPWithScripted
            }
            if continuousModeSelected {
                throw FastMLXServeArgumentError.exactQwen35MTPWithContinuousBatch
            }
            if !quantCandidateDirs.isEmpty || quantPickOnly || autoQuantBase != nil {
                throw FastMLXServeArgumentError.exactQwen35MTPWithQuantSource
            }
            guard mtpDrafterDirectory != nil else {
                throw FastMLXServeArgumentError.missingRequiredOption("--mtp-drafter-path")
            }
        }

        // --default-sampling generation-config converts param-less traffic (temperature omitted)
        // from greedy argmax into artifact-sourced SAMPLED decoding. Refuse every route/mode
        // combination where that conversion would be silently inert or actively regress accepted
        // behavior, mirroring this file's other "flag would be silently dropped/ignored"
        // refusals. Placed here -- after `continuousModeSelected` and the full `exactQwen35MTP`
        // validation block above, but BEFORE the `--quant-pick-only` early return below -- so
        // every boolean this block reads is already resolved and no early return can skip it.
        // `--default-sampling off` (the default) never enters this block.
        if defaultSampling == .generationConfig {
            // --scripted never constructs or serves a request at all.
            if scripted {
                throw FastMLXServeArgumentError.defaultSamplingWithScripted
            }
            // --quant-pick-only's dry-run return path resolves a pick and exits without ever
            // serving a request.
            if quantPickOnly {
                throw FastMLXServeArgumentError.defaultSamplingWithQuantPickOnly
            }
            // ContinuousServingBackend rejects `.sampled` with HTTP 400 (see
            // defaultSamplingWithContinuousBatch's own doc comment).
            if continuousModeSelected {
                throw FastMLXServeArgumentError.defaultSamplingWithContinuousBatch
            }
            // The exact Qwen3.5 MTP route falls back to scalar decoding on any non-greedy
            // sampling (see defaultSamplingWithExactQwen35MTP's own doc comment).
            if exactQwen35MTP {
                throw FastMLXServeArgumentError.defaultSamplingWithExactQwen35MTP
            }
            // Without --qwen4exp-sampled-mtp, every param-less request under --qwen4exp-mtp would
            // move from accelerated GREEDY MTP to UNACCELERATED scalar decode -- a throughput
            // cliff, not a silent drop (see defaultSamplingWithInCheckpointMTPRequiresSampledMTP's own
            // doc comment).
            if qwen4ExpMTP, !qwen4ExpSampledMTP {
                throw FastMLXServeArgumentError.defaultSamplingWithInCheckpointMTPRequiresSampledMTP
            }
        }

        // --quant-pick-only is its own early-return mode below, which returns `backend: nil` and
        // never reaches the general `--scripted` conflict check further down (that check runs only
        // on the loaded-model path, past this early return). Without this guard, --scripted paired
        // with --quant-pick-only would silently discard the requested --scripted transport-only
        // backend and fall through into a quant-pick run instead. Both are legitimate backend-mode
        // selections, so treat the pairing as the same general conflictingBackendModes case used
        // everywhere else in this file, checked BEFORE the early return can fire.
        if scripted, quantPickOnly {
            throw FastMLXServeArgumentError.conflictingBackendModes
        }
        // --quant-pick-only is its own early-return mode: it resolves the pick and exits without a
        // load, so it needs ONLY the candidate list (+ optional --context). It deliberately does NOT
        // reach the load-mode required-option guards below — nothing is loaded, so demanding
        // --model/--memory-limit-bytes/--cache-limit-bytes would be noise. The guards stay intact for
        // every actual serve path (see the non-regression test).
        if quantPickOnly {
            // --auto-quant enumerate mode: the source is a base repo id, not local dirs.
            if let base = autoQuantBase {
                return FastMLXServeArguments(
                    backend: nil,
                    host: host,
                    requestedHostUse: requestedHostUse,
                    osServiceReserveBytes: osServiceReserveBytes,
                    port: port,
                    model: model ?? "fastmlx-quant-pick",
                    evidencePath: evidencePath,
                    showHelp: false,
                    maximumCompletionTokens: maximumCompletionTokens,
                    maximumCompletionTokensWasExplicit: maximumCompletionTokensWasExplicit,
                    defaultCompletionTokens: defaultCompletionTokens,
                    defaultCompletionTokensWasExplicit: defaultCompletionTokensWasExplicit,
                    maximumNonStreamingCompletionTokens: maximumNonStreamingCompletionTokens,
                    maximumRequestBodyBytes: maximumRequestBodyBytes,
                    maximumNonStreamingResponseBytes: maximumNonStreamingResponseBytes,
                    completionLimitPolicy: completionLimitPolicy,
                    requestedContext: requestedContext,
                    forceServe: forceServe,
                    quantCandidateDirectories: [],
                    quantPickOnly: true,
                    quantReliabilityPath: quantReliabilityPath,
                    serveTier: serveTier,
                    planConcurrency: planConcurrency,
                    preferMode: preferMode,
                    autoQuantBase: base,
                    memoryLimitBytes: memoryLimitBytes)
            }
            guard !quantCandidateDirs.isEmpty else {
                throw FastMLXServeArgumentError.missingRequiredOption("--quant-candidates")
            }
            return FastMLXServeArguments(
                backend: nil,
                host: host,
                requestedHostUse: requestedHostUse,
                osServiceReserveBytes: osServiceReserveBytes,
                port: port,
                model: model ?? "fastmlx-quant-pick",
                evidencePath: evidencePath,
                showHelp: false,
                maximumCompletionTokens: maximumCompletionTokens,
                maximumCompletionTokensWasExplicit: maximumCompletionTokensWasExplicit,
                defaultCompletionTokens: defaultCompletionTokens,
                defaultCompletionTokensWasExplicit: defaultCompletionTokensWasExplicit,
                maximumNonStreamingCompletionTokens: maximumNonStreamingCompletionTokens,
                maximumRequestBodyBytes: maximumRequestBodyBytes,
                maximumNonStreamingResponseBytes: maximumNonStreamingResponseBytes,
                completionLimitPolicy: completionLimitPolicy,
                requestedContext: requestedContext,
                forceServe: forceServe,
                quantCandidateDirectories: quantCandidateDirs,
                quantPickOnly: true,
                quantReliabilityPath: quantReliabilityPath,
                serveTier: serveTier,
                planConcurrency: planConcurrency,
                preferMode: preferMode,
                memoryLimitBytes: memoryLimitBytes)
        }

        let hasQuantCandidates = !quantCandidateDirs.isEmpty
        let hasLoadedModelOptions =
            modelPath != nil || memoryLimitBytes != nil || cacheLimitBytes != nil
                || maxReservedKVBytes != nil || hasQuantCandidates
        if scripted, continuousModeSelected || hasLoadedModelOptions {
            throw FastMLXServeArgumentError.conflictingBackendModes
        }
        // --kv-quant is a loaded-model concern (it previews a KV sizing); it is meaningless in the
        // transport-only scripted backend, so reject the combination rather than silently ignoring it.
        if scripted, kvQuantTier != nil {
            throw FastMLXServeArgumentError.kvQuantWithScripted
        }
        // --allow-hybrid-qwen35 admits a model onto the continuous serve route; the transport-only
        // scripted backend loads no model, so the combination is a misconfig — fail closed.
        if scripted, allowHybridQwen35 {
            throw FastMLXServeArgumentError.allowHybridWithScripted
        }
        // --qwen4exp-mtp loads a model from the served target's own checkpoint; the transport-only
        // scripted backend loads no model, so the flag would be silently dropped. Not covered by
        // the earlier `hasLoadedModelOptions` check (that set intentionally excludes this flag, the
        // same way it excludes --ngram-offload-plan itself) — an explicit refusal is required.
        if scripted, qwen4ExpMTP {
            throw FastMLXServeArgumentError.qwen4ExpMTPWithScripted
        }
        // --ngram-offload-plan is consumed only at the scalar-load seam (see the doc comment on
        // .ngramOffloadPlanWithScripted); the transport-only scripted backend loads no model, so the
        // flag would be silently dropped. Placed AFTER the --qwen4exp-mtp-with-scripted check above
        // so that --scripted --qwen4exp-mtp --ngram-offload-plan keeps reporting the more specific
        // qwen4ExpMTPWithScripted error (see testInCheckpointMTPWithScripted) rather than this one.
        if scripted, ngramOffloadPlanURL != nil {
            throw FastMLXServeArgumentError.ngramOffloadPlanWithScripted
        }
        // --chat-template is consumed only at the scalar-load seam (see the doc comment on
        // .chatTemplateWithContinuousBatch above); the transport-only scripted backend loads no
        // model, so the flag would be silently dropped.
        if scripted, chatTemplateURL != nil {
            throw FastMLXServeArgumentError.chatTemplateWithScripted
        }
        // --fit-check-only reports the `resolveServingLimits` verdict for a loaded model; the
        // transport-only scripted backend loads no model and has no model directory to check, so the
        // flag would have nothing to report. Not covered by `hasLoadedModelOptions` (this flag alone
        // implies no model-path/memory/cache/reserved-KV/quant-candidates value), so an explicit
        // refusal is required, mirroring `qwen4ExpMTPWithScripted`/`ngramOffloadPlanWithScripted`
        // immediately above.
        if scripted, fitCheckOnly {
            throw FastMLXServeArgumentError.fitCheckOnlyWithScripted
        }
        if continuousDynamicPLD, allowHybridQwen35 {
            throw FastMLXServeArgumentError.dynamicPLDWithHybridQwen35
        }
        if scripted {
            let launchedModel = try validatedModel(
                model ?? "fastmlx-scripted")
            return FastMLXServeArguments(
                backend: .scripted,
                host: host,
                requestedHostUse: requestedHostUse,
                osServiceReserveBytes: osServiceReserveBytes,
                port: port,
                model: launchedModel,
                evidencePath: evidencePath,
                showHelp: false,
                maximumCompletionTokens: maximumCompletionTokens,
                maximumCompletionTokensWasExplicit: maximumCompletionTokensWasExplicit,
                defaultCompletionTokens: defaultCompletionTokens,
                defaultCompletionTokensWasExplicit: defaultCompletionTokensWasExplicit,
                maximumNonStreamingCompletionTokens: maximumNonStreamingCompletionTokens,
                maximumRequestBodyBytes: maximumRequestBodyBytes,
                maximumNonStreamingResponseBytes: maximumNonStreamingResponseBytes,
                completionLimitPolicy: completionLimitPolicy)
        }

        guard continuousModeSelected || hasLoadedModelOptions else {
            throw FastMLXServeArgumentError.missingBackendMode
        }
        if !continuousModeSelected, maxReservedKVBytes != nil {
            throw FastMLXServeArgumentError.optionRequiresContinuousBatchMode(
                "--max-reserved-kv-bytes")
        }
        // Resolve the load directory: an explicit --model-path, or (candidates mode) the first
        // candidate as a placeholder the pre-load quant auto-pick substitutes with the actual winner.
        // The two are mutually exclusive — --quant-candidates *is* the source in that mode.
        if hasQuantCandidates, modelPath != nil {
            throw FastMLXServeArgumentError.quantCandidatesWithModelPath
        }
        let modelDirectory: URL
        if let modelPath {
            guard modelPath.hasPrefix("/") else {
                throw FastMLXServeArgumentError.modelPathMustBeAbsolute
            }
            modelDirectory = URL(fileURLWithPath: modelPath, isDirectory: true)
        } else if hasQuantCandidates {
            modelDirectory = quantCandidateDirs[0]
        } else {
            throw FastMLXServeArgumentError.missingRequiredOption("--model-path")
        }
        guard let model else {
            throw FastMLXServeArgumentError.missingRequiredOption("--model")
        }
        let launchedModel = try validatedModel(model)
        // --memory-limit-bytes/--cache-limit-bytes are OPTIONAL (operator-budget-envelope-shape
        // decision): the sizer derives concrete bytes downstream regardless, so an omitted flag is
        // not a missing requirement — it is the explicit signal that no operator budget should bind.
        // The ordering invariant below still holds whenever BOTH are actually present; it is simply
        // not evaluated when either is absent (never compared against a fabricated placeholder).
        if let cacheLimitBytes, let memoryLimitBytes, cacheLimitBytes > memoryLimitBytes {
            throw FastMLXServeArgumentError.cacheLimitExceedsMemoryLimit
        }
        let resolvedMaxReservedKVBytes: Int?
        if continuousModeSelected {
            guard let maxReservedKVBytes else {
                throw FastMLXServeArgumentError.missingRequiredOption(
                    "--max-reserved-kv-bytes")
            }
            // Only enforced when --memory-limit-bytes was actually supplied; skipped (not silently
            // compared against a bogus stand-in) when it is absent.
            if let memoryLimitBytes, maxReservedKVBytes > memoryLimitBytes {
                throw FastMLXServeArgumentError
                    .reservedKVLimitExceedsMemoryLimit
            }
            resolvedMaxReservedKVBytes = maxReservedKVBytes
        } else {
            resolvedMaxReservedKVBytes = nil
        }

        return FastMLXServeArguments(
            backend: continuousBatchNoSpec
                ? .continuousBatchNoSpec(
                    modelDirectory: modelDirectory,
                    memoryLimitBytes: memoryLimitBytes,
                    cacheLimitBytes: cacheLimitBytes,
                    maxReservedKVBytes: resolvedMaxReservedKVBytes!)
                : continuousDynamicPLD
                    ? .continuousDynamicPLD(
                        modelDirectory: modelDirectory,
                        memoryLimitBytes: memoryLimitBytes,
                        cacheLimitBytes: cacheLimitBytes,
                        maxReservedKVBytes: resolvedMaxReservedKVBytes!)
                    : .scalar(
                        modelDirectory: modelDirectory,
                        memoryLimitBytes: memoryLimitBytes,
                        cacheLimitBytes: cacheLimitBytes),
            host: host,
            requestedHostUse: requestedHostUse,
            osServiceReserveBytes: osServiceReserveBytes,
            port: port,
            model: launchedModel,
            evidencePath: evidencePath,
            showHelp: false,
            maximumCompletionTokens: maximumCompletionTokens,
            maximumCompletionTokensWasExplicit: maximumCompletionTokensWasExplicit,
            defaultCompletionTokens: defaultCompletionTokens,
            defaultCompletionTokensWasExplicit: defaultCompletionTokensWasExplicit,
            maximumNonStreamingCompletionTokens: maximumNonStreamingCompletionTokens,
            maximumRequestBodyBytes: maximumRequestBodyBytes,
            maximumNonStreamingResponseBytes: maximumNonStreamingResponseBytes,
            completionLimitPolicy: completionLimitPolicy,
            requestedContext: requestedContext,
            maxPrefillTokens: maxPrefillTokens,
            forceServe: forceServe,
            quantCandidateDirectories: quantCandidateDirs,
            kvQuantTier: kvQuantTier,
            serveTier: serveTier,
            planConcurrency: planConcurrency,
            preferMode: preferMode,
            allowHybridQwen35: allowHybridQwen35,
            exactQwen35MTP: exactQwen35MTP,
            exactMTPSelection: exactMTPSelection,
            mtpDrafterDirectory: mtpDrafterDirectory,
            ngramOffloadPlanURL: ngramOffloadPlanURL,
            inCheckpointMTPSelection: qwen4ExpMTP ? .converted4Bit : nil,
            sampledMTPBlockDecisionsEnabled: qwen4ExpSampledMTP,
            memoryLimitBytes: memoryLimitBytes,
            chatTemplateURL: chatTemplateURL,
            fitCheckOnly: fitCheckOnly,
            offloadPlanCheckOnly: offloadPlanCheckOnly,
            defaultSampling: defaultSampling)
    }

    private static let supportedOptions: Set<String> = [
        "--scripted",
        "--continuous-batch-no-spec",
        "--continuous-dynamic-pld",
        "--help",
        "-h",
        "--host",
        "--host-use",
        "--os-service-reserve-bytes",
        "--port",
        "--default-completion-tokens",
        "--max-completion-tokens",
        "--max-non-streaming-completion-tokens",
        "--max-request-body-bytes",
        "--max-non-streaming-response-bytes",
        "--completion-limit-policy",
        "--model",
        "--model-path",
        "--evidence-path",
        "--memory-limit-bytes",
        "--cache-limit-bytes",
        "--max-reserved-kv-bytes",
        "--context",
        "--max-prefill-tokens",
        "--plan-concurrency",
        "--force",
        "--quant-candidates",
        "--quant-pick-only",
        "--quant-reliability",
        "--kv-quant",
        "--tier",
        "--prefer",
        "--auto-quant",
        "--allow-hybrid-qwen35",
        "--exact-qwen35-mtp",
        "--exact-mtp-selection",
        "--mtp-drafter-path",
        "--ngram-offload-plan",
        "--qwen4exp-mtp",
        "--qwen4exp-sampled-mtp",
        "--chat-template",
        "--fit-check-only",
        "--offload-plan-check-only",
        "--default-sampling",
    ]

    private static func value(
        at index: Int,
        in arguments: [String],
        for option: String
    ) throws -> String {
        guard arguments.indices.contains(index),
            !arguments[index].isEmpty,
            !arguments[index].hasPrefix("--")
        else {
            throw FastMLXServeArgumentError.missingValue(option)
        }
        return arguments[index]
    }

    private static func positiveInteger(
        _ rawValue: String,
        option: String
    ) throws -> Int {
        guard let value = Int(rawValue), value > 0 else {
            throw FastMLXServeArgumentError.invalidPositiveInteger(option)
        }
        return value
    }

    private static func strictPositiveInteger(
        _ rawValue: String,
        option: String
    ) throws -> Int {
        guard !rawValue.isEmpty,
            rawValue.utf8.allSatisfy({ (48...57).contains($0) }),
            let value = Int(rawValue),
            value > 0
        else {
            throw FastMLXServeArgumentError.invalidPositiveInteger(option)
        }
        return value
    }

    private static func validatedModel(_ rawValue: String) throws -> String {
        guard !rawValue.trimmingCharacters(
            in: .whitespacesAndNewlines
        ).isEmpty else {
            throw FastMLXServeArgumentError.invalidModelIdentifier
        }
        return rawValue
    }
}
