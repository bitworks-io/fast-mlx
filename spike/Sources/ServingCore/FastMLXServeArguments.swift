import Foundation

public enum FastMLXServeBackend: Equatable, Sendable {
    case scripted
    case scalar(
        modelDirectory: URL,
        memoryLimitBytes: Int,
        cacheLimitBytes: Int)
    case continuousBatchNoSpec(
        modelDirectory: URL,
        memoryLimitBytes: Int,
        cacheLimitBytes: Int,
        maxReservedKVBytes: Int)
    case continuousDynamicPLD(
        modelDirectory: URL,
        memoryLimitBytes: Int,
        cacheLimitBytes: Int,
        maxReservedKVBytes: Int)
}

public enum FastMLXServeHostUse: String, Equatable, Sendable {
    case shared
    case dedicatedServing = "dedicated-serving"
}

public enum FastMLXExactMTPSelection: String, Equatable, Sendable {
    case qwen35_9BDepth1 = "qwen35-9b-depth1"
    case qwen38_27BMXFP8Depth1 = "qwen38-27b-mxfp8-depth1"
    case qwen38_27B4BitDepth1 = "qwen38-27b-4bit-depth1"
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
        }
    }
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
          --quant-reliability PATH    With --quant-pick-only, overlay measured tool-call
                                      reliability (a quant-reliability/v1 artifact) onto
                                      the announce, joined by quant bits. Advisory: never
                                      changes the pick; a bad artifact only skips the
                                      overlay. Absolute local path.
          --kv-quant TIER             Requested KV-cache precision tier
                                      (fp16|int8|turbo4|tq2_5|tq3_5). ADVISORY ONLY:
                                      the serving runtime stores KV in fp16; a
                                      non-fp16 tier drives a sizing-only preview of
                                      the context ceiling it would buy and is NOT
                                      applied. An unknown tier fails closed.
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
    /// (`KVQuantAdvisory`) at the serve call site. The serving runtime still stores KV in fp16 today
    /// — a non-fp16 tier only drives a sizing-only advisory preview, never the enforced verdict.
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
        mtpDrafterDirectory: URL? = nil
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
        var evidencePath: URL?

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

        if continuousBatchNoSpec, continuousDynamicPLD {
            throw FastMLXServeArgumentError.conflictingBackendModes
        }
        let continuousModeSelected = continuousBatchNoSpec || continuousDynamicPLD

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
                    autoQuantBase: base)
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
                preferMode: preferMode)
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
        guard let memoryLimitBytes else {
            throw FastMLXServeArgumentError.missingRequiredOption(
                "--memory-limit-bytes")
        }
        guard let cacheLimitBytes else {
            throw FastMLXServeArgumentError.missingRequiredOption(
                "--cache-limit-bytes")
        }
        guard cacheLimitBytes <= memoryLimitBytes else {
            throw FastMLXServeArgumentError.cacheLimitExceedsMemoryLimit
        }
        let resolvedMaxReservedKVBytes: Int?
        if continuousModeSelected {
            guard let maxReservedKVBytes else {
                throw FastMLXServeArgumentError.missingRequiredOption(
                    "--max-reserved-kv-bytes")
            }
            guard maxReservedKVBytes <= memoryLimitBytes else {
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
            forceServe: forceServe,
            quantCandidateDirectories: quantCandidateDirs,
            kvQuantTier: kvQuantTier,
            serveTier: serveTier,
            planConcurrency: planConcurrency,
            preferMode: preferMode,
            allowHybridQwen35: allowHybridQwen35,
            exactQwen35MTP: exactQwen35MTP,
            exactMTPSelection: exactMTPSelection,
            mtpDrafterDirectory: mtpDrafterDirectory)
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
