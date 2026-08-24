import Darwin
import Dispatch
import Foundation
import HarnessCore
import ServingCore
import ServingNIO
import ServingSnapshotBridge
import SpikeCore
import SpikeServingAdapters

/// Concurrent decode slots the continuous-batch route admits. Shared between the runtime
/// `ContinuousBatchConfiguration` and the fit-check's concurrency advisory so the modeled
/// concurrent-KV line always reflects the slot count the backend actually runs.
private let continuousMaxActiveSlots = 4

@main
struct FastMLXServe {
    static func main() async throws {
        do {
            try await run()
        } catch is FitCheckRefusal {
            // The refusal summary was already written to stderr by emitFitCheck; fail closed with a
            // clean non-zero exit instead of a Swift top-level fatalError trap (exit 133, doubled
            // message). Covers both the single-model red verdict and the quant-candidates all-red set.
            exit(2)
        }
    }

    private static func run() async throws {
        let arguments = try FastMLXServeArguments.parse(
            CommandLine.arguments.dropFirst())
        if arguments.showHelp {
            print(FastMLXServeArguments.usage)
            return
        }

        // Fail closed on an unknown --kv-quant tier here, on the path every serving mode reaches, so a
        // typo exits before any model load rather than silently serving fp16. The validated tier is
        // re-read (cheaply, always valid after this) at the fit-check to compose the sizing advisory.
        if let rawTier = arguments.kvQuantTier {
            _ = try KVQuantAdvisory.validateTier(rawTier)
        }

        // Fail closed on an unknown --tier value here too, before any load, on the shared path.
        if let rawServeTier = arguments.serveTier {
            _ = try ServeTier.validated(rawServeTier)
        }

        // Fail closed on an unknown --prefer ranking axis here too, before any load, on the shared path.
        if let rawPrefer = arguments.preferMode {
            _ = try QuantPickPreference.validated(rawPrefer)
        }

        if arguments.quantPickOnly {
            try runQuantPickOnly(arguments)
            return
        }

        let apiKey = ProcessInfo.processInfo.environment["FASTMLX_API_KEY"].flatMap {
            $0.isEmpty ? nil : $0
        }
        let prepared = try await prepareBackend(arguments)
        let evidenceSink: ServingEvidenceJSONLSink?
        do {
            evidenceSink = try arguments.evidencePath.map {
                try ServingEvidenceJSONLSink(path: $0.path)
            }
        } catch {
            await prepared.backend.shutdown()
            throw error
        }
        let evidenceConfiguration = evidenceSink.map { sink in
            ServingHTTPEvidenceConfiguration(
                snapshot: prepared.evidenceSnapshot,
                record: { evidence in
                    try await sink.record(evidence)
                },
                reportFailure: { message in
                    FileHandle.standardError.write(
                        Data("fastmlx-serve evidence_failure=\(message)\n".utf8))
                })
        }
        let requestLimits = OpenAIChatRequestLimits(
            maximumBodyBytes: OpenAIChatRequestLimits.productionDefault.maximumBodyBytes,
            maximumCompletionTokens: arguments.maximumCompletionTokens)
        let configuration = ServingHTTPConfiguration(
            launchedModel: arguments.model,
            requestLimits: requestLimits,
            requiredBearerToken: apiKey,
            maximumNonStreamingResponseBytes: 1_048_576,
            backpressureStallTimeout: .seconds(5),
            evidence: evidenceConfiguration)
        let server: ServingHTTPServer
        do {
            server = try await ServingHTTPServer.start(
                bind: .init(host: arguments.host, port: arguments.port),
                configuration: configuration,
                backend: prepared.backend)
        } catch {
            try? await evidenceSink?.finish()
            await prepared.backend.shutdown()
            throw error
        }

        print(prepared.startupLine(
            localAddress: server.localAddress.description,
            maximumCompletionTokens: requestLimits.maximumCompletionTokens))
        print("fastmlx-serve ready=true; press Control-C to stop.")

        await waitForShutdownSignal()
        do {
            try await server.shutdown(gracePeriod: .seconds(5))
            try await evidenceSink?.finish()
        } catch {
            try? await evidenceSink?.finish()
            throw error
        }
        print("fastmlx-serve shutdown=complete")
    }
}

private enum PreparedServingStartupReport {
    case scalar(ScalarServingModelStartupReport)
    case continuous(ContinuousServingModelStartupReport)
}

private struct PreparedServingBackend {
    let backend: any ServingGenerationBackend
    let evidenceSnapshot:
        ServingHTTPEvidenceConfiguration.SnapshotProvider?
    let mode: String
    let launchedModel: String
    let startupReport: PreparedServingStartupReport?
    let fitDecision: ServingFitDecision?
    let exactMTPReport: ExactQwen35MTPServeStartupReport?

    func startupLine(
        localAddress: String,
        maximumCompletionTokens: Int
    ) -> String {
        guard let startupReport else {
            return """
                fastmlx-serve mode=\(mode) transport_only=true \
                model=\(launchedModel) \
                max_completion_tokens_limit=\(maximumCompletionTokens) \
                listening=\(localAddress)
                """
        }
        let fit = fitDecision?.machineReadableFields() ?? "fit_check=skipped"
        switch startupReport {
        case .scalar(let report):
            let nativeCacheKinds = Set(
                report.nativeCacheKinds.map(\.rawValue)
            ).sorted().joined(separator: ",")
            let route: String
            let scalarFallbackRoute: String
            if case .exactSuccess = exactMTPReport?.status {
                route = ServingExecutionRoute.exactQwen35MTP.rawValue
                scalarFallbackRoute = "scalar_fallback_route=\(report.route.rawValue)"
            } else {
                route = report.route.rawValue
                scalarFallbackRoute = ""
            }
            let exactMTP = exactMTPReport?.machineReadableFields() ?? ""
            return """
                fastmlx-serve mode=\(mode) route=\(route) \
                model=\(report.launchedModel) \
                memory_limit_bytes=\(report.memoryLimitBytes) \
                cache_limit_bytes=\(report.cacheLimitBytes) \
                native_cache_layers=\(report.nativeCacheKinds.count) \
                native_cache_kinds=\(nativeCacheKinds) \
                stop_token_count=\(report.stopTokenCount) \
                stop_string_count=\(report.stopStringCount) \
                startup_prompt_token_count=\(report.startupPromptTokenCount) \
                startup_generated_token_count=\(report.startupGeneratedTokenCount) \
                reset_parity_verified=\(report.resetParityVerified) \
                max_completion_tokens_limit=\(maximumCompletionTokens) \
                \(report.memoryFieldsFragment) \
                \(scalarFallbackRoute) \
                \(exactMTP) \
                \(fit) \
                listening=\(localAddress)
                """
        case .continuous(let report):
            let nativeCacheKinds = Set(
                report.nativeCacheKinds.map(\.rawValue)
            ).sorted().joined(separator: ",")
            let soloPLD = report.soloPLDPolicy
            return """
                fastmlx-serve mode=\(mode) route=\(report.route.rawValue) \
                model=\(report.launchedModel) \
                memory_limit_bytes=\(report.memoryLimitBytes) \
                cache_limit_bytes=\(report.cacheLimitBytes) \
                max_reserved_kv_bytes=\(report.maxReservedKVBytes) \
                max_context_tokens=\(report.maxContextTokens) \
                max_reserved_context_tokens=\(report.maxReservedContextTokens) \
                model_family=\(report.modelFamily.rawValue) \
                model_config_sha256=\(report.modelConfigurationSHA256) \
                model_layers=\(report.layerCount) \
                kv_heads=\(report.keyValueHeadCount) \
                head_dimension=\(report.headDimension) \
                native_cache_layers=\(report.nativeCacheKinds.count) \
                native_cache_kinds=\(nativeCacheKinds) \
                stop_token_count=\(report.stopTokenCount) \
                stop_string_count=\(report.stopStringCount) \
                startup_prompt_token_count=\(report.startupPromptTokenCount) \
                startup_generated_token_count=\(report.startupGeneratedTokenCount) \
                max_active_slots=\(report.maxActiveSlots) \
                max_prefill_slots=\(report.maxPrefillSlots) \
                prefill_chunk_size=\(report.prefillChunkSize) \
                max_queued_requests=\(report.maxQueuedRequests) \
                publication_capacity=\(report.publicationCapacity) \
                solo_pld_policy_configured=\(soloPLD != nil) \
                solo_pld_ngram=\(soloPLD?.ngram ?? 0) \
                solo_pld_max_draft=\(soloPLD?.maxDraft ?? 0) \
                solo_pld_lookback=\(soloPLD?.lookback ?? 0) \
                solo_pld_compiled_verify=\(soloPLD?.compiledVerify ?? false) \
                model_proof_verified=\(report.modelProofVerified) \
                max_completion_tokens_limit=\(maximumCompletionTokens) \
                \(fit) \
                listening=\(localAddress)
                """
        }
    }
}

/// MLX limits resolved for a backend after the pre-load fit-check: either the sizer's own
/// derivation (when the on-disk config was fit-checkable and the model fits) or the operator-
/// provided values unchanged (when the arch isn't fit-checkable yet — no serving regression).
private struct ResolvedServingLimits {
    let memoryLimitBytes: Int
    let cacheLimitBytes: Int
    /// `nil` for the scalar route (which has no reserved-KV knob); the continuous route falls back
    /// to its provided value when this is `nil`.
    let maxReservedKVBytes: Int?
    /// Served context cap the sizer computed; `nil` when the fit-check was skipped.
    let maxContextTokens: Int?
    /// The computed fit-check decision; `nil` when the fit-check was skipped (decoder failure).
    let fitDecision: ServingFitDecision?
}

private struct FitCheckRefusal: Error, CustomStringConvertible {
    let lines: [String]
    var description: String { lines.joined(separator: "\n") }
}

private func emitFitCheck(_ lines: [String]) {
    FileHandle.standardError.write(Data((lines.joined(separator: "\n") + "\n").utf8))
}

/// `--quant-pick-only`: resolve which candidate quant would load for this host and print the
/// machine-readable winner line to STDOUT, then return without loading a model. Per-candidate
/// diagnostics go to STDERR (via `emitFitCheck`) so STDOUT carries only the pipeable winner line
/// (`WINNER=$(fastmlx-serve --quant-pick-only …)`). A red-only set has no winner → throw
/// `FitCheckRefusal`, which `main()` maps to exit 2 — the same code the serve path refuses with.
private func runQuantPickOnly(_ arguments: FastMLXServeArguments) throws {
    // --auto-quant: OFFLINE-enumerate the base repo's HF quant variants and print the candidate list.
    // This is the network-free half (probe/download is not built): it emits the ordered repo names a
    // downstream step would probe, on a frozen-key STDOUT line mirroring the winner line's style.
    if let base = arguments.autoQuantBase {
        let candidates = QuantCandidateSourcer.enumerate(baseRepoID: base)
        emitFitCheck([
            "auto-quant enumerate base=\(base) candidates=\(candidates.count) "
                + "(offline name generation; network probe/download not yet built)"
        ])
        print("quant_enumerate base=\(base) candidates=\(candidates.joined(separator: ","))")
        return
    }

    // Apply the serve dial to the auto-pick: --tier resolves the KV-tier escalation set + the
    // context-capping stance the picker obeys. Safe to apply fully here because --quant-pick-only is a
    // PLAN — it loads no model, so a non-fp16 winner is a recommendation, not an enforced runtime
    // state the fp16-only backend would contradict. Unset --tier → nil policy → today's behavior.
    let policy = try servingPolicyForTier(arguments)
    if let note = policy?.conflictAnnotation { emitFitCheck(["serve-tier note: \(note)"]) }
    let resolution = QuantCandidateResolver.resolve(
        candidateDirectories: arguments.quantCandidateDirectories,
        host: SystemProfile.detectHost(),
        requestedContext: arguments.requestedContext,
        concurrency: arguments.planConcurrency ?? 1,
        allowedKVTiers: policy?.allowedKVTiers,
        allowContextCapping: policy?.allowContextCapping ?? true,
        preference: try quantPickPreference(arguments))
    emitFitCheck(resolution.summaryLines())
    emitQuantReliabilityOverlay(arguments: arguments, pick: resolution.pick)
    guard let winnerLine = resolution.machineReadableWinnerLine() else {
        throw FitCheckRefusal(lines: resolution.summaryLines())
    }
    print(winnerLine)
}

/// Resolve the `--tier` serve dial into a `ServingPolicy`, composing any explicit `--kv-quant`
/// override (pins the KV-tier set + records a conflict note when it departs from the tier's default)
/// and `--context`. Returns `nil` when no `--tier` was given, so every call site falls back to the
/// shipped fp16 + cap-and-proceed behavior byte-for-byte. Validation already ran fail-closed at the
/// serve entry; the re-parse here is cheap and always succeeds.
private func servingPolicyForTier(_ arguments: FastMLXServeArguments) throws -> ServingPolicy? {
    guard let rawTier = arguments.serveTier else { return nil }
    let tier = try ServeTier.validated(rawTier)
    let explicitKV = try arguments.kvQuantTier.map { try KVQuantAdvisory.validateTier($0) }
    return ServeTierPolicy.resolve(
        tier: tier, explicitKVQuant: explicitKV, explicitContext: arguments.requestedContext)
}

/// Resolve `--prefer` into the quant auto-pick's ranking axis, defaulting to context-first when unset.
/// Validation already ran fail-closed at the serve entry; the re-parse here is cheap and always
/// succeeds, mirroring `servingPolicyForTier`.
private func quantPickPreference(_ arguments: FastMLXServeArguments) throws -> QuantPickPreference {
    guard let raw = arguments.preferMode else { return .context }
    return try QuantPickPreference.validated(raw)
}

/// Overlay measured tool-call reliability (a `quant-reliability/v1` artifact) onto the pick announce,
/// to STDERR (via `emitFitCheck`) so the STDOUT winner-line contract stays byte-identical. This is
/// ADVISORY: it never influences which quant is picked, and a missing/malformed/foreign artifact
/// only emits a skip note — it never fails the pick (whose winner line is the pipeable contract).
private func emitQuantReliabilityOverlay(arguments: FastMLXServeArguments, pick: QuantPickResult) {
    guard let path = arguments.quantReliabilityPath else { return }
    do {
        let artifact = try QuantReliabilityArtifactRenderer.decodeValidated(
            from: Data(contentsOf: path))
        emitFitCheck(QuantPickReliabilityAnnounce.compose(
            pick: pick, artifactModel: artifact.model, rows: artifact.rows()))
    } catch {
        emitFitCheck(["quant reliability overlay skipped: \(error)"])
    }
}

/// Fit-check a real on-disk checkpoint against the detected host before loading (fit-checked-serve,
/// differentiator #2): derive the MLX memory/cache/reserved-KV limits and served-context cap from
/// the sizer instead of the flat RAM-percentage values serve.sh passes, and fail closed on a red
/// verdict unless `--force`. Falls back to the provided limits (with a `fit_check=skipped` note)
/// when the arch is not fit-checkable yet, so serving an unmodeled family is never regressed.
private func resolveServingLimits(
    modelDirectory: URL, arguments: FastMLXServeArguments,
    providedMemory: Int, providedCache: Int, providedReservedKV: Int?,
    preParsed: ParsedModelArch? = nil, advisorySlotCount: Int? = nil
) throws -> ResolvedServingLimits {
    let parsed: ParsedModelArch
    if let preParsed {
        // The quant auto-pick already decoded the winning directory — reuse it instead of a second
        // disk read + JSON parse.
        parsed = preParsed
    } else {
        do {
            parsed = try ModelConfigDecoder.decodeModelDirectory(modelDirectory, id: arguments.model)
        } catch let error as ModelConfigDecodeError where error.indicatesUnservableCheckpoint {
            // The checkpoint has no loadable weights on disk (interrupted / metadata-only download):
            // skipping the fit-check here would drive on to a guaranteed load failure. Fail closed with
            // a clear refusal instead, keeping the differentiator's fail-closed promise honest for an
            // incomplete checkpoint rather than crashing deep in the loader. (An *unmodeled arch* whose
            // weights ARE present still takes the skip path below — see indicatesUnservableCheckpoint.)
            let lines = [
                "fastmlx-serve fit_check=refused reason=\(error.unservableRefusalReason ?? "unservable_checkpoint")",
                "  \(error)",
            ]
            emitFitCheck(lines)
            throw FitCheckRefusal(lines: lines)
        } catch {
            emitFitCheck(["fastmlx-serve fit_check=skipped reason=\"\(error)\" — using provided limits"])
            return ResolvedServingLimits(
                memoryLimitBytes: providedMemory, cacheLimitBytes: providedCache,
                maxReservedKVBytes: providedReservedKV, maxContextTokens: nil, fitDecision: nil)
        }
    }

    let host = SystemProfile.detectHost()

    // Resolve the `--tier` serve dial (differentiator #4) into its policy so the dial governs the REAL
    // serve decision, not only the `--quant-pick-only` dry-run: transparent/balanced refuse a
    // memory-bound context cap (allowContextCapping=false) while maxfit caps-and-proceeds. `nil` when
    // no `--tier` was given → allowContextCapping defaults `true` → byte-identical prior serve behavior
    // (regression lock). The tier string is already validated at entry in `run(...)`, so this
    // re-resolve cannot throw here; the KV-tier escalation the policy also carries stays advisory below
    // (the fp16-only backend cannot honor a lossy KV tier at runtime — see the runtime_not_wired note).
    let servePolicy = try servingPolicyForTier(arguments)
    if let note = servePolicy?.conflictAnnotation { emitFitCheck(["serve-tier note: \(note)"]) }
    // The tier's KV escalation stays advisory on the LOADED path (fp16-only runtime); surface what it
    // wanted so the operator is not surprised the enforced verdict is fp16. See enforcedPathKVAdvisory.
    if let advisory = ServeTierPolicy.enforcedPathKVAdvisory(for: servePolicy) { emitFitCheck([advisory]) }

    let decision = ServingFitPlanner.decide(
        profile: parsed.profile, weightsAreMeasured: parsed.weightsAreMeasured, host: host,
        requestedContext: arguments.requestedContext, concurrency: arguments.planConcurrency ?? 1,
        force: arguments.forceServe,
        quantBits: parsed.quantBits, weightsAreDeclared: parsed.weightsAreDeclared,
        advisorySlotCount: advisorySlotCount,
        allowContextCapping: servePolicy?.allowContextCapping ?? true)
    emitFitCheck(decision.summaryLines())

    // Sizing-only advisory for a requested non-fp16 KV-cache tier. Emitted BEFORE the refusal guard
    // so an operator whose fp16 verdict is red still sees the mitigation ("int8 would fit at ceiling
    // X"). The what-if decision is HarnessCore's; every line is labeled runtime_not_wired. The
    // enforced decision above (fp16) is untouched — this can never manufacture a phantom GREEN.
    if let rawTier = arguments.kvQuantTier, let tier = try? KVQuantAdvisory.validateTier(rawTier) {
        let advisory = KVQuantAdvisory.previewLines(
            tier: tier, profile: parsed.profile, host: host,
            requestedContext: arguments.requestedContext, quantBits: parsed.quantBits)
        if !advisory.isEmpty { emitFitCheck(advisory) }
    }

    // Surface the PREDICTED cold-snapshot reuse granularity (roadmap #3a) at serve time, off-box from
    // the decoded arch class alone. Evidence-labeled `predicted`; the live cache-kind classifier
    // confirms it after load. Informational — emitted whether or not the fit-check proceeds.
    emitFitCheck([ServingSnapshotBridge.snapshotReuseAnnounceLine(for: parsed.profile.modelType)])

    guard decision.shouldProceed else { throw FitCheckRefusal(lines: decision.summaryLines()) }

    // Apply only the two STRICT wins from the sizer: the MLX memory + cache limits (clamped so
    // cache ≤ memory, the invariant the loaders assume). Deliberately do NOT override
    // maxReservedKVBytes: the sizer's KV figure is computed at concurrency=1, but the continuous
    // backend runs multiple slots + a decode reserve, so imposing that exact-boundary byte cap could
    // reject valid concurrent/long requests that the provided (generous, e.g. serve.sh 30%-RAM) cap
    // admits — and --force does not relax runtime caps. Pass the provided reserved-KV through.
    let memory = decision.memoryLimitBytes
    let cache = min(decision.cacheLimitBytes, memory)
    // Only enforce a served-context cap when the operator explicitly asked for one. On the default
    // path leave it nil: the backend then uses the model's native max (proof.maximumContextTokens),
    // identical to prior behavior — no previously-serving request is newly rejected.
    let contextCap: Int? = decision.explicitContextRequested ? decision.servedContext : nil
    return ResolvedServingLimits(
        memoryLimitBytes: memory, cacheLimitBytes: cache,
        maxReservedKVBytes: providedReservedKV, maxContextTokens: contextCap, fitDecision: decision)
}

/// Resolve which directory the serve path actually loads. With `--quant-candidates` set, run the
/// pre-load quant auto-pick over the candidate directories (fit-checked-serve #2 full shape), emit the
/// per-candidate + winner summary, and return the winning directory plus its already-decoded arch (so
/// `resolveServingLimits` need not decode it again). Fail closed when no candidate fits — `--force`
/// does not override an auto-pick in candidates mode (a red-only set refuses; see
/// docs/task-inbox/2026-08-18-quant-auto-pick-policy.md). Without candidates, return the single
/// provided directory unchanged.
private func resolveServedDirectory(
    _ arguments: FastMLXServeArguments, fallback: URL
) throws -> (directory: URL, parsed: ParsedModelArch?) {
    guard !arguments.quantCandidateDirectories.isEmpty else { return (fallback, nil) }
    // Apply the `--tier` serve dial (differentiator #4, item #5) to the ENFORCED candidates path so the
    // dial governs the real load, not only the `--quant-pick-only` dry-run and the single-model path.
    // Thread the tier's context-capping stance: transparent/balanced refuse a memory-bound cap while
    // maxfit caps-and-proceeds. `nil` when no `--tier` was given → allowContextCapping defaults `true`
    // → byte-identical prior behavior. Deliberately do NOT pass the policy's allowedKVTiers here: the
    // serving runtime stores KV in fp16 (int8 runtime KV is metallib-gated), so escalating a LOADED
    // model's KV tier would let the fit-check claim a compressed ceiling the runtime can't honor — an
    // OOM the fit-check exists to prevent. Instead surface the escalation the tier WANTED as an advisory.
    let servePolicy = try servingPolicyForTier(arguments)
    if let note = servePolicy?.conflictAnnotation { emitFitCheck(["serve-tier note: \(note)"]) }
    if let advisory = ServeTierPolicy.enforcedPathKVAdvisory(for: servePolicy) { emitFitCheck([advisory]) }
    let resolution = QuantCandidateResolver.resolve(
        candidateDirectories: arguments.quantCandidateDirectories,
        host: SystemProfile.detectHost(),
        requestedContext: arguments.requestedContext,
        concurrency: arguments.planConcurrency ?? 1,
        allowContextCapping: servePolicy?.allowContextCapping ?? true,
        preference: try quantPickPreference(arguments))
    emitFitCheck(resolution.summaryLines())
    // Surface the measured tool-call reliability overlay (roadmap #4) where the operator ACTUALLY
    // serves, not only on the `--quant-pick-only` dry-run — same advisory STDERR block, same compose
    // call. Display-only: it never influences the pick and the STDOUT startup line stays byte-identical.
    emitQuantReliabilityOverlay(arguments: arguments, pick: resolution.pick)
    guard resolution.shouldProceed, let winner = resolution.winnerDirectory else {
        throw FitCheckRefusal(lines: resolution.summaryLines())
    }
    return (winner, resolution.winnerParsed)
}

/// Load a scalar serving backend with the standard scalar backend configuration. Shared by the
/// explicit `--scalar` route and the continuous→scalar hybrid fallback so both produce an
/// identical `mode=scalar` startup line; callers pass the already-resolved fit-check limits.
private func loadScalarServingBackend(
    arguments: FastMLXServeArguments,
    modelDirectory: URL,
    memoryLimitBytes: Int,
    cacheLimitBytes: Int,
    fitDecision: ServingFitDecision?
) async throws -> PreparedServingBackend {
    let loaded = try await loadScalarServingModel(
        configuration: ScalarServingModelLoadConfiguration(
            launchedModel: arguments.model,
            modelDirectory: modelDirectory,
            memoryLimitBytes: memoryLimitBytes,
            cacheLimitBytes: cacheLimitBytes,
            backendConfiguration: ScalarServingBackendConfiguration(
                defaultMaximumCompletionTokens: 512,
                maximumQueuedRequests: 2,
                queueRetryAfterSeconds: 1,
                mailboxCapacity: .init(
                    maxDeltas: 8,
                    maxBytes: 32 * 1_024)),
            // Thread the requested KV tier into the load path so `selectKVCacheQuant` resolves it
            // fail-closed at cache-construction time. Validated once at startup (run()) and re-validated
            // here as the single source that reaches the runtime; nil (omitted flag) → `.fp16`.
            kvQuantTier: try arguments.kvQuantTier.map { try KVQuantAdvisory.validateTier($0) } ?? .fp16))
    return PreparedServingBackend(
        backend: loaded.backend,
        evidenceSnapshot: nil,
        mode: "scalar",
        launchedModel: arguments.model,
        startupReport: .scalar(loaded.startupReport),
        fitDecision: fitDecision,
        exactMTPReport: nil)
}

private func prepareBackend(
    _ arguments: FastMLXServeArguments
) async throws -> PreparedServingBackend {
    switch arguments.backend {
    case .scripted:
        return PreparedServingBackend(
            backend: ScriptedTransportBackend(),
            evidenceSnapshot: nil,
            mode: "scripted",
            launchedModel: arguments.model,
            startupReport: nil,
            fitDecision: nil,
            exactMTPReport: nil)
    case .scalar(
        let modelDirectory,
        let memoryLimitBytes,
        let cacheLimitBytes
    ):
        let served = try resolveServedDirectory(arguments, fallback: modelDirectory)
        let exactDrafterDirectory: URL?
        if arguments.exactQwen35MTP {
            guard let drafterDirectory = arguments.mtpDrafterDirectory else {
                throw FastMLXServeArgumentError.missingRequiredOption("--mtp-drafter-path")
            }
            exactDrafterDirectory = drafterDirectory
        } else {
            exactDrafterDirectory = nil
        }
        let fitParsed: ParsedModelArch?
        if let exactDrafterDirectory {
            do {
                let targetParsed = try served.parsed ?? ModelConfigDecoder.decodeModelDirectory(
                    served.directory,
                    id: arguments.model)
                fitParsed = try ExactQwen35MTPCompositeFitProfile.make(
                    target: targetParsed,
                    drafterDirectory: exactDrafterDirectory)
            } catch {
                let lines = [
                    "fastmlx-serve fit_check=refused "
                        + "reason=exact_qwen35_mtp_composition_weights_unavailable",
                    "  exact Qwen3.5 MTP requires a complete local target and drafter "
                        + "whose resident safetensor bytes can be sized before load",
                ]
                emitFitCheck(lines)
                throw FitCheckRefusal(lines: lines)
            }
        } else {
            fitParsed = served.parsed
        }
        let limits = try resolveServingLimits(
            modelDirectory: served.directory, arguments: arguments,
            providedMemory: memoryLimitBytes, providedCache: cacheLimitBytes, providedReservedKV: nil,
            preParsed: fitParsed)
        if let drafterDirectory = exactDrafterDirectory {
            let loaded = try await loadExactQwen35MTPServeComposition(
                configuration: ExactQwen35MTPServeCompositionConfiguration(
                    launchedModel: arguments.model,
                    targetDirectory: served.directory,
                    drafterDirectory: drafterDirectory,
                    memoryLimitBytes: limits.memoryLimitBytes,
                    cacheLimitBytes: limits.cacheLimitBytes,
                    scalarBackendConfiguration: ScalarServingBackendConfiguration(
                        defaultMaximumCompletionTokens: 512,
                        maximumQueuedRequests: 2,
                        queueRetryAfterSeconds: 1,
                        mailboxCapacity: .init(
                            maxDeltas: 8,
                            maxBytes: 32 * 1_024)),
                    kvQuantTier: try arguments.kvQuantTier.map { try KVQuantAdvisory.validateTier($0) } ?? .fp16))
            let mode: String
            switch loaded.exactStartupReport.status {
            case .exactSuccess:
                mode = "exact-qwen35-mtp"
            case .scalarFallback:
                mode = "scalar"
            }
            return PreparedServingBackend(
                backend: loaded.backend,
                evidenceSnapshot: nil,
                mode: mode,
                launchedModel: arguments.model,
                startupReport: .scalar(loaded.scalarStartupReport),
                fitDecision: limits.fitDecision,
                exactMTPReport: loaded.exactStartupReport)
        }
        return try await loadScalarServingBackend(
            arguments: arguments,
            modelDirectory: served.directory,
            memoryLimitBytes: limits.memoryLimitBytes,
            cacheLimitBytes: limits.cacheLimitBytes,
            fitDecision: limits.fitDecision)
    case .continuousBatchNoSpec(
        let modelDirectory,
        let memoryLimitBytes,
        let cacheLimitBytes,
        let maxReservedKVBytes
    ):
        let served = try resolveServedDirectory(arguments, fallback: modelDirectory)
        let limits = try resolveServingLimits(
            modelDirectory: served.directory, arguments: arguments,
            providedMemory: memoryLimitBytes, providedCache: cacheLimitBytes,
            providedReservedKV: maxReservedKVBytes, preParsed: served.parsed,
            advisorySlotCount: continuousMaxActiveSlots)
        let loaded: LoadedContinuousServingModel
        do {
            loaded = try await loadContinuousServingModel(
                configuration: ContinuousServingModelLoadConfiguration(
                launchedModel: arguments.model,
                modelDirectory: served.directory,
                memoryLimitBytes: limits.memoryLimitBytes,
                cacheLimitBytes: limits.cacheLimitBytes,
                maxReservedKVBytes: limits.maxReservedKVBytes ?? maxReservedKVBytes,
                maxContextTokens: limits.maxContextTokens,
                maxReservedContextTokens: limits.maxContextTokens,
                coordinatorConfiguration: try ContinuousBatchConfiguration(
                    maxActiveSlots: continuousMaxActiveSlots,
                    maxPrefillSlots: 2,
                    prefillChunkSize: 512,
                    maxQueuedRequests: 8),
                publicationCapacity: 1,
                backendConfiguration: ContinuousServingBackendConfiguration(
                    defaultMaximumCompletionTokens: 512,
                    queueRetryAfterSeconds: 1,
                    mailboxCapacity: .init(
                        maxDeltas: 8,
                        maxBytes: 32 * 1_024),
                    // The continuous route serves dense models; keep the legacy tool-thinking-off
                    // workaround here (QwenLM/Qwen3 #1817). The agentic qwen3_5 family serves on the
                    // scalar route, which respects the template default instead.
                    disableThinkingWhenToolsActive: true),
                // Thread the requested KV tier so `selectKVCacheQuant` resolves it fail-closed at load
                // on the continuous route too (the default serve.sh path). nil (omitted flag) → `.fp16`.
                kvQuantTier: try arguments.kvQuantTier.map { try KVQuantAdvisory.validateTier($0) } ?? .fp16,
                // Opt-in qwen3_5 hybrid admission (--allow-hybrid-qwen35). Off (default) → the proof
                // throws unsupportedModelFamily and the catch below falls back to scalar serving.
                allowHybridQwen35: arguments.allowHybridQwen35))
        } catch DenseContinuousBatchRuntimeError.unsupportedModelFamily(let family)
            where isScalarHybridServingFamily(family)
        {
            // The continuous route only supports qwen3; the proof fails closed before any
            // weight load or global Memory mutation (MLXContinuousServing.verifying runs first).
            // For families the scalar route is live-proven to serve (currently qwen3_5's hybrid
            // GatedDeltaNet-linear/full-attention layout), keep the operator's one-command
            // `serve.sh --model <repo>` working by falling back to scalar serving. The resolved
            // fit-check limits are route-agnostic (arch-derived) and reused as-is; scalar ignores
            // the reservedKV/context fields. Every non-allowlisted family rethrows, so
            // `verifying` stays the sole authority on continuous support.
            FileHandle.standardError.write(
                Data((scalarHybridFallbackAnnounceLine(modelType: family) + "\n").utf8))
            return try await loadScalarServingBackend(
                arguments: arguments,
                modelDirectory: served.directory,
                memoryLimitBytes: limits.memoryLimitBytes,
                cacheLimitBytes: limits.cacheLimitBytes,
                fitDecision: limits.fitDecision)
        }
        // The opt-in hybrid family was actually admitted onto the continuous route (the proof carried
        // qwen3_5 through rather than throwing). Announce it explicitly so the operator can tell a
        // successful hybrid-continuous serve from a silent scalar fallback (same machine-readable style).
        if loaded.startupReport.modelFamily == .qwen35 {
            FileHandle.standardError.write(
                Data((hybridQwen35ContinuousAdmissionAnnounceLine(modelType: "qwen3_5") + "\n").utf8))
        }
        return PreparedServingBackend(
            backend: loaded.backend,
            evidenceSnapshot: {
                let snapshot = await loaded.backend.snapshot()
                // Measured-vs-modeled drift (differentiator #2): compare the sizer's modeled peak
                // (+ term breakdown) against the live allocator high-water mark. Present only when the
                // fit-check ran; the peak accumulates under KV load, so a metrics-query snapshot is the
                // honest "peak-so-far vs modeled" comparison (never a startup trivial-conservative).
                let drift = limits.fitDecision.map { decision in
                    FitCheckMeasuredReport(
                        prediction: decision.prediction,
                        measuredPeakBytes: snapshot.mlxPeakBytes,
                        measuredActiveBytes: snapshot.mlxActiveBytes,
                        measuredCacheBytes: snapshot.mlxCacheBytes)
                }
                return try ServingEvidence.ResourceSnapshot(
                    activeRequests: snapshot.activeRequests,
                    coordinatorSlots: snapshot.coordinatorSlots,
                    reservedKVBytes: snapshot.reservedKVBytes,
                    maxReservedKVBytes: snapshot.maxReservedKVBytes,
                    mlxActiveBytes: snapshot.mlxActiveBytes,
                    mlxCacheBytes: snapshot.mlxCacheBytes,
                    mlxPeakBytes: snapshot.mlxPeakBytes,
                    fitModeledPeakBytes: drift?.modeledPeakBytes,
                    fitMeasuredPeakBytes: drift?.measuredPeakBytes,
                    fitDriftVerdict: drift?.drift.rawValue,
                    fitDriftFraction: drift?.deltaFraction,
                    fitModeledWeightsBytes: drift?.modeledWeightsBytes,
                    fitModeledKVBytes: drift?.modeledKVBytes,
                    fitModeledTransientBytes: drift?.modeledTransientBytes,
                    fitModeledHeadroomBytes: drift?.modeledHeadroomBytes)
            },
            mode: "continuous-batch-no-spec",
            launchedModel: arguments.model,
            startupReport: .continuous(loaded.startupReport),
            fitDecision: limits.fitDecision,
            exactMTPReport: nil)
    case nil:
        preconditionFailure("help is the only invocation without a backend")
    }
}

private final class ScriptedTransportBackend: ServingGenerationBackend, Sendable {
    func start(_ request: OpenAIChatCompletionRequest) async throws -> ServingGenerationHandle {
        let mailbox = BoundedDeltaMailbox(
            capacity: .init(maxDeltas: 2, maxBytes: 32 * 1_024))
        let lease = ServingRequestLease(
            id: ServingRequestID("scripted-\(UUID().uuidString)"),
            onCancel: { [mailbox] in
                await mailbox.cancel(.clientDisconnected)
            })
        let handle = ServingGenerationHandle(
            responseID: "chatcmpl-scripted-\(UUID().uuidString)",
            created: Int(Date().timeIntervalSince1970),
            model: request.model,
            route: .scriptedTransport,
            mailbox: mailbox,
            lease: lease)

        Task {
            do {
                try await mailbox.send(
                    .text("fast-mlx scripted transport is ready; no model is loaded."))
                try await mailbox.send(
                    .completion(
                        ServingGenerationCompletion(
                            finishReason: .stop,
                            usage: OpenAIChatUsage(
                                promptTokens: 0,
                                completionTokens: 0))))
                await mailbox.finish()
            } catch {
                // The request lease owns terminal and cancellation state.
            }
        }
        return handle
    }
}

private func waitForShutdownSignal() async {
    signal(SIGINT, SIG_IGN)
    signal(SIGTERM, SIG_IGN)
    signal(SIGPIPE, SIG_IGN)

    let (signals, continuation) = AsyncStream<Int32>.makeStream(
        bufferingPolicy: .bufferingNewest(1))
    let interruptSource = DispatchSource.makeSignalSource(signal: SIGINT)
    let terminateSource = DispatchSource.makeSignalSource(signal: SIGTERM)
    interruptSource.setEventHandler {
        continuation.yield(SIGINT)
    }
    terminateSource.setEventHandler {
        continuation.yield(SIGTERM)
    }
    interruptSource.resume()
    terminateSource.resume()

    for await _ in signals {
        break
    }
    continuation.finish()
    interruptSource.cancel()
    terminateSource.cancel()
}
