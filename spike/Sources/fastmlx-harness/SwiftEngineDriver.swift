import Foundation
import HarnessCore
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import SpikeCore
import Tokenizers

/// Unsupported `RunConfig` features fail loudly — a "measurement" must never silently
/// measure something other than what was asked for.
enum SwiftEngineDriverError: Error, CustomStringConvertible {
    case unsupportedConfig(String)
    var description: String {
        switch self {
        case .unsupportedConfig(let what): return "SwiftEngineDriver: unsupported config: \(what)"
        }
    }
}

/// Exact result of one restricted-choice scoring forward. Only actor-safe CPU values cross the
/// engine boundary; the MLX arrays and concrete cache objects remain actor-confined.
struct TaskChoiceLogitsResult: Sendable {
    let logits: [Float]
    let engagement: EngagementCounters
}

/// Actor-safe output of the private KVTuner candidate generation path. The generated token IDs
/// are paired with telemetry captured from the exact decoder/cache instance before its MLX arrays
/// leave actor isolation.
struct KVTunerCandidateRunResult: Sendable {
    let promptOrdinal: Int
    let promptTokenIDsSHA256: String
    let tokens: [Int]
    let finishReason: KVTunerCandidateFinishReason
    let telemetry: KVTunerCandidateKVCacheTelemetry
}

/// Single-owner actor over the spike's compiled decode core. Owns BOTH the model and the
/// `CompiledMLXDecoder` (one isolation region) because `logprobs` needs direct forward access
/// to full-vocab logits, which the token-only `Decoder` protocol deliberately doesn't expose.
///
/// The generate path preserves the spike's compiled-step + no-sync-readback design by
/// construction: it only calls `CompiledMLXDecoder.prefill`/`.step`, where that pattern lives
/// (the next step's compiled forward is asyncEval'd before the current token's `.item()`).
actor HarnessEngineActor {
    private let model: any LanguageModel
    /// Captured from the same model directory and live tokenizer used to construct `model`.
    /// Candidate policies must reconcile against this independent identity inside the actor.
    private let kvtunerRuntimeIdentity: KVTunerCandidateRuntimeIdentity?
    /// One decoder per KV-cache kind, built lazily and kept alive so each kind's compiled
    /// step function survives across runs (an fp16 baseline and a TurboQuant candidate can
    /// alternate within one verify invocation without retracing either).
    private var decoders: [KVCacheKind: CompiledMLXDecoder] = [:]
    /// Largest affine scoring allocation observed in this actor. Scoring caches are ephemeral,
    /// so their scalar geometry/byte evidence must be retained before the arrays are released.
    private var maximumAffineScoringTelemetry: AffineKVCacheTelemetry?
    /// Heterogeneous affine policies are keyed by the immutable exact-byte selection, so two
    /// same-cell artifacts cannot share retained scoring evidence.
    private var maximumKVTunerScoringTelemetry: [
        KVTunerRuntimeSelection: KVTunerKVCacheTelemetry
    ] = [:]
    /// KVarN cells share packed geometry but not codec work. Key retained scalar evidence by
    /// runtime cell so an i8 scoring pass can never be mislabeled as i16 (or vice versa).
    private var maximumKVarNScoringTelemetry: [
        KVarNKVRuntimeCell: KVarNKVCacheTelemetry
    ] = [:]

    init(
        model: sending any LanguageModel,
        kvtunerRuntimeIdentity: KVTunerCandidateRuntimeIdentity? = nil
    ) {
        self.model = model
        self.kvtunerRuntimeIdentity = kvtunerRuntimeIdentity
    }

    /// Greedy decode via the compiled core. The returned ids INCLUDE a terminal eos if one is
    /// produced (mirroring scripts/harness_reference.py exactly, so token streams diff cleanly).
    /// Quantized-cache engagement is read AFTER timing so its synchronization cannot skew the
    /// benchmark. Affine and KVarN return Sendable scalar snapshots; TurboQuant retains its
    /// legacy token marker until that format's evidence schema is generalized.
    func generate(prompt: [Int], maxTokens: Int, eos: Int, kvCache kind: KVCacheKind)
        -> (
            tokens: [Int], submitTime: Double, tokenTimes: [Double],
            turboQuantTokens: Int?, affineTelemetry: AffineKVCacheTelemetry?,
            kvtunerTelemetry: KVTunerKVCacheTelemetry?,
            kvarnTelemetry: KVarNKVCacheTelemetry?
        )
    {
        if case .kvtunerCandidate = kind {
            preconditionFailure(
                "KVTuner candidates require the private qualification path")
        }
        if decoders[kind] == nil {
            decoders[kind] = CompiledMLXDecoder(model: model, kvCache: kind)
        }
        var decoder = decoders[kind]!
        defer { decoders[kind] = decoder }
        decoder.reset() // in-place KV reset: compiled graph stays valid across runs
        let submitTime = Date().timeIntervalSinceReferenceDate
        guard maxTokens > 0 else {
            return ([], submitTime, [], nil, nil, nil, nil)
        }
        var tokens: [Int] = []
        var tokenTimes: [Double] = []
        var tok = decoder.prefill(prompt)
        tokens.append(tok)
        tokenTimes.append(Date().timeIntervalSinceReferenceDate)
        while tokens.count < maxTokens && tok != eos {
            tok = decoder.step(last: tok)
            tokens.append(tok)
            tokenTimes.append(Date().timeIntervalSinceReferenceDate)
        }
        let turboQuantTokens = decoder.turboQuantCachedTokens()
        let affineTelemetry = decoder.affineKVTelemetry()
        let kvtunerTelemetry = decoder.kvtunerKVTelemetry()
        let kvarnTelemetry = decoder.kvarnKVTelemetry()
        if case .turboQuant = kind {
            // A TurboQuant run whose decoder somehow holds an fp16 cache is a plumbing bug,
            // not a measurement — fail loudly (mirrors requireSupported's contract).
            precondition(turboQuantTokens != nil, "TurboQuant tier requested but the quantized cache did not engage")
        }
        if case .affine = kind {
            precondition(
                affineTelemetry != nil,
                "affine tier requested but the affine cache did not engage")
        }
        if case .kvtuner(let selection) = kind {
            precondition(
                kvtunerTelemetry?.artifactSHA256
                    == selection.artifactSHA256,
                "KVTuner tier requested but matching schedule telemetry did not engage")
        }
        if case .kvarn(let cell) = kind {
            precondition(
                kvarnTelemetry?.tier == cell.tier
                    && kvarnTelemetry?.iterations == cell.iterations
                    && kvarnTelemetry?.executionMode == .uncompiledCorrectness,
                "KVarN tier requested but matching KVarN telemetry did not engage")
        }
        return (
            tokens, submitTime, tokenTimes, turboQuantTokens, affineTelemetry,
            kvtunerTelemetry, kvarnTelemetry)
    }

    /// Greedy candidate generation for exhaustive KVTuner qualification only. This has no
    /// `RunConfig` or tier-string entry point, cannot be combined with speculation, and returns
    /// the exact policy/cache receipt needed by schema-2 candidate evidence.
    func evaluateKVTunerCandidateCohort(
        prompts: [[Int]],
        maxTokens: Int,
        policy: KVTunerCandidateRuntimePolicy
    ) throws -> [KVTunerCandidateRunResult] {
        precondition(!prompts.isEmpty, "KVTuner candidate cohort must be nonempty")
        precondition(
            prompts.allSatisfy { !$0.isEmpty },
            "KVTuner candidate prompts must be nonempty")
        precondition(maxTokens > 0, "KVTuner candidate maxTokens must be positive")
        guard let kvtunerRuntimeIdentity else {
            throw KVTunerCandidateRuntimeIdentityError.missingRuntimeIdentity
        }
        _ = try kvtunerRuntimeIdentity.validate(runtimePolicy: policy)
        let kind = KVCacheKind.kvtunerCandidate(policy)
        // The decoder is local to exactly one complete prompt cohort. Prompts reuse its compiled
        // graph and monotonically grown allocation, but retries and later candidates start from a
        // fresh cache so first-row capacity evidence is reproducible and memory cannot accumulate
        // with the number of candidates visited.
        var decoder = CompiledMLXDecoder(model: model, kvCache: kind)
        var results: [KVTunerCandidateRunResult] = []
        results.reserveCapacity(prompts.count)
        for (promptOrdinal, prompt) in prompts.enumerated() {
            decoder.reset()
            var tokens: [Int] = []
            var token = decoder.prefill(prompt)
            tokens.append(token)
            while tokens.count < maxTokens
                && token != kvtunerRuntimeIdentity.eosTokenID
            {
                token = decoder.step(last: token)
                tokens.append(token)
            }
            guard let telemetry = decoder.kvtunerCandidateKVTelemetry() else {
                preconditionFailure(
                    "KVTuner candidate decoder did not return candidate telemetry")
            }
            precondition(
                telemetry.runtimePolicySHA256
                    == policy.runtimePolicySHA256,
                "KVTuner candidate telemetry does not match its runtime policy")
            let finishReason: KVTunerCandidateFinishReason
            if tokens.last == kvtunerRuntimeIdentity.eosTokenID {
                finishReason = .endOfSequence
            } else {
                precondition(
                    tokens.count == maxTokens,
                    "KVTuner candidate stopped without EOS or exhausting its generation budget")
                finishReason = .generationBudgetExhausted
            }
            results.append(KVTunerCandidateRunResult(
                promptOrdinal: promptOrdinal,
                promptTokenIDsSHA256: taskTokenIDsSHA256(prompt),
                tokens: tokens,
                finishReason: finishReason,
                telemetry: telemetry))
        }
        return results
    }

    /// Captures KVTuner's offline per-layer metrics inside the model's isolation region. The
    /// exact config comes from the source snapshot sampled around this live model load; callers
    /// provide only Sendable token IDs and receive only scalar samples.
    func captureKVTunerSensitivity(
        promptTokenIDs: [[Int]],
        groupSize: Int,
        expectedRuntimeIdentity: KVTunerCandidateRuntimeIdentity
    ) throws -> [KVTunerSensitivitySample] {
        guard let kvtunerRuntimeIdentity else {
            throw KVTunerCandidateRuntimeIdentityError.missingRuntimeIdentity
        }
        guard kvtunerRuntimeIdentity == expectedRuntimeIdentity else {
            throw KVTunerCandidateRuntimeIdentityError
                .sourceIdentityChangedDuringModelLoad
        }
        return try KVTunerSensitivityCapture.capture(
            model: model,
            exactModelConfigData:
                kvtunerRuntimeIdentity.exactModelConfigData,
            promptTokenIDs: promptTokenIDs,
            groupSize: groupSize,
            precisionPairs:
                KVTunerSensitivityArtifact.canonicalPrecisionPairs)
    }

    /// Speculative-decoding generate (PLD first): routes to `CompiledMLXDecoder.generateSpec`,
    /// which drafts from the context, batch-verifies, accept-walks, and rolls the KV back —
    /// byte-identical to the plain greedy loop at temp 0 by construction. Same decoder-per-kind
    /// reuse as `generate` (compiled step + compiled verify survive across runs; in-place reset).
    func generateSpec(prompt: [Int], maxTokens: Int, eos: Int, kvCache kind: KVCacheKind, spec: SpecDecodeConfig)
        -> (tokens: [Int], submitTime: Double, tokenTimes: [Double], stats: SpecDecodeStats)
    {
        if decoders[kind] == nil {
            decoders[kind] = CompiledMLXDecoder(model: model, kvCache: kind)
        }
        var decoder = decoders[kind]!
        defer { decoders[kind] = decoder }
        decoder.reset() // in-place KV reset: compiled graph stays valid across runs
        return decoder.generateSpec(prompt: prompt, maxTokens: maxTokens, eos: eos, spec: spec)
    }

    /// Full-vocab RAW LOGITS per generated position at temp=0 — the `EngineDriver.logprobs`
    /// contract: index == token id, length == vocab, NOT top-k, NOT softmaxed. Runs the plain
    /// (uncompiled) forward on a fresh cache: this is the measurement path, not the perf path,
    /// and the per-position full-vocab readback is inherently synchronous anyway.
    /// fp16 -> float32 conversion is exact, so argmax over a returned row reproduces the
    /// greedy token chosen at that position.
    func logprobs(prompt: [Int], maxTokens: Int, eos: Int, kvCache kind: KVCacheKind) -> [[Float]] {
        let cache = makeScoringCache(kind: kind, capacity: prompt.count + max(maxTokens, 0))
        var rows: [[Float]] = []
        var y = MLXArray(prompt).reshaped([1, prompt.count])
        for _ in 0..<max(maxTokens, 0) {
            let logits = model(y, cache: cache)
            let last = logits[0..., -1, 0...] // [1, vocab] raw logits
            rows.append(last.asType(.float32).asArray(Float.self))
            let tok = argMax(last, axis: -1).item(Int.self)
            if tok == eos { break }
            y = MLXArray([tok]).reshaped([1, 1])
        }
        captureQuantizedScoringTelemetry(kind, cache: cache, minTokens: prompt.count)
        return rows
    }

    /// Scores the four-choice task head with one fresh full-prompt cache and returns engagement
    /// from that exact local cache. The ordinary scoring accessors retain maximum telemetry for
    /// KL accounting; they are intentionally not used here because a prior larger run could make
    /// a silently unengaged task row appear valid.
    func taskChoiceLogits(
        prompt: [Int], kvCache kind: KVCacheKind
    ) -> TaskChoiceLogitsResult {
        precondition(!prompt.isEmpty, "task scoring requires a nonempty prompt")
        let cache = makeScoringCache(kind: kind, capacity: prompt.count)
        let ids = MLXArray(prompt).reshaped([1, prompt.count])
        let logits = model(ids, cache: cache)
        let last = logits[0..., -1, 0...]
        let row = last.asType(.float32).asArray(Float.self)
        var counts: [String: Int] = [:]

        switch kind {
        case .fp16:
            break
        case .affine(let tier):
            let affineCaches = cache.compactMap { $0 as? AffineKVCache }
            precondition(
                affineCaches.count == cache.count,
                "affine task-scoring cache contains a different cache type")
            let telemetry = AffineKVCacheTelemetry.capture(
                tier: tier, caches: affineCaches)
            precondition(
                telemetry.cachedTokens == prompt.count,
                "affine task-scoring cache did not consume the full prompt")
            counts["scoring_cached_tokens"] = telemetry.cachedTokens
        case .kvtuner(let selection):
            let affineCaches = cache.compactMap { $0 as? AffineKVCache }
            precondition(
                affineCaches.count == cache.count,
                "KVTuner task-scoring cache contains a different cache type")
            let telemetry: KVTunerKVCacheTelemetry
            do {
                telemetry = try KVTunerKVCacheTelemetry.capture(
                    selection: selection, caches: affineCaches)
            } catch {
                preconditionFailure(
                    "KVTuner task-scoring telemetry mismatch: \(error)")
            }
            precondition(
                telemetry.cachedTokens == prompt.count,
                "KVTuner task-scoring cache did not consume the full prompt")
            counts["scoring_cached_tokens"] = telemetry.cachedTokens
            counts["scoring_kvtuner_layers"] = telemetry.layerCount
        case .kvtunerCandidate:
            preconditionFailure(
                "KVTuner candidates are generation-only qualification inputs")
        case .kvarn(let cell):
            let kvarnCaches = cache.compactMap { $0 as? KVarNKVCache }
            precondition(
                kvarnCaches.count == cache.count,
                "KVarN task-scoring cache contains a different cache type")
            let telemetry = KVarNKVCacheTelemetry.capture(caches: kvarnCaches)
            precondition(
                telemetry.tier == cell.tier
                    && telemetry.iterations == cell.iterations
                    && telemetry.executionMode == .uncompiledCorrectness
                    && telemetry.cachedTokens == prompt.count,
                "KVarN task-scoring telemetry does not match its runtime cell")
            counts["scoring_cached_tokens"] = telemetry.cachedTokens
            counts["scoring_kvarn_completed_tiles"] =
                telemetry.completedTileCount
            counts["scoring_kvarn_compressed_tokens"] =
                telemetry.compressedTokens
        case .turboQuant:
            preconditionFailure(
                "TurboQuant is outside the authenticated task-coherence tier map")
        }
        return TaskChoiceLogitsResult(
            logits: row, engagement: EngagementCounters(counts))
    }

    /// TEACHER-FORCED `logprobs`: row i is the next-token distribution given
    /// context = prompt + forced[0..<i]; forced[i] is fed as the next input regardless of
    /// argmax, and eos does NOT stop the loop (the forced continuation already encodes where
    /// its producer stopped). Exactly forced.count rows. Same measurement path as the
    /// free-running variant (plain forward, fresh cache) — the perf path's compiled-step +
    /// no-sync-readback design lives untouched in `generate`.
    func teacherForcedLogprobs(prompt: [Int], forced: [Int], kvCache kind: KVCacheKind) -> [[Float]] {
        scoreForced(prompt: prompt, forced: forced, wanted: nil, kind: kind)
    }

    /// Sampled variant: same chunked forward over the full forced continuation (causal decoding
    /// requires every intermediate token as context regardless), but only converts+keeps a
    /// full-vocab row at `positions` — a long-context entry can be thousands of positions, and
    /// materializing every row would be ~0.6MB/row x thousands x 2 drivers. `positions` must be
    /// ascending (evenlySpacedPositions's contract); rows are returned in that order.
    func teacherForcedLogprobsAtPositions(
        prompt: [Int], forced: [Int], positions: [Int], kvCache kind: KVCacheKind
    ) -> [[Float]] {
        scoreForced(prompt: prompt, forced: forced, wanted: positions, kind: kind)
    }

    /// CHUNKED teacher-forced scoring (`forcedScoringPlan` in HarnessCore holds the pure
    /// bookkeeping): multi-token chunk forwards instead of one forward per forced token.
    ///
    /// WHY (the ~7K-context SIGKILL root cause): single-token stepping makes every step's
    /// transient buffers slightly LARGER than the last step's (the stock cache returns growing
    /// K/V slices), so MLX's buffer cache can never reuse a freed buffer and grows as
    /// O(context²) — measured 43GB of dead cache (active flat at 17GB) by position 6750 on
    /// Qwen3-32B-4bit, which, with the Python reference process ballooning identically, is
    /// exactly the ~6.7–7.1K jetsam SIGKILL ceiling the harness hit.
    ///
    /// The fix is BOTH layers, each necessary: chunking cuts the number of growing-transient
    /// events by the chunk factor and scores at prefill speed instead of decode speed (24K
    /// tokens: ~1 min/side instead of ~10), but the materialized K/V slices still grow chunk to
    /// chunk, so unbounded the cache still reaches ~62GB by 16K context (measured, ctxprobe
    /// `score` mode) — it is the allocator-cache bound in `loadSwiftDriver` that guarantees
    /// flat memory by evicting those unreusable buffers (32K tokens: 33.8GB peak footprint,
    /// cache pinned at 8GB). The per-chunk `eval` bounds the lazy graph so pending work cannot
    /// pile up across chunks.
    private func scoreForced(prompt: [Int], forced: [Int], wanted: [Int]?, kind: KVCacheKind) -> [[Float]] {
        let input = prompt + forced.dropLast()
        let cache = makeScoringCache(kind: kind, capacity: input.count)
        let plan = forcedScoringPlan(
            promptCount: prompt.count, forcedCount: forced.count,
            wantedPositions: wanted, chunkSize: Self.scoringChunkSize)
        var rows: [[Float]] = []
        for chunk in plan.chunks {
            let ids = MLXArray(Array(input[chunk.inputRange])).reshaped([1, chunk.inputRange.count])
            let logits = model(ids, cache: cache)
            eval(logits)
            for sel in chunk.rows {
                rows.append(logits[0..., sel.localIndex, 0...].asType(.float32).asArray(Float.self))
            }
        }
        captureQuantizedScoringTelemetry(kind, cache: cache, minTokens: input.count)
        return rows
    }

    /// Scoring-path cache selection (Task 7): the MEASUREMENT forwards must run the same KV
    /// tier the config asked for, not silently fp16 — Phase 3's KL/ppl numbers come through
    /// here, not through the compiled decode path. fp16 keeps the stock model cache; affine,
    /// KVarN, and TurboQuant tiers get their requested concrete cache per layer, sized for the
    /// whole pass up front (scoring knows its total length; no chunked growth needed).
    private func makeScoringCache(kind: KVCacheKind, capacity: Int) -> [any KVCache] {
        switch kind {
        case .fp16:
            return model.newCache(parameters: nil)
        case .affine, .kvtuner, .turboQuant, .kvarn:
            let layerCount = model.newCache(parameters: nil).count
            do {
                return try kind.makeCaches(
                    layerCount: layerCount,
                    capacity: max(capacity, 1))
            } catch {
                preconditionFailure(
                    "scoring KV-cache policy does not match the loaded model: \(error)")
            }
        case .kvtunerCandidate:
            preconditionFailure(
                "KVTuner candidates are unavailable to generic scoring paths")
        }
    }

    /// Engagement backstop for the scoring paths: a requested lossy cache must have the
    /// matching concrete type and must have cached every scored position. A silent fp16
    /// fallback here would make the quality evidence measure the wrong thing.
    private func captureQuantizedScoringTelemetry(
        _ kind: KVCacheKind, cache: [any KVCache], minTokens: Int
    ) {
        switch kind {
        case .fp16:
            return
        case .affine(let tier):
            guard let affine = cache.first as? AffineKVCache, affine.offset >= minTokens else {
                preconditionFailure("affine tier requested but the affine scoring cache did not engage")
            }
            let affineCaches = cache.compactMap { $0 as? AffineKVCache }
            precondition(
                affineCaches.count == cache.count,
                "affine scoring cache contains a different cache type")
            let telemetry = AffineKVCacheTelemetry.capture(
                tier: tier, caches: affineCaches)
            if maximumAffineScoringTelemetry.map({
                telemetry.capacityTokens > $0.capacityTokens
            }) ?? true {
                maximumAffineScoringTelemetry = telemetry
            }
        case .kvtuner(let selection):
            guard let affine = cache.first as? AffineKVCache,
                affine.offset >= minTokens
            else {
                preconditionFailure(
                    "KVTuner requested but its scoring cache did not engage")
            }
            let affineCaches = cache.compactMap { $0 as? AffineKVCache }
            precondition(
                affineCaches.count == cache.count,
                "KVTuner scoring cache contains a different cache type")
            let telemetry: KVTunerKVCacheTelemetry
            do {
                telemetry = try KVTunerKVCacheTelemetry.capture(
                    selection: selection, caches: affineCaches)
            } catch {
                preconditionFailure(
                    "KVTuner scoring telemetry mismatch: \(error)")
            }
            if maximumKVTunerScoringTelemetry[selection].map({
                telemetry.capacityTokens > $0.capacityTokens
            }) ?? true {
                maximumKVTunerScoringTelemetry[selection] = telemetry
            }
        case .kvtunerCandidate:
            preconditionFailure(
                "KVTuner candidates are unavailable to generic scoring paths")
        case .turboQuant:
            guard let tq = cache.first as? TurboQuantKVCache, tq.offset >= minTokens else {
                preconditionFailure("TurboQuant tier requested but the quantized scoring cache did not engage")
            }
        case .kvarn(let cell):
            guard let kvarn = cache.first as? KVarNKVCache,
                kvarn.offset >= minTokens
            else {
                preconditionFailure("KVarN tier requested but the KVarN scoring cache did not engage")
            }
            let kvarnCaches = cache.compactMap { $0 as? KVarNKVCache }
            precondition(
                kvarnCaches.count == cache.count,
                "KVarN scoring cache contains a different cache type")
            let telemetry = KVarNKVCacheTelemetry.capture(caches: kvarnCaches)
            precondition(
                telemetry.tier == cell.tier
                    && telemetry.iterations == cell.iterations
                    && telemetry.executionMode == .uncompiledCorrectness,
                "KVarN scoring telemetry does not match its requested runtime cell")
            if maximumKVarNScoringTelemetry[cell].map({
                telemetry.capacityTokens > $0.capacityTokens
            }) ?? true {
                maximumKVarNScoringTelemetry[cell] = telemetry
            }
        }
    }

    func affineScoringTelemetry() -> AffineKVCacheTelemetry? {
        maximumAffineScoringTelemetry
    }

    func kvtunerScoringTelemetry(
        for selection: KVTunerRuntimeSelection
    ) -> KVTunerKVCacheTelemetry? {
        maximumKVTunerScoringTelemetry[selection]
    }

    func kvarnScoringTelemetry(
        for cell: KVarNKVRuntimeCell
    ) -> KVarNKVCacheTelemetry? {
        maximumKVarNScoringTelemetry[cell]
    }

    /// 512 balances per-chunk transient size (a [1, 512, vocab] fp16 logits buffer ~150MB)
    /// against forward-call count; matches mlx-lm's default prefill step size.
    private static let scoringChunkSize = 512
}

/// In-process `EngineDriver` over the compiled decode core — the only MLX-touching harness impl.
struct SwiftEngineDriver: EngineDriver {
    let engine: HarnessEngineActor
    let eos: Int

    func generate(prompt: [Int], config: RunConfig) async throws -> RunResult {
        let kind = try Self.cacheKind(config)
        if let spec = try Self.specConfig(config) {
            let out = await engine.generateSpec(
                prompt: prompt, maxTokens: config.maxTokens, eos: eos, kvCache: kind, spec: spec)
            // Engagement telemetry for the spec triad: `spec_drafted` is the marker proving
            // drafting actually happened (byte-identical output with zero drafts would be a
            // vacuous equivalence "pass"); accepted/steps feed the measurement verdict.
            let counts = [
                "decode": out.tokens.count,
                "spec_drafted": out.stats.drafted,
                "spec_accepted": out.stats.accepted,
                "spec_verify_steps": out.stats.verifySteps,
                "spec_normal_steps": out.stats.normalSteps,
                "spec_gate_disabled_steps": out.stats.gateDisabledSteps,
            ]
            return RunResult(
                tokens: out.tokens,
                engagement: .init(counts),
                acceptanceRate: out.stats.acceptanceRate,
                submitTime: out.submitTime,
                tokenTimes: out.tokenTimes)
        }
        let out = await engine.generate(prompt: prompt, maxTokens: config.maxTokens, eos: eos, kvCache: kind)
        var counts = ["decode": out.tokens.count]
        if let tq = out.turboQuantTokens {
            // In-graph cached-token count from the quantized cache — the lossy triad's
            // engagement marker (delta-checked by `verify`; absent on fp16 runs).
            counts["turboquant_tokens"] = tq
        }
        if let affine = out.affineTelemetry {
            counts["affine_tokens"] = affine.cachedTokens
            counts["affine_payload_bytes"] = affine.payloadBytes
            counts["affine_metadata_bytes"] = affine.metadataBytes
            counts["affine_control_bytes"] = affine.controlBytes
            counts["affine_workspace_bytes"] = affine.materializationWorkspaceBytes
        }
        if let kvtuner = out.kvtunerTelemetry {
            counts["kvtuner_tokens"] = kvtuner.cachedTokens
            counts["kvtuner_layers"] = kvtuner.layerCount
            counts["kvtuner_payload_bytes"] = kvtuner.payloadBytes
            counts["kvtuner_metadata_bytes"] = kvtuner.metadataBytes
            counts["kvtuner_control_bytes"] = kvtuner.controlBytes
            counts["kvtuner_workspace_bytes"] =
                kvtuner.materializationWorkspaceBytes
        }
        if let kvarn = out.kvarnTelemetry {
            counts["kvarn_tokens"] = kvarn.cachedTokens
            counts["kvarn_completed_tiles"] = kvarn.completedTileCount
            counts["kvarn_compressed_tokens"] = kvarn.compressedTokens
            counts["kvarn_payload_bytes"] = kvarn.payloadBytes
            counts["kvarn_metadata_bytes"] = kvarn.metadataBytes
            counts["kvarn_alignment_padding_bytes"] = kvarn.alignmentPaddingBytes
            counts["kvarn_fp16_sink_bytes"] = kvarn.fp16SinkBytes
            counts["kvarn_fp16_tail_bytes"] = kvarn.fp16TailBytes
            counts["kvarn_control_bytes"] = kvarn.controlBytes
            counts["kvarn_workspace_bytes"] = kvarn.materializationWorkspaceBytes
            counts["kvarn_codec_iterations"] = kvarn.iterations
            counts["kvarn_uncompiled_correctness"] =
                kvarn.executionMode == .uncompiledCorrectness ? 1 : 0
        }
        return RunResult(
            tokens: out.tokens,
            engagement: .init(counts),
            acceptanceRate: nil, // plain (non-speculative) decode
            submitTime: out.submitTime,
            tokenTimes: out.tokenTimes)
    }

    func logprobs(prompt: [Int], config: RunConfig) async throws -> [[Float]] {
        let kind = try Self.cacheKind(config, allowSpec: false)
        return await engine.logprobs(prompt: prompt, maxTokens: config.maxTokens, eos: eos, kvCache: kind)
    }

    func taskChoiceLogits(
        prompt: [Int], config: RunConfig
    ) async throws -> TaskChoiceLogitsResult {
        guard !prompt.isEmpty else {
            throw SwiftEngineDriverError.unsupportedConfig(
                "empty task-scoring prompt")
        }
        guard config.maxTokens == 1 else {
            throw SwiftEngineDriverError.unsupportedConfig(
                "maxTokens=\(config.maxTokens) on one-position task scoring")
        }
        let kind = try Self.cacheKind(config, allowSpec: false)
        if case .turboQuant = kind {
            throw SwiftEngineDriverError.unsupportedConfig(
                "TurboQuant has no authenticated task-coherence cell")
        }
        return await engine.taskChoiceLogits(prompt: prompt, kvCache: kind)
    }

    /// Private, policy-typed qualification seam. Unlike `generate(config:)`, there is no string
    /// parsing or user-selectable dial route which could execute an unevaluated candidate.
    func evaluateKVTunerCandidateCohort(
        prompts: [[Int]],
        maxTokens: Int,
        policy: KVTunerCandidateRuntimePolicy
    ) async throws -> [KVTunerCandidateRunResult] {
        guard !prompts.isEmpty,
            prompts.allSatisfy({ !$0.isEmpty })
        else {
            throw SwiftEngineDriverError.unsupportedConfig(
                "empty KVTuner candidate cohort or prompt")
        }
        guard maxTokens > 0 else {
            throw SwiftEngineDriverError.unsupportedConfig(
                "KVTuner candidate maxTokens=\(maxTokens)")
        }
        return try await engine.evaluateKVTunerCandidateCohort(
            prompts: prompts,
            maxTokens: maxTokens,
            policy: policy)
    }

    /// Qualification-only sensitivity capture. Canonical cardinality and group sizes are
    /// enforced before entering the actor so partial or unsupported experiments cannot be
    /// serialized as protocol evidence.
    func captureKVTunerSensitivity(
        prompts: [[Int]],
        groupSize: Int,
        expectedRuntimeIdentity: KVTunerCandidateRuntimeIdentity
    ) async throws -> [KVTunerSensitivitySample] {
        guard prompts.count
                == KVTunerSensitivityArtifact.requiredSensitivityPromptCount
        else {
            throw SwiftEngineDriverError.unsupportedConfig(
                "KVTuner sensitivity prompt count=\(prompts.count)")
        }
        guard prompts.allSatisfy({ !$0.isEmpty }) else {
            throw SwiftEngineDriverError.unsupportedConfig(
                "empty KVTuner sensitivity prompt")
        }
        guard [64, 128].contains(groupSize) else {
            throw SwiftEngineDriverError.unsupportedConfig(
                "KVTuner sensitivity groupSize=\(groupSize)")
        }
        return try await engine.captureKVTunerSensitivity(
            promptTokenIDs: prompts,
            groupSize: groupSize,
            expectedRuntimeIdentity: expectedRuntimeIdentity)
    }

    func logprobs(prompt: [Int], forcedContinuation: [Int], config: RunConfig) async throws -> [[Float]] {
        let kind = try Self.cacheKind(config, allowSpec: false)
        return await engine.teacherForcedLogprobs(prompt: prompt, forced: forcedContinuation, kvCache: kind)
    }

    func logprobs(prompt: [Int], forcedContinuation: [Int], atPositions positions: [Int], config: RunConfig) async throws -> [[Float]] {
        let kind = try Self.cacheKind(config, allowSpec: false)
        return await engine.teacherForcedLogprobsAtPositions(
            prompt: prompt, forced: forcedContinuation, positions: positions, kvCache: kind)
    }

    func affineScoringTelemetry() async -> AffineKVCacheTelemetry? {
        await engine.affineScoringTelemetry()
    }

    func kvtunerScoringTelemetry(
        for selection: KVTunerRuntimeSelection
    ) async -> KVTunerKVCacheTelemetry? {
        await engine.kvtunerScoringTelemetry(for: selection)
    }

    func kvarnScoringTelemetry(
        for cell: KVarNKVRuntimeCell
    ) async -> KVarNKVCacheTelemetry? {
        await engine.kvarnScoringTelemetry(for: cell)
    }

    /// Validates the whole config and maps `kvQuant` through `KVCacheKind`'s closed affine /
    /// KVarN / TurboQuant allowlist. Anything else throws — a measurement must never silently
    /// run a different cache from the one requested.
    /// The scoring paths pass `allowSpec: false`: speculation changes how a decode loop steps,
    /// not what a teacher-forced forward scores, so a spec config there is a caller bug.
    private static func cacheKind(_ config: RunConfig, allowSpec: Bool = true) throws -> KVCacheKind {
        guard config.temperature == 0 else {
            throw SwiftEngineDriverError.unsupportedConfig("temperature=\(config.temperature) (greedy-only engine)")
        }
        if !allowSpec, let spec = config.specDecode {
            throw SwiftEngineDriverError.unsupportedConfig("specDecode=\(spec) on a scoring path (decode-only feature)")
        }
        if let selection = config.kvtunerSelection {
            guard config.kvQuant == selection.cellID else {
                throw SwiftEngineDriverError.unsupportedConfig(
                    "KVTuner selection cell \(selection.cellID) != kvQuant=\(config.kvQuant ?? "nil")")
            }
            return .kvtuner(selection)
        }
        if config.kvQuant?.hasPrefix("kvtuner-") == true {
            throw SwiftEngineDriverError.unsupportedConfig(
                "kvQuant=\(config.kvQuant!) requires an authenticated KVTuner selection")
        }
        guard let kind = KVCacheKind(kvQuant: config.kvQuant) else {
            throw SwiftEngineDriverError.unsupportedConfig(
                "kvQuant=\(config.kvQuant ?? "fp16") (unknown tier)")
        }
        return kind
    }

    /// Maps `RunConfig.specDecode` to the engine's spec-decode configuration. nil → plain decode;
    /// "pld" → prompt-lookup drafter with the config's ngram/K/compile-strategy knobs. Unknown
    /// drafters and every unmeasured spec + lossy-KV combination fail loudly.
    private static func specConfig(_ config: RunConfig) throws -> SpecDecodeConfig? {
        guard let spec = config.specDecode else { return nil }
        guard spec == "pld" else {
            throw SwiftEngineDriverError.unsupportedConfig("specDecode=\(spec) (known drafters: pld)")
        }
        if let kv = config.kvQuant, kv != "fp16" {
            // Structurally supported (truncate is on the CompiledCache protocol) but never
            // measured together — reject rather than silently "measure" an untested combo.
            throw SwiftEngineDriverError.unsupportedConfig("specDecode=pld with kvQuant=\(kv) (unmeasured combination; use fp16)")
        }
        return SpecDecodeConfig(
            drafter: PromptLookupDrafter(ngram: config.specNgram ?? 3),
            maxDraft: config.specMaxDraft ?? 8,
            compiledVerify: config.specCompiledVerify ?? false)
    }
}

/// Loads model + tokenizer from a local directory and builds the in-process driver.
/// The (Sendable) tokenizer is bound into its own local BEFORE `ctx.model` is sent into the
/// actor init — detaching it from the region that transfers — so CPU-side encode/decode stays
/// usable afterward (same region discipline as the spike's `loadActor`).
func captureKVTunerQualificationRuntimeSourceSnapshot(
    modelPath: String
) throws -> KVTunerCandidateRuntimeSourceSnapshot {
    let modelDirectory = URL(fileURLWithPath: modelPath)
    return try KVTunerCandidateRuntimeSourceSnapshot.load(
        exactModelConfigData: Data(
            contentsOf: modelDirectory.appendingPathComponent("config.json")),
        checkpointManifestHash: try ProvenanceCLI.checkpointManifestHash(
            at: modelPath),
        tokenizerSHA256: try ProvenanceCLI.tokenizerManifestSHA256(
            at: modelPath))
}

func loadSwiftDriver(
    modelPath: String,
    requireKVTunerQualificationIdentity: Bool = false
) async throws -> (
    driver: SwiftEngineDriver,
    tokenizer: MLXLMCommon.Tokenizer,
    eos: Int
) {
    // Bound MLX's buffer cache for the measurement process. The default cache limit tracks the
    // (raised) GPU memory limit, so unreusable transients can hoard tens of GB before anything
    // evicts — and the harness runs a Python reference process with its own allocator on the
    // same box. 8GB is far above any measurement path's steady-state reuse working set (decode
    // transients are MBs; a scoring chunk's logits are ~150MB) but keeps two co-resident
    // processes comfortably inside physical RAM. harness_reference.py sets the same bound.
    Memory.cacheLimit =
        KVTunerSensitivityCaptureEnvironment.requiredMemoryCacheLimitBytes
    let sourceIdentityBeforeLoad = requireKVTunerQualificationIdentity
        ? try captureKVTunerQualificationRuntimeSourceSnapshot(
            modelPath: modelPath)
        : nil
    let ctx = try await loadModel(
        from: URL(fileURLWithPath: modelPath),
        using: #huggingFaceTokenizerLoader()
    )
    let tokenizer = ctx.tokenizer
    let eos = tokenizer.eosToken.flatMap { tokenizer.convertTokenToId($0) } ?? -1
    let runtimeIdentity: KVTunerCandidateRuntimeIdentity?
    if let sourceIdentityBeforeLoad {
        let sourceIdentityAfterLoad = try
            captureKVTunerQualificationRuntimeSourceSnapshot(
                modelPath: modelPath)
        let stableSourceIdentity = try
            KVTunerCandidateRuntimeSourceSnapshot.validateUnchanged(
                before: sourceIdentityBeforeLoad,
                after: sourceIdentityAfterLoad)
        runtimeIdentity = try KVTunerCandidateRuntimeIdentity.load(
            sourceSnapshot: stableSourceIdentity,
            eosTokenID: eos)
    } else {
        runtimeIdentity = nil
    }
    let engine = HarnessEngineActor(
        model: ctx.model,
        kvtunerRuntimeIdentity: runtimeIdentity)
    return (SwiftEngineDriver(engine: engine, eos: eos), tokenizer, eos)
}
