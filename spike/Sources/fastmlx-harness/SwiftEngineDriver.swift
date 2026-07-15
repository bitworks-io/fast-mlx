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

/// Single-owner actor over the spike's compiled decode core. Owns BOTH the model and the
/// `CompiledMLXDecoder` (one isolation region) because `logprobs` needs direct forward access
/// to full-vocab logits, which the token-only `Decoder` protocol deliberately doesn't expose.
///
/// The generate path preserves the spike's compiled-step + no-sync-readback design by
/// construction: it only calls `CompiledMLXDecoder.prefill`/`.step`, where that pattern lives
/// (the next step's compiled forward is asyncEval'd before the current token's `.item()`).
actor HarnessEngineActor {
    private let model: any LanguageModel
    /// One decoder per KV-cache kind, built lazily and kept alive so each kind's compiled
    /// step function survives across runs (an fp16 baseline and a TurboQuant candidate can
    /// alternate within one verify invocation without retracing either).
    private var decoders: [KVCacheKind: CompiledMLXDecoder] = [:]
    /// Largest affine scoring allocation observed in this actor. Scoring caches are ephemeral,
    /// so their scalar geometry/byte evidence must be retained before the arrays are released.
    private var maximumAffineScoringTelemetry: AffineKVCacheTelemetry?

    init(model: sending any LanguageModel) {
        self.model = model
    }

    /// Greedy decode via the compiled core. The returned ids INCLUDE a terminal eos if one is
    /// produced (mirroring scripts/harness_reference.py exactly, so token streams diff cleanly).
    /// Quantized-cache engagement is read AFTER timing so its synchronization cannot skew the
    /// benchmark. Affine returns a Sendable scalar snapshot; TurboQuant retains its legacy token
    /// marker until that format's evidence schema is generalized.
    func generate(prompt: [Int], maxTokens: Int, eos: Int, kvCache kind: KVCacheKind)
        -> (
            tokens: [Int], submitTime: Double, tokenTimes: [Double],
            turboQuantTokens: Int?, affineTelemetry: AffineKVCacheTelemetry?
        )
    {
        if decoders[kind] == nil {
            decoders[kind] = CompiledMLXDecoder(model: model, kvCache: kind)
        }
        var decoder = decoders[kind]!
        defer { decoders[kind] = decoder }
        decoder.reset() // in-place KV reset: compiled graph stays valid across runs
        let submitTime = Date().timeIntervalSinceReferenceDate
        guard maxTokens > 0 else { return ([], submitTime, [], nil, nil) }
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
        return (tokens, submitTime, tokenTimes, turboQuantTokens, affineTelemetry)
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
    /// here, not through the compiled decode path. fp16 keeps the stock model cache; affine
    /// and TurboQuant tiers get their requested concrete cache per layer, sized for the whole
    /// pass up front (scoring knows its total length; no chunked growth needed).
    private func makeScoringCache(kind: KVCacheKind, capacity: Int) -> [any KVCache] {
        switch kind {
        case .fp16:
            return model.newCache(parameters: nil)
        case .affine, .turboQuant, .kvarn:
            let layerCount = model.newCache(parameters: nil).count
            return (0 ..< layerCount).map { _ in kind.makeCache(capacity: max(capacity, 1)) }
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
        case .turboQuant:
            guard let tq = cache.first as? TurboQuantKVCache, tq.offset >= minTokens else {
                preconditionFailure("TurboQuant tier requested but the quantized scoring cache did not engage")
            }
        case .kvarn:
            guard let kvarn = cache.first as? KVarNKVCache, kvarn.offset >= minTokens else {
                preconditionFailure("KVarN tier requested but the KVarN scoring cache did not engage")
            }
        }
    }

    func affineScoringTelemetry() -> AffineKVCacheTelemetry? {
        maximumAffineScoringTelemetry
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

    /// Validates the whole config and maps `kvQuant` through `KVCacheKind`'s closed affine /
    /// TurboQuant allowlist. Anything else throws — a measurement must never silently run a
    /// different cache from the one requested.
    /// The scoring paths pass `allowSpec: false`: speculation changes how a decode loop steps,
    /// not what a teacher-forced forward scores, so a spec config there is a caller bug.
    private static func cacheKind(_ config: RunConfig, allowSpec: Bool = true) throws -> KVCacheKind {
        guard config.temperature == 0 else {
            throw SwiftEngineDriverError.unsupportedConfig("temperature=\(config.temperature) (greedy-only engine)")
        }
        if !allowSpec, let spec = config.specDecode {
            throw SwiftEngineDriverError.unsupportedConfig("specDecode=\(spec) on a scoring path (decode-only feature)")
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
func loadSwiftDriver(modelPath: String) async throws -> (driver: SwiftEngineDriver, tokenizer: MLXLMCommon.Tokenizer, eos: Int) {
    // Bound MLX's buffer cache for the measurement process. The default cache limit tracks the
    // (raised) GPU memory limit, so unreusable transients can hoard tens of GB before anything
    // evicts — and the harness runs a Python reference process with its own allocator on the
    // same box. 8GB is far above any measurement path's steady-state reuse working set (decode
    // transients are MBs; a scoring chunk's logits are ~150MB) but keeps two co-resident
    // processes comfortably inside physical RAM. harness_reference.py sets the same bound.
    Memory.cacheLimit = 8 << 30
    let ctx = try await loadModel(
        from: URL(fileURLWithPath: modelPath),
        using: #huggingFaceTokenizerLoader()
    )
    let tokenizer = ctx.tokenizer
    let eos = tokenizer.eosToken.flatMap { tokenizer.convertTokenToId($0) } ?? -1
    let engine = HarnessEngineActor(model: ctx.model)
    return (SwiftEngineDriver(engine: engine, eos: eos), tokenizer, eos)
}
