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
    private var decoder: CompiledMLXDecoder

    init(model: sending any LanguageModel) {
        self.model = model
        self.decoder = CompiledMLXDecoder(model: model)
    }

    /// Greedy decode via the compiled core. The returned ids INCLUDE a terminal eos if one is
    /// produced (mirroring scripts/harness_reference.py exactly, so token streams diff cleanly).
    func generate(prompt: [Int], maxTokens: Int, eos: Int) -> (tokens: [Int], submitTime: Double, tokenTimes: [Double]) {
        decoder.reset() // in-place KV reset: compiled graph stays valid across runs
        let submitTime = Date().timeIntervalSinceReferenceDate
        guard maxTokens > 0 else { return ([], submitTime, []) }
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
        return (tokens, submitTime, tokenTimes)
    }

    /// Full-vocab RAW LOGITS per generated position at temp=0 — the `EngineDriver.logprobs`
    /// contract: index == token id, length == vocab, NOT top-k, NOT softmaxed. Runs the plain
    /// (uncompiled) forward on a fresh cache: this is the measurement path, not the perf path,
    /// and the per-position full-vocab readback is inherently synchronous anyway.
    /// fp16 -> float32 conversion is exact, so argmax over a returned row reproduces the
    /// greedy token chosen at that position.
    func logprobs(prompt: [Int], maxTokens: Int, eos: Int) -> [[Float]] {
        let cache = model.newCache(parameters: nil)
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
        return rows
    }

    /// TEACHER-FORCED `logprobs`: row i is the next-token distribution given
    /// context = prompt + forced[0..<i]; forced[i] is fed as the next input regardless of
    /// argmax, and eos does NOT stop the loop (the forced continuation already encodes where
    /// its producer stopped). Exactly forced.count rows. Same measurement path as the
    /// free-running variant (plain forward, fresh cache) — the perf path's compiled-step +
    /// no-sync-readback design lives untouched in `generate`.
    func teacherForcedLogprobs(prompt: [Int], forced: [Int]) -> [[Float]] {
        scoreForced(prompt: prompt, forced: forced, wanted: nil)
    }

    /// Sampled variant: same chunked forward over the full forced continuation (causal decoding
    /// requires every intermediate token as context regardless), but only converts+keeps a
    /// full-vocab row at `positions` — a long-context entry can be thousands of positions, and
    /// materializing every row would be ~0.6MB/row x thousands x 2 drivers. `positions` must be
    /// ascending (evenlySpacedPositions's contract); rows are returned in that order.
    func teacherForcedLogprobsAtPositions(prompt: [Int], forced: [Int], positions: [Int]) -> [[Float]] {
        scoreForced(prompt: prompt, forced: forced, wanted: positions)
    }

    /// CHUNKED teacher-forced scoring (`forcedScoringPlan` in HarnessCore holds the pure
    /// bookkeeping): multi-token chunk forwards instead of one forward per forced token.
    ///
    /// WHY (the ~7K-context SIGKILL root cause): single-token stepping makes every step's
    /// transient buffers slightly LARGER than the last step's (the stock cache returns growing
    /// K/V slices), so MLX's buffer cache can never reuse a freed buffer and grows as
    /// O(context²) — measured 43GB of dead cache (active flat at 17GB) by position 6750 on
    /// Qwen3-32B-4bit, which, with the Python reference process ballooning identically, is
    /// exactly the ~6.7–7.1K jetsam SIGKILL ceiling the harness hit. Chunk forwards keep
    /// transients same-shaped chunk to chunk (cache reuse works, memory stays flat) and score
    /// at prefill speed instead of decode speed. The per-chunk `eval` bounds the lazy graph so
    /// pending work cannot pile up across chunks.
    private func scoreForced(prompt: [Int], forced: [Int], wanted: [Int]?) -> [[Float]] {
        let cache = model.newCache(parameters: nil)
        let input = prompt + forced.dropLast()
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
        return rows
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
        try Self.requireSupported(config)
        let out = await engine.generate(prompt: prompt, maxTokens: config.maxTokens, eos: eos)
        return RunResult(
            tokens: out.tokens,
            engagement: .init(["decode": out.tokens.count]),
            acceptanceRate: nil, // no spec-decode path yet
            submitTime: out.submitTime,
            tokenTimes: out.tokenTimes)
    }

    func logprobs(prompt: [Int], config: RunConfig) async throws -> [[Float]] {
        try Self.requireSupported(config)
        return await engine.logprobs(prompt: prompt, maxTokens: config.maxTokens, eos: eos)
    }

    func logprobs(prompt: [Int], forcedContinuation: [Int], config: RunConfig) async throws -> [[Float]] {
        try Self.requireSupported(config)
        return await engine.teacherForcedLogprobs(prompt: prompt, forced: forcedContinuation)
    }

    func logprobs(prompt: [Int], forcedContinuation: [Int], atPositions positions: [Int], config: RunConfig) async throws -> [[Float]] {
        try Self.requireSupported(config)
        return await engine.teacherForcedLogprobsAtPositions(prompt: prompt, forced: forcedContinuation, positions: positions)
    }

    private static func requireSupported(_ config: RunConfig) throws {
        guard config.temperature == 0 else {
            throw SwiftEngineDriverError.unsupportedConfig("temperature=\(config.temperature) (greedy-only engine)")
        }
        if let spec = config.specDecode {
            throw SwiftEngineDriverError.unsupportedConfig("specDecode=\(spec) (not implemented)")
        }
        if let kv = config.kvQuant, kv != "fp16" {
            throw SwiftEngineDriverError.unsupportedConfig("kvQuant=\(kv) (only fp16 KV cache exists)")
        }
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
