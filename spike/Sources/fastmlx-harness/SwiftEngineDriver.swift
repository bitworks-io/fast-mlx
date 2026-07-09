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
    let ctx = try await loadModel(
        from: URL(fileURLWithPath: modelPath),
        using: #huggingFaceTokenizerLoader()
    )
    let tokenizer = ctx.tokenizer
    let eos = tokenizer.eosToken.flatMap { tokenizer.convertTokenToId($0) } ?? -1
    let engine = HarnessEngineActor(model: ctx.model)
    return (SwiftEngineDriver(engine: engine, eos: eos), tokenizer, eos)
}
