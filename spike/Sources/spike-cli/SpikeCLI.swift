import Foundation
import MLX
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers
import SpikeCore

/// Minimal `--flag value` parser shared by all subcommands (no ArgumentParser dependency
/// pulled in for this throwaway CLI).
struct Flags {
    private var values: [String: String] = [:]

    init(_ arguments: [String]) {
        var i = 0
        while i < arguments.count {
            if arguments[i].hasPrefix("--"), i + 1 < arguments.count {
                values[String(arguments[i].dropFirst(2))] = arguments[i + 1]
                i += 2
            } else {
                i += 1
            }
        }
    }

    func string(_ key: String, default def: String) -> String { values[key] ?? def }
    func string(_ key: String) -> String? { values[key] }
    func int(_ key: String, default def: Int) -> Int { values[key].flatMap(Int.init) ?? def }
}

/// Engine selector: `baseline` = per-token lazy graph build (MLXDecoder),
/// `compiled` = MLX.compile'd decode step (CompiledMLXDecoder).
enum Engine: String {
    case baseline
    case compiled
}

/// Loads model + tokenizer and builds a ready-to-submit InferenceActor + eos id.
/// Bind the (Sendable) tokenizer into its own local before constructing the decoder:
/// the decoder captures ctx.model (a non-Sendable class ref) and is `sending` into the
/// actor init, which merges ctx's whole region into the actor's isolation. Detaching
/// `tokenizer` first keeps CPU-side encode/decode usable afterward without a race.
func loadActor(modelPath: String, engine: Engine) async throws -> (actor: InferenceActor, tokenizer: MLXLMCommon.Tokenizer, eos: Int) {
    let (model, tokenizer, eosId) = try await loadModelAndTokenizer(modelPath: modelPath)
    let actor = makeActor(model: model, engine: engine)
    return (actor, tokenizer, eosId)
}

func makeActor(model: sending any LanguageModel, engine: Engine) -> InferenceActor {
    switch engine {
    case .baseline:
        return InferenceActor(
            decoder: MLXDecoder(model: model, cache: model.newCache(parameters: nil)))
    case .compiled:
        return InferenceActor(decoder: CompiledMLXDecoder(model: model))
    }
}

/// Loads model + tokenizer once, without pinning a cache/actor to it — used by `bench`,
/// which must build a *fresh* actor (and thus a fresh, empty KVCache) per run. Reusing one
/// actor/cache across bench runs would let each run's prefill see the previous run's
/// already-generated tokens as false history (growing context across runs), which is a
/// bench-methodology bug, not just a style choice.
func loadModelAndTokenizer(modelPath: String) async throws -> (model: any LanguageModel, tokenizer: MLXLMCommon.Tokenizer, eos: Int) {
    let ctx = try await loadModel(
        from: URL(fileURLWithPath: modelPath),
        using: #huggingFaceTokenizerLoader()
    )
    let tokenizer = ctx.tokenizer
    let eosId = tokenizer.eosToken.flatMap { tokenizer.convertTokenToId($0) } ?? -1
    return (ctx.model, tokenizer, eosId)
}

func runPrompt(modelPath: String, prompt: String, maxTokens: Int, engine: Engine) async {
    do {
        let (actor, tokenizer, eosId) = try await loadActor(modelPath: modelPath, engine: engine)
        let promptTokens = tokenizer.encode(text: prompt)
        for try await id in await actor.submit(promptTokens: promptTokens, maxTokens: maxTokens, eos: eosId) {
            print(tokenizer.decode(tokenIds: [id], skipSpecialTokens: true), terminator: "")
        }
        print()
    } catch {
        print("run FAILED: \(error)")
        exit(1)
    }
}

/// Dumps the first-N greedy token ids as a JSON array, for diffing against
/// scripts/reference_tokens.py (Python mlx-lm, the equivalence reference).
/// With `rounds > 1`, repeats generation on the SAME actor via resetForNewRun()
/// between rounds — proving the reset + (for the compiled engine) in-place cache
/// reset + compiled-graph replay reproduce identical greedy tokens.
func equiv(modelPath: String, prompt: String, n: Int, engine: Engine, rounds: Int = 1) async {
    do {
        let (actor, tokenizer, eosId) = try await loadActor(modelPath: modelPath, engine: engine)
        let promptTokens = tokenizer.encode(text: prompt)
        for round in 0 ..< max(rounds, 1) {
            if round > 0 { await actor.resetForNewRun() }
            var ids: [Int] = []
            for try await id in await actor.submit(promptTokens: promptTokens, maxTokens: n, eos: eosId) {
                ids.append(id)
                if ids.count >= n { break }
            }
            let data = try JSONEncoder().encode(ids)
            print(String(data: data, encoding: .utf8)!)
        }
    } catch {
        print("equiv FAILED: \(error)")
        exit(1)
    }
}

func tokenizeDebug(modelPath: String, prompt: String) async {
    do {
        let ctx = try await loadModel(
            from: URL(fileURLWithPath: modelPath),
            using: #huggingFaceTokenizerLoader()
        )
        let ids = ctx.tokenizer.encode(text: prompt)
        print("swift ids: \(ids)")
    } catch {
        print("tokenize FAILED: \(error)")
        exit(1)
    }
}

func apiCheck(modelPath: String) async {
    do {
        let ctx = try await loadModel(
            from: URL(fileURLWithPath: modelPath),
            using: #huggingFaceTokenizerLoader()
        )
        let promptTokens = ctx.tokenizer.encode(text: "Hello")
        let ids = MLXArray(promptTokens).reshaped([1, promptTokens.count])
        let cache = ctx.model.newCache(parameters: nil)
        let logits = ctx.model(ids, cache: cache)
        eval(logits)
        print("logits.shape: \(logits.shape)")
        let last = logits[0..., -1, 0...]
        let next = argMax(last, axis: -1)
        let tokenId = next.item(Int.self)
        let decoded = ctx.tokenizer.decode(tokenIds: [tokenId], skipSpecialTokens: true)
        print("argMax token id: \(tokenId)")
        print("decoded token: \(decoded)")
    } catch {
        print("api-check FAILED: \(error)")
        exit(1)
    }
}

@main
struct SpikeCLI {
    static func main() async {
        let arguments = CommandLine.arguments
        guard arguments.count >= 2 else {
            print("spike ok: \(Spike.ok)")
            return
        }
        let sub = arguments[1]
        let flags = Flags(Array(arguments.dropFirst(2)))
        guard let engine = Engine(rawValue: flags.string("engine", default: "baseline")) else {
            print("unknown --engine (use: baseline | compiled)")
            exit(1)
        }

        switch sub {
        case "api-check":
            guard let modelPath = flags.string("model") else {
                print("usage: spike-cli api-check --model <PATH>")
                exit(1)
            }
            await apiCheck(modelPath: modelPath)

        case "run":
            guard let modelPath = flags.string("model") else {
                print("usage: spike-cli run --model <PATH> --prompt <TEXT> --max-tokens <N>")
                exit(1)
            }
            let prompt = flags.string("prompt", default: "Explain unified memory on Apple Silicon in one paragraph.")
            let maxTokens = flags.int("max-tokens", default: 128)
            await runPrompt(modelPath: modelPath, prompt: prompt, maxTokens: maxTokens, engine: engine)

        case "equiv":
            guard let modelPath = flags.string("model") else {
                print("usage: spike-cli equiv --model <PATH> --prompt <TEXT> --n <N>")
                exit(1)
            }
            let prompt = flags.string("prompt", default: "Write a haiku about unified memory.")
            let n = flags.int("n", default: 40)
            let rounds = flags.int("rounds", default: 1)
            await equiv(modelPath: modelPath, prompt: prompt, n: n, engine: engine, rounds: rounds)

        case "tokenize":
            guard let modelPath = flags.string("model") else {
                print("usage: spike-cli tokenize --model <PATH> --prompt <TEXT>")
                exit(1)
            }
            let prompt = flags.string("prompt", default: "Hello")
            await tokenizeDebug(modelPath: modelPath, prompt: prompt)

        case "phase-timing":
            guard let modelPath = flags.string("model") else {
                print("usage: spike-cli phase-timing --model <PATH> --prompt <TEXT> --steps <N>")
                exit(1)
            }
            let prompt = flags.string("prompt", default: "Explain how continuous batching improves LLM serving throughput.")
            let steps = flags.int("steps", default: 100)
            await phaseTimingDiagnostic(modelPath: modelPath, prompt: prompt, steps: steps, engine: engine)

        case "bench":
            guard let modelPath = flags.string("model") else {
                print("usage: spike-cli bench --model <PATH> --prompt <TEXT> --max-tokens <N> --runs <R>")
                exit(1)
            }
            let prompt = flags.string("prompt", default: "Explain how continuous batching improves LLM serving throughput.")
            let maxTokens = flags.int("max-tokens", default: 256)
            let runs = flags.int("runs", default: 3)
            await bench(modelPath: modelPath, prompt: prompt, maxTokens: maxTokens, runs: runs, engine: engine)

        default:
            print("spike ok: \(Spike.ok)")
        }
    }
}
