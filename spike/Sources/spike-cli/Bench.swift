import Foundation
import MLXLMCommon
import SpikeCore

/// Stream-timed, warmup-dropped decode bench (carry-forward methodology): run 0 is a
/// warmup and its rate is dropped; each run's prompt is salted with a unique nonce so
/// the KV cache / tokenizer can't trivially reuse a cached run; rate is computed from the
/// live token stream timestamps (DecodeMetrics), not from any self-reported usage field.
func bench(modelPath: String, prompt: String, maxTokens: Int, runs: Int, engine: Engine) async {
    #if DEBUG
    print("bench FAILED: Debug build — perf numbers are meaningless. Build with -configuration Release.")
    exit(1)
    #else
    do {
        // Load once (expensive: reads weights from disk) and build ONE actor. Each run
        // below calls actor.resetForNewRun() instead of constructing a new actor, so a
        // fresh (non-Sendable) model reference never has to cross the actor boundary
        // twice — see InferenceActor.resetForNewRun() / MLXDecoder.reset(). This also
        // means runs don't see each other's tokens as false KV-cache history.
        let (model, tokenizer, eosId) = try await loadModelAndTokenizer(modelPath: modelPath)
        let actor = makeActor(model: model, engine: engine)
        let nonce = Int.random(in: 0..<1_000_000)

        var decodeRates: [Double] = []
        var ttfts: [Double] = []

        for i in 0...runs { // run 0 = warmup, dropped from the average
            if i > 0 { await actor.resetForNewRun() }
            let salted = "[run-\(i)-\(nonce)] \(prompt)"
            let promptTokens = tokenizer.encode(text: salted)

            let submitTime = Date().timeIntervalSinceReferenceDate
            var tokenTimes: [Double] = []
            for try await _ in await actor.submit(promptTokens: promptTokens, maxTokens: maxTokens, eos: eosId) {
                tokenTimes.append(Date().timeIntervalSinceReferenceDate)
            }
            guard !tokenTimes.isEmpty else {
                print("bench FAILED: run \(i) produced zero tokens")
                exit(1)
            }
            let metrics = DecodeMetrics(submitTime: submitTime, tokenTimes: tokenTimes)

            if i == 0 {
                print("# warmup run dropped: \(metrics.generatedTokenCount) tokens, ttft=\(String(format: "%.3f", metrics.ttftSeconds))s")
                continue
            }
            if let rate = metrics.decodeTokensPerSecond {
                decodeRates.append(rate)
            }
            ttfts.append(metrics.ttftSeconds)
            print("# run \(i): \(metrics.generatedTokenCount) tokens, ttft=\(String(format: "%.3f", metrics.ttftSeconds))s, decode_tok_s=\(String(format: "%.2f", metrics.decodeTokensPerSecond ?? .nan))")
        }

        guard !decodeRates.isEmpty else {
            print("bench FAILED: no measurable decode rate across \(runs) runs")
            exit(1)
        }
        let avgRate = decodeRates.reduce(0, +) / Double(decodeRates.count)
        let avgTtft = ttfts.reduce(0, +) / Double(ttfts.count)

        print("model,engine,max_tokens,runs,decode_tok_s_avg,ttft_ms_avg")
        let modelName = URL(fileURLWithPath: modelPath).lastPathComponent
        print("\(modelName),\(engine.rawValue),\(maxTokens),\(runs),\(String(format: "%.2f", avgRate)),\(String(format: "%.1f", avgTtft * 1000))")
    } catch {
        print("bench FAILED: \(error)")
        exit(1)
    }
    #endif
}
