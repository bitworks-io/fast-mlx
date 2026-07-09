import Foundation
import MLX
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers
import SpikeCore

/// Diagnostic-only decode loop, structurally identical to the selected engine's
/// submit-first lookahead, timing three phases per step:
///   - `submit`: building/replaying + asyncEval-ing the next forward graph
///   - `readback`: `.item(Int.self)` on the current token (the internal `eval()` call)
///   - `total`: wall time of one full step
/// Used to localize where time goes (Task 8 INVESTIGATE; reused for the compiled-step
/// optimization pass), as a lighter-weight alternative to a full Instruments export.
func phaseTimingDiagnostic(modelPath: String, prompt: String, steps: Int, engine: Engine) async {
    do {
        let ctx = try await loadModel(
            from: URL(fileURLWithPath: modelPath),
            using: #huggingFaceTokenizerLoader()
        )
        let tokenizer = ctx.tokenizer
        let promptTokens = tokenizer.encode(text: prompt)

        var submitTimes: [Double] = []
        var readbackTimes: [Double] = []
        var totalTimes: [Double] = []

        switch engine {
        case .baseline:
            let ids = MLXArray(promptTokens).reshaped([1, promptTokens.count])
            let cache = ctx.model.newCache(parameters: nil)

            let logits = ctx.model(ids, cache: cache)
            let last = logits[0..., -1, 0...]
            var next = argMax(last, axis: -1)
            var nextIds = next.reshaped([1, 1])
            var pendingLogits = ctx.model(nextIds, cache: cache)
            asyncEval(pendingLogits)
            _ = next.item(Int.self) // discard the prefill token, start timing at step 1

            for _ in 0 ..< steps {
                let t0 = Date().timeIntervalSinceReferenceDate

                let l = pendingLogits[0..., -1, 0...]
                next = argMax(l, axis: -1)
                nextIds = next.reshaped([1, 1])
                let nextLogits = ctx.model(nextIds, cache: cache)
                asyncEval(nextLogits)
                let t1 = Date().timeIntervalSinceReferenceDate

                _ = next.item(Int.self)
                let t2 = Date().timeIntervalSinceReferenceDate

                pendingLogits = nextLogits
                submitTimes.append(t1 - t0)
                readbackTimes.append(t2 - t1)
                totalTimes.append(t2 - t0)
            }

        case .compiled:
            // Mirrors CompiledMLXDecoder: uncompiled prefill into CompiledKVCache,
            // then a compiled single-token step replayed per token.
            let layerCount = ctx.model.newCache(parameters: nil).count
            let chunk = 256
            let cap = ((promptTokens.count + steps + 64 + chunk - 1) / chunk) * chunk
            let caches = (0 ..< layerCount).map { _ in CompiledKVCache(capacity: cap) }

            let ids = MLXArray(promptTokens).reshaped([1, promptTokens.count])
            let logits = ctx.model(ids, cache: caches)
            let first = argMax(logits[0..., -1, 0...], axis: -1)

            let model = ctx.model
            let step = compile(inputs: caches, outputs: caches) { args in
                let y = args[0].reshaped([1, 1])
                let logits = model(y, cache: caches)
                return [argMax(logits[0..., -1, 0...], axis: -1)]
            }

            var pending = step([first])[0] // traces + compiles here (untimed warmup)
            asyncEval(pending)
            _ = first.item(Int.self)

            for _ in 0 ..< steps {
                let t0 = Date().timeIntervalSinceReferenceDate

                let following = step([pending])[0]
                asyncEval(following)
                let t1 = Date().timeIntervalSinceReferenceDate

                _ = pending.item(Int.self)
                let t2 = Date().timeIntervalSinceReferenceDate

                pending = following
                submitTimes.append(t1 - t0)
                readbackTimes.append(t2 - t1)
                totalTimes.append(t2 - t0)
            }
        }

        func stats(_ xs: [Double]) -> String {
            let n = Double(xs.count)
            let mean = xs.reduce(0, +) / n
            let sorted = xs.sorted()
            let p50 = sorted[sorted.count / 2]
            return "mean=\(String(format: "%.2f", mean * 1000))ms p50=\(String(format: "%.2f", p50 * 1000))ms"
        }
        print("engine=\(engine.rawValue) steps=\(steps)")
        print("submit (graph-build/replay + asyncEval dispatch):  \(stats(submitTimes))")
        print("readback (.item() blocking wait for `next`):       \(stats(readbackTimes))")
        print("total per step:                                    \(stats(totalTimes))")
        print("implied tok/s from total mean: \(String(format: "%.2f", 1.0 / (totalTimes.reduce(0, +) / Double(totalTimes.count))))")
    } catch {
        print("phase-timing FAILED: \(error)")
        exit(1)
    }
}
