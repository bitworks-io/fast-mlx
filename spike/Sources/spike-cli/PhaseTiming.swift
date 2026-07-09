import Foundation
import MLX
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers

/// Diagnostic-only decode loop, structurally identical to MLXDecoder's submit-first
/// lookahead, but timing three phases per step:
///   - `submit`: building + asyncEval-ing the next forward graph (should be ~free, lazy)
///   - `readback`: `.item(Int.self)` on the current token (the internal `eval()` call)
///   - `total`: wall time of one full step
/// Used once (Task 8 "INVESTIGATE" gate) to find where the ~20% gap vs Zig comes from,
/// as a lighter-weight alternative to an Instruments trace.
func phaseTimingDiagnostic(modelPath: String, prompt: String, steps: Int) async {
    do {
        let ctx = try await loadModel(
            from: URL(fileURLWithPath: modelPath),
            using: #huggingFaceTokenizerLoader()
        )
        let tokenizer = ctx.tokenizer
        let promptTokens = tokenizer.encode(text: prompt)
        let ids = MLXArray(promptTokens).reshaped([1, promptTokens.count])
        var cache = ctx.model.newCache(parameters: nil)

        var logits = ctx.model(ids, cache: cache)
        var last = logits[0..., -1, 0...]
        var next = argMax(last, axis: -1)
        var nextIds = next.reshaped([1, 1])
        var pendingLogits = ctx.model(nextIds, cache: cache)
        asyncEval(pendingLogits)
        _ = next.item(Int.self) // discard the prefill token, start timing from step 1

        var submitTimes: [Double] = []
        var readbackTimes: [Double] = []
        var totalTimes: [Double] = []

        for _ in 0..<steps {
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

        func stats(_ xs: [Double]) -> String {
            let n = Double(xs.count)
            let mean = xs.reduce(0, +) / n
            let sorted = xs.sorted()
            let p50 = sorted[sorted.count / 2]
            return "mean=\(String(format: "%.2f", mean * 1000))ms p50=\(String(format: "%.2f", p50 * 1000))ms"
        }
        print("steps=\(steps)")
        print("submit (graph-build + asyncEval, GPU dispatch):  \(stats(submitTimes))")
        print("readback (.item() blocking wait for `next`):     \(stats(readbackTimes))")
        print("total per step:                                  \(stats(totalTimes))")
        print("implied tok/s from total mean: \(String(format: "%.2f", 1.0 / (totalTimes.reduce(0, +) / Double(totalTimes.count))))")
        _ = cache // silence unused warning if optimized away
    } catch {
        print("phase-timing FAILED: \(error)")
        exit(1)
    }
}
