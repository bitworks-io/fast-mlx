import Darwin
import Foundation
import HarnessCore
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import SpikeCore
import Tokenizers

// MARK: - ctxprobe: long-context memory diagnostic (root-causing the ~7K SIGKILL ceiling)
//
// Reproduces the harness's three long-context code paths in isolation, with MLX memory
// snapshots (active/cache/peak) + the process physical footprint printed as the context
// grows, and a self-abort threshold so a probe run degrades into a diagnosis instead of
// taking the whole box down with a jetsam kill.
//
// modes:
//   steps     stock-cache single-token teacher-forced loop (the ORIGINAL kl long-context path,
//             kept as the regression reproducer for the O(context²) cache blowup)
//   score     chunked teacher-forced scoring (the FIXED kl long-context path)
//   prefill   one-shot model() prefill of the full prompt (stock cache)
//   generate  CompiledMLXDecoder prefill + a few decode steps (the serving path)

/// Process physical footprint (the number jetsam actually kills on), in bytes.
func physFootprintBytes() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) { ptr in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return 0 }
    return info.phys_footprint
}

func gb(_ bytes: Int) -> String { String(format: "%.2f", Double(bytes) / 1_073_741_824) }
func gb(_ bytes: UInt64) -> String { gb(Int(bytes)) }

func memReport(_ tag: String) {
    let s = Memory.snapshot()
    print(
        "[mem] \(tag): active=\(gb(s.activeMemory))GB cache=\(gb(s.cacheMemory))GB "
            + "peak=\(gb(s.peakMemory))GB footprint=\(gb(physFootprintBytes()))GB")
    fflush(stdout)
}

actor CtxProbeActor {
    private let model: any LanguageModel

    init(model: sending any LanguageModel) {
        self.model = model
    }

    /// The kl long-context path: 1-token prompt, then single-token teacher-forced steps on a
    /// stock cache, materializing a full-vocab row every `sampleEvery` steps.
    func runSteps(tokens: [Int], sampleEvery: Int, reportEvery: Int, abortBytes: UInt64) {
        let cache = model.newCache(parameters: nil)
        var y = MLXArray([tokens[0]]).reshaped([1, 1])
        memReport("steps: start")
        for (i, tok) in tokens.dropFirst().enumerated() {
            let logits = model(y, cache: cache)
            if i % sampleEvery == 0 {
                _ = logits[0..., -1, 0...].asType(.float32).asArray(Float.self)
            }
            y = MLXArray([tok]).reshaped([1, 1])
            if (i + 1) % reportEvery == 0 {
                memReport("steps: pos \(i + 1)/\(tokens.count - 1)")
                if physFootprintBytes() > abortBytes {
                    print("[abort] footprint exceeded threshold at position \(i + 1) — diagnosis, not a crash")
                    return
                }
            }
        }
        memReport("steps: done (\(tokens.count - 1) positions)")
    }

    /// The FIXED measurement path: chunked teacher-forced scoring (same plan/loop shape as
    /// `HarnessEngineActor.scoreForced`) of an N-token sequence against itself, sampling
    /// full-vocab rows every `sampleEvery` positions.
    func runChunkedScore(tokens: [Int], sampleEvery: Int, abortBytes: UInt64) {
        let cache = model.newCache(parameters: nil)
        let prompt = [tokens[0]]
        let forced = Array(tokens.dropFirst())
        let wanted = stride(from: 0, to: forced.count, by: sampleEvery).map { $0 }
        let plan = forcedScoringPlan(
            promptCount: prompt.count, forcedCount: forced.count,
            wantedPositions: wanted, chunkSize: 512)
        let input = prompt + forced.dropLast()
        memReport("score: start (n=\(tokens.count), \(plan.chunks.count) chunks, \(wanted.count) sampled rows)")
        var rows = 0
        for (i, chunk) in plan.chunks.enumerated() {
            let ids = MLXArray(Array(input[chunk.inputRange])).reshaped([1, chunk.inputRange.count])
            let logits = model(ids, cache: cache)
            eval(logits)
            for sel in chunk.rows {
                _ = logits[0..., sel.localIndex, 0...].asType(.float32).asArray(Float.self)
                rows += 1
            }
            if (i + 1) % 8 == 0 || i == plan.chunks.count - 1 {
                memReport("score: chunk \(i + 1)/\(plan.chunks.count) (rows \(rows))")
                if physFootprintBytes() > abortBytes {
                    print("[abort] footprint exceeded threshold at chunk \(i + 1)")
                    return
                }
            }
        }
        memReport("score: done (\(rows) rows)")
    }

    /// One-shot prefill of the whole prompt through the plain forward (stock cache).
    func runPrefill(tokens: [Int]) {
        let cache = model.newCache(parameters: nil)
        memReport("prefill: start (n=\(tokens.count))")
        let y = MLXArray(tokens).reshaped([1, tokens.count])
        let logits = model(y, cache: cache)
        let row = logits[0..., -1, 0...].asType(.float32).asArray(Float.self)
        memReport("prefill: done (last row argmax=\(row.indices.max(by: { row[$0] < row[$1] }) ?? -1))")
    }

    /// The serving path: CompiledMLXDecoder prefill + a few compiled decode steps.
    func runGenerate(tokens: [Int], decodeSteps: Int) {
        var decoder = CompiledMLXDecoder(model: model)
        memReport("generate: start (prompt=\(tokens.count))")
        var tok = decoder.prefill(tokens)
        memReport("generate: after prefill")
        var out = [tok]
        for _ in 0..<decodeSteps {
            tok = decoder.step(last: tok)
            out.append(tok)
        }
        memReport("generate: after \(decodeSteps) decode steps")
        print("[generate] first tokens: \(out)")
    }
}

func runCtxProbe(_ flags: Flags) async {
    guard let modelPath = flags.string("model") else {
        print("usage: fastmlx-harness ctxprobe --model <PATH> --tokens <N> --mode <steps|score|prefill|generate> [--sample-every 50] [--report-every 500] [--abort-gb 60] [--cache-limit-gb N]")
        exit(2)
    }
    let n = flags.int("tokens", default: 7200)
    let mode = flags.string("mode", default: "steps")
    let sampleEvery = flags.int("sample-every", default: 50)
    let reportEvery = flags.int("report-every", default: 500)
    let abortBytes = UInt64(flags.int("abort-gb", default: 60)) * 1_073_741_824
    if let cacheGB = flags.string("cache-limit-gb").flatMap(Int.init) {
        Memory.cacheLimit = cacheGB * 1_073_741_824
        print("[probe] MLX cache limit set to \(cacheGB)GB")
    }
    do {
        let ctx = try await loadModel(
            from: URL(fileURLWithPath: modelPath),
            using: #huggingFaceTokenizerLoader()
        )
        let tokenizer = ctx.tokenizer
        // Natural text as the token source (tiled to length): a probe should walk through
        // realistic token distributions, not a single repeated id.
        let corpus = try loadMeasurementCorpus(flags)
        guard let long = corpus.entries(tagged: .longContext).first else {
            print("ctxprobe FAILED: corpus has no long-context entry to draw tokens from")
            exit(1)
        }
        let base = tokenizer.encode(text: long.text)
        let tokens = (0..<n).map { base[$0 % base.count] }
        print("[probe] model=\(modelPath) mode=\(mode) tokens=\(n) sampleEvery=\(sampleEvery)")
        let probe = CtxProbeActor(model: ctx.model)
        memReport("model loaded")
        switch mode {
        case "steps":
            await probe.runSteps(
                tokens: tokens, sampleEvery: sampleEvery, reportEvery: reportEvery,
                abortBytes: abortBytes)
        case "score":
            await probe.runChunkedScore(tokens: tokens, sampleEvery: sampleEvery, abortBytes: abortBytes)
        case "prefill":
            await probe.runPrefill(tokens: tokens)
        case "generate":
            await probe.runGenerate(tokens: tokens, decodeSteps: flags.int("decode-steps", default: 8))
        default:
            print("ctxprobe FAILED: unknown mode \(mode)")
            exit(2)
        }
        memReport("probe exit")
    } catch {
        print("ctxprobe FAILED: \(error)")
        exit(1)
    }
}
