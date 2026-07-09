import Foundation
import HarnessCore
import MLXLMCommon
import SpikeCore

/// Minimal `--flag value` parser (same shape as the spike's; no ArgumentParser dependency).
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

/// Known-good prompts from the spike's equivalence work (see 2026-07-08-swift-spike-verdict.md):
/// "The capital of France is" matched 60/60 on the MoE target model.
let knownGoodPrompt = "The capital of France is"
let benchPrompt = "Explain how continuous batching improves LLM serving throughput."
let klPrompts = [
    knownGoodPrompt,
    "Write a haiku about unified memory.",
    benchPrompt,
]

func referenceDriver(_ flags: Flags, modelPath: String, eos: Int) -> ReferenceDriver {
    ReferenceDriver(
        pythonPath: (flags.string("python", default: "~/harness-venv/bin/python") as NSString).expandingTildeInPath,
        scriptPath: flags.string("script", default: "scripts/harness_reference.py"),
        modelPath: flags.string("reference-model", default: modelPath),
        eos: eos)
}

func fmt(_ x: Double, _ digits: Int = 3) -> String { String(format: "%.\(digits)f", x) }

// MARK: - corpus (hermetic; no model)

func runCorpus() {
    var failures = 0
    for entry in HarnessCorpus.entries {
        let out = HarnessCorpus.process(entry.raw)
        var problems: [String] = []
        if out.visibleText.contains("<|") { problems.append("control-tag leak") }
        if let argsJSON = out.toolArgsJSON,
           (try? JSONSerialization.jsonObject(with: Data(argsJSON.utf8))) == nil {
            problems.append("tool args not valid JSON")
        }
        if let expected = entry.expectedVisible, out.visibleText != expected {
            problems.append("visible text mismatch: got \(String(reflecting: out.visibleText))")
        }
        if let expectedTool = entry.expectedTool, out.toolArgsJSON != expectedTool {
            problems.append("tool args mismatch: got \(String(reflecting: out.toolArgsJSON ?? "nil"))")
        }
        if problems.isEmpty {
            print("PASS \(entry.name)")
        } else {
            print("FAIL \(entry.name): \(problems.joined(separator: "; "))")
            failures += 1
        }
    }
    if failures == 0 {
        print("corpus: PASS (\(HarnessCorpus.entries.count) entries, universal invariants hold)")
    } else {
        print("corpus: FAIL (\(failures)/\(HarnessCorpus.entries.count) entries)")
        exit(1)
    }
}

// MARK: - verify (the triad)

func runVerify(_ flags: Flags) async {
    guard let modelPath = flags.string("model") else {
        print("usage: fastmlx-harness verify --model <PATH> [--prompt <TEXT>] [--n 60] [--min-prefix 30] [--python <PY>] [--script <REF.py>] [--reference-model <PATH>]")
        exit(2)
    }
    let prompt = flags.string("prompt", default: knownGoodPrompt)
    let n = flags.int("n", default: 60)
    let minPrefix = flags.int("min-prefix", default: 30)
    do {
        let (driver, tokenizer, eos) = try await loadSwiftDriver(modelPath: modelPath)
        let promptTokens = tokenizer.encode(text: prompt)
        let config = RunConfig.greedy(maxTokens: n)

        let candidate = try await driver.generate(prompt: promptTokens, config: config)
        let reference = referenceDriver(flags, modelPath: modelPath, eos: eos)
        let refRun = try await reference.generate(prompt: promptTokens, config: config)

        let eq = EquivalenceCheck(minPrefix: minPrefix)
            .evaluate(candidate: candidate.tokens, reference: refRun.tokens)
        let decodeCount = candidate.engagement.counts["decode"] ?? 0
        let engaged = EngagementCheck(marker: "decode", floor: 1).passed(before: 0, after: decodeCount)
        // Acceptance applies to spec-decode runs; none is configured -> not applicable (nil).
        let verdict = TriadVerdict(equivalenceOK: eq.passed, engaged: engaged, acceptanceOK: nil)

        let comparable = min(candidate.tokens.count, refRun.tokens.count)
        print("prompt: \(String(reflecting: prompt)) (\(promptTokens.count) tokens), n=\(n), temp=0")
        print("equivalence: identical-prefix \(eq.prefix)/\(comparable) (gate >= \(minPrefix)) -> \(eq.passed ? "PASS" : "FAIL")")
        print("engagement:  decode counter 0 -> \(decodeCount) (floor 1) -> \(engaged ? "PASS" : "FAIL")")
        print("acceptance:  n/a (no spec-decode configured)")
        print("triad: \(verdict.passed ? "PASS" : "FAIL")")
        if !verdict.passed {
            print("candidate: \(candidate.tokens)")
            print("reference: \(refRun.tokens)")
            exit(1)
        }
    } catch {
        print("verify FAILED: \(error)")
        exit(1)
    }
}

// MARK: - bench (cell matrix -> CSV)

func runBench(_ flags: Flags) async {
    guard let modelPath = flags.string("model") else {
        print("usage: fastmlx-harness bench --model <PATH> [--prompt <TEXT>] [--max-tokens 256] [--runs 3] [--label <L>] [--csv <FILE>]")
        exit(2)
    }
    do {
        try assertReleaseBuild()
        let prompt = flags.string("prompt", default: benchPrompt)
        let maxTokens = flags.int("max-tokens", default: 256)
        let runs = flags.int("runs", default: 3)
        let label = flags.string("label", default: "harness")
        let (driver, tokenizer, _) = try await loadSwiftDriver(modelPath: modelPath)

        let modelName = URL(fileURLWithPath: modelPath).lastPathComponent
        let lower = modelName.lowercased()
        let quant = lower.contains("8bit") ? "int8" : lower.contains("4bit") ? "int4" : "fp16"
        let cell = Cell(workload: .decode, mode: .none, model: modelName, quant: quant, concurrency: 1)
        let nonce = String(Int.random(in: 0..<1_000_000))

        var ttfts: [Double] = []
        let agg = try await BenchRunner().run(cell: cell, iterations: runs + 1, nonce: nonce, basePrompt: prompt) { i, salted in
            let promptTokens = tokenizer.encode(text: salted)
            let result = try await driver.generate(prompt: promptTokens, config: .greedy(maxTokens: maxTokens))
            guard !result.tokenTimes.isEmpty else {
                print("# run \(i): produced zero tokens -> skipped")
                return nil
            }
            let metrics = DecodeMetrics(submitTime: result.submitTime, tokenTimes: result.tokenTimes)
            let tag = i == 0 ? "warmup (dropped)" : "run \(i)"
            print("# \(tag): \(metrics.generatedTokenCount) tokens, ttft=\(fmt(metrics.ttftSeconds))s, decode_tok_s=\(fmt(metrics.decodeTokensPerSecond ?? .nan, 2))")
            if i > 0 { ttfts.append(metrics.ttftSeconds) }
            return metrics.decodeTokensPerSecond
        }
        guard agg.runs > 0 else {
            print("bench FAILED: no measurable post-warmup runs")
            exit(1)
        }
        let avgTtftMs = ttfts.isEmpty ? 0 : ttfts.reduce(0, +) / Double(ttfts.count) * 1000
        let row = BenchRow(
            label: label, workload: .decode, mode: .none, model: modelName,
            decodeTokS: (agg.mean * 100).rounded() / 100, ttftMs: (avgTtftMs * 10).rounded() / 10,
            quant: quant, concurrency: 1)
        print(BenchRow.csvHeader)
        print(row.csvLine)
        if let csvPath = flags.string("csv") {
            let url = URL(fileURLWithPath: csvPath)
            let existing = (try? String(contentsOf: url, encoding: String.Encoding.utf8)) ?? ""
            var content: String = existing.isEmpty ? BenchRow.csvHeader + "\n" : existing
            content += row.csvLine + "\n"
            try content.write(to: url, atomically: true, encoding: String.Encoding.utf8)
            print("# appended to \(csvPath)")
        }
    } catch BenchGuardError.debugBuild {
        print("bench FAILED: Debug build — perf numbers would be meaningless. Build with -configuration Release.")
        exit(1)
    } catch {
        print("bench FAILED: \(error)")
        exit(1)
    }
}

// MARK: - kl (KLDivergenceMetric: candidate vs reference)

func runKL(_ flags: Flags) async {
    guard let modelPath = flags.string("model") else {
        print("usage: fastmlx-harness kl --model <PATH> [--reference-model <PATH>] [--positions 24] [--python <PY>] [--script <REF.py>]")
        exit(2)
    }
    do {
        let positions = flags.int("positions", default: 24)
        let (driver, tokenizer, eos) = try await loadSwiftDriver(modelPath: modelPath)
        let reference = referenceDriver(flags, modelPath: modelPath, eos: eos)
        let sameWeights = reference.modelPath == modelPath
        let config = RunConfig.greedy(maxTokens: positions)
        let prompts = klPrompts.map { tokenizer.encode(text: $0) }

        print("candidate: Swift engine on \(modelPath)")
        print("reference: mlx-lm on \(reference.modelPath)")
        print(sameWeights
            ? "# SAME weights both sides -> this is a PIPELINE PROOF (cross-implementation float-reduction noise), not a quantization-loss measurement."
            : "# DIFFERENT weights -> candidate-vs-reference quantization/behavior divergence measurement.")

        var allKLs: [Double] = []
        var alignedKLs: [Double] = []
        var spotChecked = false
        for (text, prompt) in zip(klPrompts, prompts) {
            let c = try await driver.logprobs(prompt: prompt, config: config)
            let r = try await reference.logprobs(prompt: prompt, config: config)
            guard let c0 = c.first, let r0 = r.first else {
                print("kl FAILED: empty logprobs for prompt \(String(reflecting: text))")
                exit(1)
            }
            guard c0.count == r0.count else {
                print("kl FAILED: vocab mismatch (candidate \(c0.count) vs reference \(r0.count)) — index-aligned KL would be meaningless")
                exit(1)
            }
            if !spotChecked {
                spotCheckOrdering(candidateRow: c0, referenceRow: r0, eos: eos, sameWeights: sameWeights)
                spotChecked = true
            }
            // Greedy tokens are recoverable from the rows themselves (argmax; fp16->fp32 is exact).
            let candTokens = c.map(argmaxIndex)
            let refTokens = r.map(argmaxIndex)
            let prefix = identicalPrefix(candTokens, refTokens)
            let kls = perPositionKLs(reference: r, candidate: c)
            // Positions 0...prefix have IDENTICAL contexts on both sides (the token at `prefix`
            // differs, but its producing context is the shared prefix). Beyond that, each side's
            // own greedy path diverges and per-position KL compares different contexts.
            let alignedCount = min(prefix + 1, kls.count)
            allKLs.append(contentsOf: kls)
            alignedKLs.append(contentsOf: kls.prefix(alignedCount))
            print("prompt \(String(reflecting: text)): positions=\(kls.count), identical-prefix=\(prefix)/\(min(candTokens.count, refTokens.count)), median KL=\(sci(medianOf(kls)))")
        }

        // Headline = KLDivergenceMetric's exact computation (perPositionKLs + medianOf are the
        // metric's own internals, so this equals KLDivergenceMetric.measure on the same rows).
        allKLs.sort()
        alignedKLs.sort()
        let p95 = allKLs[min(Int(Double(allKLs.count - 1) * 0.95), allKLs.count - 1)]
        print("kl_median (KLDivergenceMetric, all \(allKLs.count) positions): \(sci(medianOf(allKLs))) nats")
        print("kl_p95    (all positions): \(sci(p95)) nats")
        print("kl_median (context-aligned positions only, \(alignedKLs.count)): \(sci(medianOf(alignedKLs))) nats")
    } catch {
        print("kl FAILED: \(error)")
        exit(1)
    }
}

func argmaxIndex(_ row: [Float]) -> Int {
    var best = 0
    for i in row.indices where row[i] > row[best] { best = i }
    return best
}

func sci(_ x: Double) -> String { String(format: "%.3e", x) }

/// The logprobs ORDERING contract check: both sides must be full-vocab, token-id-ordered raw
/// logits. Same vocab length + equal argmax id + close raw values at sampled token ids on a
/// shared-context position is strong evidence the index<->token-id mapping matches.
/// Only meaningful when both sides run the SAME weights.
func spotCheckOrdering(candidateRow: [Float], referenceRow: [Float], eos: Int, sameWeights: Bool) {
    guard sameWeights else {
        print("# ordering spot-check: skipped (different weights; validated by the same-model run)")
        return
    }
    let vocab = candidateRow.count
    let cArg = argmaxIndex(candidateRow)
    let rArg = argmaxIndex(referenceRow)
    var sample = [0, 1000, vocab / 2, vocab - 1, cArg]
    if eos >= 0 && eos < vocab { sample.append(eos) }
    print("# ordering spot-check (position 0, shared context): vocab=\(vocab), argmax candidate=\(cArg) reference=\(rArg) \(cArg == rArg ? "(MATCH)" : "(MISMATCH!)")")
    var maxDiff: Float = 0
    for id in sample {
        let d = abs(candidateRow[id] - referenceRow[id])
        maxDiff = max(maxDiff, d)
        print("#   token id \(id): candidate=\(fmt(Double(candidateRow[id]), 4)) reference=\(fmt(Double(referenceRow[id]), 4)) |diff|=\(fmt(Double(d), 4))")
    }
    if cArg != rArg || maxDiff > 0.5 {
        print("# ordering spot-check: WARNING — differences exceed cross-implementation float noise; check the token-id ordering contract")
    } else {
        print("# ordering spot-check: OK (raw logits agree at sampled ids within float tolerance)")
    }
}

// MARK: - entry point

@main
struct Harness {
    static func main() async {
        let arguments = CommandLine.arguments
        guard arguments.count >= 2 else {
            usage()
            exit(2)
        }
        let flags = Flags(Array(arguments.dropFirst(2)))
        switch arguments[1] {
        case "corpus": runCorpus()
        case "verify": await runVerify(flags)
        case "bench": await runBench(flags)
        case "kl": await runKL(flags)
        default:
            usage()
            exit(2)
        }
    }

    static func usage() {
        print("""
        fastmlx-harness — conformance + precision-loss harness spine

        subcommands:
          corpus                              hermetic corpus + universal invariants (no model)
          verify --model <PATH>               triad: equivalence vs mlx-lm + engagement delta
          bench  --model <PATH>               stream-timed decode bench -> CSV (Release builds only)
          kl     --model <PATH>               KLDivergenceMetric vs mlx-lm reference
                 [--reference-model <PATH>]   (defaults to --model: pipeline proof)

        common flags: --python <PY=~/harness-venv/bin/python> --script <scripts/harness_reference.py>
        """)
    }
}
