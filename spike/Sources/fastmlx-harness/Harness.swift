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
/// "The capital of France is" matched 60/60 on the MoE target model. Still used as the `verify`
/// / `bench` default `--prompt` — the versioned corpus below is `kl`'s measurement input.
let knownGoodPrompt = "The capital of France is"
let benchPrompt = "Explain how continuous batching improves LLM serving throughput."

/// Loads the checked-in, versioned measurement corpus (Task 3): a stable `corpusId` + content
/// hash, entries tagged prose/code/long-context, replacing the formerly CLI-hardcoded `kl`
/// prompts. Never falls back to a hardcoded list on failure — a silently-substituted corpus would
/// make the recorded `corpusId` a lie.
func loadMeasurementCorpus(_ flags: Flags) throws -> MeasurementCorpus {
    let path = flags.string("corpus", default: "corpus/measurement-corpus-v2.json")
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    return try MeasurementCorpusLoader.load(from: data)
}

// MARK: - provenance (Task 5): every subcommand appends one JSONL record to --evidence.

func evidencePath(_ flags: Flags) -> String { flags.string("evidence", default: "harness-evidence.jsonl") }

struct VerifyPayload: Codable, Sendable {
    let prompt: String
    let promptTokens: Int
    let n: Int
    let mode: String
    let kvQuantTier: String
    let equivalencePassed: Bool
    let engaged: Bool
    let triadPassed: Bool
    /// Lossy-tier quality statistic (adjudicated re-spec): teacher-forced top-1 agreement
    /// RATE vs the same engine at fp16 KV — context-locked, so one flipped high-entropy
    /// token cannot cascade the way a free-running prefix does. nil in exact mode.
    let teacherForcedTop1AgreementRate: Double?
    /// In-graph cached-token count from the TurboQuant cache after the candidate run —
    /// the lossy triad's engagement marker (nil on fp16 runs).
    let turboquantTokens: Int?
}

struct KLPayload: Codable, Sendable {
    /// Candidate-side KV-cache tier ("fp16" when unset). The reference side is always fp16 KV.
    let kvQuantTier: String
    /// Headline for SHORT entries: median of PER-ENTRY medians (equal weight per entry). NOT a
    /// position-weighted pool -- see the pooled diagnostics below, which exist for visibility only.
    let klMedianNats: Double
    /// Headline for LONG-CONTEXT entries: median across long entries of the per-entry p95
    /// (`longContextTailKL`). Over natural long text the per-position median sits BELOW the
    /// same-weights noise floor (easy tokens dominate); KV-quant divergence is a TAIL
    /// phenomenon, so the tail statistic is the honest long-context headline. nil when the
    /// corpus has no long-context entries.
    let klLongContextTailP95Nats: Double?
    let klPooledMedianNats: Double
    let klPooledP95Nats: Double
    let pplCandidate: Double
    let pplReference: Double
    let pplDeltaPct: Double
    let totalPositions: Int
    let entryCount: Int
}

struct BenchPayload: Codable, Sendable {
    let label: String
    let workload: String
    let mode: String
    let decodeTokS: Double
    let ttftMs: Double
    let quant: String
    /// KV-cache tier the timed decode ran with ("fp16" when unset) — distinct from `quant`,
    /// which is the model's own weight quantization.
    let kvQuantTier: String
    let concurrency: Int
}

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
        print("usage: fastmlx-harness verify --model <PATH> [--prompt <TEXT>] [--n 60] [--min-prefix 30] [--kv-quant fp16|tq2.5|tq3.5] [--python <PY>] [--script <REF.py>] [--reference-model <PATH>] [--evidence <FILE=harness-evidence.jsonl>]")
        exit(2)
    }
    let prompt = flags.string("prompt", default: knownGoodPrompt)
    let n = flags.int("n", default: 60)
    let minPrefix = flags.int("min-prefix", default: 30)
    // KV-quant tier (Task 7): passed INTO the RunConfig, so the engine actually runs the
    // selected cache (TurboQuantKVCache for tq2.5/tq3.5) — no longer a declaration-only flag.
    let kvQuantTier = flags.string("kv-quant")
    guard KVCacheKind(kvQuant: kvQuantTier) != nil else {
        print("verify FAILED: unknown --kv-quant tier \(kvQuantTier ?? "nil") (known: fp16, tq2.5/tqB2, tq3.5/tqB3)")
        exit(2)
    }
    let mode = triadMode(forKVQuantTier: kvQuantTier)
    do {
        let (driver, tokenizer, eos) = try await loadSwiftDriver(modelPath: modelPath)
        let promptTokens = tokenizer.encode(text: prompt)
        let config = RunConfig(temperature: 0, maxTokens: n, kvQuant: kvQuantTier)

        let candidate = try await driver.generate(prompt: promptTokens, config: config)
        let decodeCount = candidate.engagement.counts["decode"] ?? 0
        var engaged = EngagementCheck(marker: "decode", floor: 1).passed(before: 0, after: decodeCount)

        let verdict: TriadVerdict
        var referenceVersions: ReferenceDriver.ReferenceVersions?
        var teacherForcedTop1: Double?  // context-locked top-1 agreement rate vs fp16 KV
        var turboquantTokens: Int?  // in-graph engagement marker from the quantized cache
        switch mode {
        case .exact:
            let reference = referenceDriver(flags, modelPath: modelPath, eos: eos)
            let refRun = try await reference.generate(prompt: promptTokens, config: config)
            referenceVersions = await reference.versionSink.versions
            let eq = EquivalenceCheck(minPrefix: minPrefix)
                .evaluate(candidate: candidate.tokens, reference: refRun.tokens)
            let comparable = min(candidate.tokens.count, refRun.tokens.count)
            print("prompt: \(String(reflecting: prompt)) (\(promptTokens.count) tokens), n=\(n), temp=0")
            print("equivalence (exact): identical-prefix \(eq.prefix)/\(comparable) (gate >= \(minPrefix)) -> \(eq.passed ? "PASS" : "FAIL")")
            verdict = TriadVerdict(equivalenceOK: eq.passed, engaged: engaged, acceptanceOK: nil)
            if !verdict.passed {
                print("candidate: \(candidate.tokens)")
                print("reference: \(refRun.tokens)")
            }
        case .lossy:
            // TurboQuant engagement (Task 7): the quantized cache's IN-GRAPH cached-token
            // count must have advanced past the prompt — a silent fp16 fallback cannot
            // fake this marker (fp16 runs don't emit it at all).
            let tqTokens = candidate.engagement.counts["turboquant_tokens"] ?? 0
            let tqEngaged = EngagementCheck(marker: "turboquant_tokens", floor: promptTokens.count + 1)
                .passed(before: 0, after: tqTokens)
            engaged = engaged && tqEngaged
            // non-crash: candidate already produced >=1 token (checked below via prefix).
            // non-NaN: scan a full-vocab logprobs pass for any non-finite value — this ALSO
            // exercises the harness scoring forward with the quantized cache (Phase 3's path).
            let rows = try await driver.logprobs(prompt: promptTokens, config: config)
            let allFinite = rows.allSatisfy { row in row.allSatisfy { $0.isFinite } }
            // Integration diagnostic (not a gate): the uncompiled scoring forward free-runs
            // its own greedy path at the SAME tier, so its argmax stream vs the compiled
            // decode's tokens separates compile-trace fidelity from codec loss. Near-tie
            // flips from kernel-fusion reduction order can still differ, so a long-but-not-
            // full match is float noise; an immediate mismatch would be a trace bug.
            let scoringGreedy = rows.map(argmaxIndex)
            let traceAgreement = identicalPrefix(candidate.tokens, scoringGreedy)
            print("compiled-vs-uncompiled greedy agreement (same tier): identical-prefix \(traceAgreement)/\(min(candidate.tokens.count, scoringGreedy.count))")
            // Perturbation-scale diagnostic: position 0 scores the SAME context (the bare
            // prompt) on both tiers, so the row-0 logit delta is the pure KV-quantization
            // perturbation on real tensors — O(0.1–1) means near-tie argmax flips (expected
            // lossy behavior), ≫1 means a broken path.
            let fp16Row0 = try await driver.logprobs(prompt: promptTokens, config: .greedy(maxTokens: 1)).first ?? []
            if let tqRow0 = rows.first, fp16Row0.count == tqRow0.count {
                var maxDiff: Float = 0
                var sumSq: Double = 0
                for i in fp16Row0.indices {
                    let d = fp16Row0[i] - tqRow0[i]
                    maxDiff = max(maxDiff, abs(d))
                    sumSq += Double(d) * Double(d)
                }
                let rms = (sumSq / Double(fp16Row0.count)).squareRoot()
                var top1 = -Float.infinity, top2 = -Float.infinity
                for v in fp16Row0 { if v > top1 { top2 = top1; top1 = v } else if v > top2 { top2 = v } }
                let fpArg = argmaxIndex(fp16Row0)
                let tqArg = argmaxIndex(tqRow0)
                print("pos-0 logit perturbation: max|Δ| \(fmt(Double(maxDiff), 3)), rms \(fmt(rms, 4)) (fp16 top-2 gap \(fmt(Double(top1 - top2), 3))); argmax \(fpArg == tqArg ? "MATCH" : "FLIP \(fpArg) -> \(tqArg)")")
            }
            // coherence canary: a fixed prompt whose greedy answer must contain a known
            // substring — run AT THE QUANTIZED TIER (config carries kvQuant).
            let canary = CoherenceCanary.capitalOfFrance
            let canaryTokens = tokenizer.encode(text: canary.prompt)
            let canaryRun = try await driver.generate(
                prompt: canaryTokens, config: RunConfig(temperature: 0, maxTokens: 20, kvQuant: kvQuantTier))
            let canaryText = tokenizer.decode(tokenIds: canaryRun.tokens)
            let canaryPassed = canary.passed(canaryText)
            let lossy = LossyEquivalenceCheck(minPrefix: 1)
                .evaluate(prefix: candidate.tokens.count, allFinite: allFinite, canaryPassed: canaryPassed)
            // Teacher-forced top-1 agreement vs the SAME engine at fp16 KV, BOTH TurboQuant
            // tiers (adjudicated re-spec): the earlier free-running identical-prefix gate was
            // chaotic — one flipped high-entropy token diverges everything after it, the same
            // lesson that made the KL metric teacher-forced. The context-locked per-position
            // agreement RATE is the stable statistic; it is REPORTED (recorded in evidence),
            // and the triad gates on the lossy floor above, not on prefix length.
            let fp16Run = try await driver.generate(prompt: promptTokens, config: .greedy(maxTokens: n))
            let forced = fp16Run.tokens
            if !forced.isEmpty {
                let tfRows = try await driver.logprobs(
                    prompt: promptTokens, forcedContinuation: forced, config: config)
                let agree = zip(tfRows.map(argmaxIndex), forced).filter(==).count
                let rate = Double(agree) / Double(forced.count)
                teacherForcedTop1 = rate
                print("teacher-forced top-1 agreement vs fp16 KV: \(agree)/\(forced.count) (\(fmt(rate * 100, 1))%)")
            }
            turboquantTokens = tqTokens
            print("prompt: \(String(reflecting: prompt)) (\(promptTokens.count) tokens), n=\(n), temp=0")
            print("equivalence (lossy, kv_quant_tier=\(kvQuantTier ?? "fp16")): produced=\(candidate.tokens.count), all-finite=\(allFinite), canary=\(canaryPassed ? "PASS" : "FAIL") -> \(lossy.passed ? "PASS" : "FAIL")")
            if !lossy.reasons.isEmpty { print("  reasons: \(lossy.reasons.joined(separator: "; "))") }
            verdict = TriadVerdict(equivalenceOK: lossy.passed, engaged: engaged, acceptanceOK: nil)
            if !verdict.passed { print("candidate: \(candidate.tokens)") }
        }

        print("kv_quant_tier: \(kvQuantTier ?? "fp16") (mode=\(mode.rawValue))")
        if let tq = turboquantTokens {
            print("engagement:  decode counter 0 -> \(decodeCount) (floor 1); turboquant_tokens 0 -> \(tq) (floor \(promptTokens.count + 1)) -> \(engaged ? "PASS" : "FAIL")")
        } else {
            print("engagement:  decode counter 0 -> \(decodeCount) (floor 1) -> \(engaged ? "PASS" : "FAIL")")
        }
        print("acceptance:  n/a (no spec-decode configured)")
        print("triad: \(verdict.passed ? "PASS" : "FAIL")")

        let (provenance, _) = ProvenanceCLI.build(modelPath: modelPath, referenceVersions: referenceVersions, corpus: nil)
        let payload = VerifyPayload(
            prompt: prompt, promptTokens: promptTokens.count, n: n, mode: mode.rawValue,
            kvQuantTier: kvQuantTier ?? "fp16", equivalencePassed: verdict.equivalenceOK,
            engaged: verdict.engaged, triadPassed: verdict.passed,
            teacherForcedTop1AgreementRate: teacherForcedTop1, turboquantTokens: turboquantTokens)
        appendJSONLRecord(ResultRecord(subcommand: "verify", provenance: provenance, payload: payload), to: evidencePath(flags))

        if !verdict.passed { exit(1) }
    } catch {
        print("verify FAILED: \(error)")
        exit(1)
    }
}

// MARK: - bench (cell matrix -> CSV)

func runBench(_ flags: Flags) async {
    guard let modelPath = flags.string("model") else {
        print("usage: fastmlx-harness bench --model <PATH> [--prompt <TEXT>] [--max-tokens 256] [--runs 3] [--label <L>] [--csv <FILE>] [--evidence <FILE=harness-evidence.jsonl>]")
        exit(2)
    }
    do {
        try assertReleaseBuild()
        let prompt = flags.string("prompt", default: benchPrompt)
        let maxTokens = flags.int("max-tokens", default: 256)
        let runs = flags.int("runs", default: 3)
        let label = flags.string("label", default: "harness")
        // KV tier for the timed decode (Task 8: the materialize-then-attend perf cost).
        let kvQuantTier = flags.string("kv-quant")
        guard KVCacheKind(kvQuant: kvQuantTier) != nil else {
            print("bench FAILED: unknown --kv-quant tier \(kvQuantTier ?? "nil") (known: fp16, tq2.5/tqB2, tq3.5/tqB3)")
            exit(2)
        }
        let (driver, tokenizer, _) = try await loadSwiftDriver(modelPath: modelPath)

        let modelName = URL(fileURLWithPath: modelPath).lastPathComponent
        // Task 5: the model's OWN declared quantization (config.json), not a dirname-substring
        // guess — a mislabeled checkpoint directory can no longer record the wrong tier.
        let quant = ProvenanceCLI.modelConfig(at: modelPath).quant.label
        let hardware = ProvenanceCLI.chipBrand()
        let cell = Cell(workload: .decode, mode: .none, model: modelName, quant: quant, concurrency: 1)
        let nonce = String(Int.random(in: 0..<1_000_000))

        var ttfts: [Double] = []
        let agg = try await BenchRunner().run(cell: cell, iterations: runs + 1, nonce: nonce, basePrompt: prompt) { i, salted in
            let promptTokens = tokenizer.encode(text: salted)
            let result = try await driver.generate(
                prompt: promptTokens,
                config: RunConfig(temperature: 0, maxTokens: maxTokens, kvQuant: kvQuantTier))
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
            quant: quant, concurrency: 1, hardware: hardware)
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

        let (provenance, _) = ProvenanceCLI.build(modelPath: modelPath, referenceVersions: nil, corpus: nil)
        let payload = BenchPayload(
            label: label, workload: Workload.decode.rawValue, mode: Mode.none.rawValue,
            decodeTokS: row.decodeTokS, ttftMs: row.ttftMs, quant: quant,
            kvQuantTier: kvQuantTier ?? "fp16", concurrency: 1)
        appendJSONLRecord(ResultRecord(subcommand: "bench", provenance: provenance, payload: payload), to: evidencePath(flags))
    } catch BenchGuardError.debugBuild {
        print("bench FAILED: Debug build — perf numbers would be meaningless. Build with -configuration Release.")
        exit(1)
    } catch {
        print("bench FAILED: \(error)")
        exit(1)
    }
}

// MARK: - kl (KLDivergenceMetric: candidate vs reference, TEACHER-FORCED)

func runKL(_ flags: Flags) async {
    guard let modelPath = flags.string("model") else {
        print("usage: fastmlx-harness kl --model <PATH> [--reference-model <PATH>] [--positions 24] [--corpus <FILE>] [--long-context-sample-positions 128] [--python <PY>] [--script <REF.py>] [--evidence <FILE=harness-evidence.jsonl>]")
        exit(2)
    }
    do {
        let positions = flags.int("positions", default: 24)
        let longContextSampleSize = flags.int("long-context-sample-positions", default: 128)
        // Candidate-side KV tier (Task 8): the CANDIDATE scores with the selected cache;
        // the reference NEVER sees kvQuant (referenceConfig strips it) — it is the baseline.
        let kvQuantTier = flags.string("kv-quant")
        guard KVCacheKind(kvQuant: kvQuantTier) != nil else {
            print("kl FAILED: unknown --kv-quant tier \(kvQuantTier ?? "nil") (known: fp16, tq2.5/tqB2, tq3.5/tqB3)")
            exit(2)
        }
        let corpus = try loadMeasurementCorpus(flags)
        let (driver, tokenizer, eos) = try await loadSwiftDriver(modelPath: modelPath)
        let reference = referenceDriver(flags, modelPath: modelPath, eos: eos)
        let sameWeights = reference.modelPath == modelPath
        let config = RunConfig(temperature: 0, maxTokens: positions, kvQuant: kvQuantTier)
        let referenceConfig = RunConfig.greedy(maxTokens: positions)
        let shortEntries = corpus.entries(tagged: .prose) + corpus.entries(tagged: .code)
        let longEntries = corpus.entries(tagged: .longContext)

        print("candidate: Swift engine on \(modelPath) (kv_quant_tier=\(kvQuantTier ?? "fp16"))")
        print("reference: mlx-lm on \(reference.modelPath) (fp16 KV)")
        print("corpus: \(corpus.corpusId) (content hash \(corpus.contentHash), \(corpus.entries.count) entries)")
        print(sameWeights
            ? "# SAME weights both sides -> this is a PIPELINE PROOF (cross-implementation float-reduction noise), not a quantization-loss measurement."
            : "# DIFFERENT weights -> candidate-vs-reference quantization/behavior divergence measurement.")
        print("# TEACHER-FORCED: both sides score the reference's greedy continuation, so every position is context-locked.")

        var allKLs: [Double] = []
        // One median per ENTRY, not per position — the headline. Pooling raw per-position KLs
        // across entries with wildly different position counts (three ~24-position prompts vs a
        // 128-sampled-position long-context entry) lets the larger entry's positions dominate the
        // pooled median even though it is exactly one measurement among four; per-entry medians
        // give every entry equal weight regardless of how many positions it was scored at.
        var entryMedians: [Double] = []
        var candNLLTotal = 0.0, refNLLTotal = 0.0, totalPositions = 0
        var spotChecked = false
        for entry in shortEntries {
            let prompt = tokenizer.encode(text: entry.text)
            // teacherForcedScores + perPositionKLs + medianOf ARE KLDivergenceMetric's own
            // internals: the headline below equals metric.measure on the same prompts.
            let s = try await teacherForcedScores(
                driver: driver, reference: reference, prompt: prompt, config: config,
                referenceConfig: referenceConfig)
            guard let c0 = s.candidateRows.first, let r0 = s.referenceRows.first else {
                print("kl FAILED: empty logprobs for entry \(entry.id)")
                exit(1)
            }
            guard c0.count == r0.count else {
                print("kl FAILED: vocab mismatch (candidate \(c0.count) vs reference \(r0.count)) — index-aligned KL would be meaningless")
                exit(1)
            }
            if !spotChecked {
                // Position 0's context is the bare prompt on both sides — shared by construction.
                spotCheckOrdering(candidateRow: c0, referenceRow: r0, eos: eos, sameWeights: sameWeights)
                spotChecked = true
            }
            let kls = perPositionKLs(reference: s.referenceRows, candidate: s.candidateRows)
            // Diagnostic: at how many positions would the candidate have picked the same token
            // the reference did, given the reference's context? (Top-1 agreement, not a gate.)
            let agree = zip(s.candidateRows.map(argmaxIndex), s.continuation).filter(==).count
            allKLs.append(contentsOf: kls)
            entryMedians.append(medianOf(kls))
            // Perplexity pools NLL over positions from the SAME rows (identical math to
            // teacherForcedPerplexities — one forward pass serves both metrics).
            let n = Double(s.continuation.count)
            candNLLTotal += meanNLL(rows: s.candidateRows, tokens: s.continuation) * n
            refNLLTotal += meanNLL(rows: s.referenceRows, tokens: s.continuation) * n
            totalPositions += s.continuation.count
            print("entry \(entry.id) (\(entry.tag.rawValue)): forced-positions=\(kls.count), top1-agreement=\(agree)/\(s.continuation.count), median KL=\(sci(medianOf(kls)))")
        }

        // Long-context entries (Task 3): teacher-forced AGAINST THEMSELVES (wikitext-perplexity
        // style, no "generate a continuation" step) at a SAMPLED subset of positions — a >=4K
        // token entry scored at every position would exhaust memory (~0.6MB/row x thousands of
        // positions x 2 drivers).
        var longEntryKLs: [[Double]] = []
        for entry in longEntries {
            let docTokens = tokenizer.encode(text: entry.text)
            guard docTokens.count > 1 else {
                print("kl FAILED: long-context entry \(entry.id) tokenized to \(docTokens.count) tokens, need > 1")
                exit(1)
            }
            let prompt = [docTokens[0]]
            let continuation = Array(docTokens.dropFirst())
            let sampled = evenlySpacedPositions(total: continuation.count, sampleSize: longContextSampleSize)
            let s = try await teacherForcedScoresAtSampledPositions(
                driver: driver, reference: reference, prompt: prompt, continuation: continuation,
                positions: sampled, config: config, referenceConfig: referenceConfig)
            let kls = perPositionKLs(reference: s.referenceRows, candidate: s.candidateRows)
            allKLs.append(contentsOf: kls)
            longEntryKLs.append(kls)
            let n = Double(s.forcedTokens.count)
            candNLLTotal += meanNLL(rows: s.candidateRows, tokens: s.forcedTokens) * n
            refNLLTotal += meanNLL(rows: s.referenceRows, tokens: s.forcedTokens) * n
            totalPositions += s.forcedTokens.count
            // The long-context ENTRY headline is the TAIL (p95), not the median: over natural
            // long text the median sits below the same-weights noise floor (easy tokens both
            // quants agree on dominate it), while KV-quant loss accrues in the tail.
            print("entry \(entry.id) (long-context, \(docTokens.count) doc tokens): sampled-positions=\(kls.count)/\(continuation.count), p95 KL=\(sci(quantile(kls, 0.95))), median KL=\(sci(medianOf(kls)))")
        }

        let pooledP95 = quantile(allKLs, 0.95)
        let headlineMedian = medianOf(entryMedians)
        let longContextTail: Double? = longEntryKLs.isEmpty ? nil : longContextTailKL(perEntryKLs: longEntryKLs)
        // HEADLINES: short entries -> median of PER-ENTRY medians (equal weight per entry);
        // long-context entries -> median of PER-ENTRY p95s (`longContextTailKL`), because the
        // KV-quant divergence signal lives in the tail at long context. The pooled numbers below
        // are a diagnostic only — a heavily-sampled entry would dominate a position-weighted pool.
        print("kl_median (headline, SHORT entries, median of \(entryMedians.count) per-entry medians): \(sci(headlineMedian)) nats")
        if let longContextTail {
            print("kl_long_context_tail_p95 (headline, LONG-CONTEXT entries, median of \(longEntryKLs.count) per-entry p95s): \(sci(longContextTail)) nats")
        }
        print("kl_pooled_median (diagnostic, position-weighted, all \(allKLs.count) positions -- do NOT use as headline): \(sci(medianOf(allKLs))) nats")
        print("kl_pooled_p95    (diagnostic, position-weighted, all positions): \(sci(pooledP95)) nats")

        let pplPair = PerplexityPair(
            candidate: exp(candNLLTotal / Double(totalPositions)),
            reference: exp(refNLLTotal / Double(totalPositions)))
        print("ppl_candidate (teacher-forced, pooled \(totalPositions) positions): \(fmt(pplPair.candidate, 4))")
        print("ppl_reference (its own greedy continuation): \(fmt(pplPair.reference, 4))")
        print("ppl_delta (PerplexityMetric): \(String(format: "%+.2f%%", pplPair.relativeDelta * 100)) (dial gate: <= 1%)")

        let referenceVersions = await reference.versionSink.versions
        let (provenance, _) = ProvenanceCLI.build(modelPath: modelPath, referenceVersions: referenceVersions, corpus: corpus)
        let payload = KLPayload(
            kvQuantTier: kvQuantTier ?? "fp16",
            klMedianNats: headlineMedian, klLongContextTailP95Nats: longContextTail,
            klPooledMedianNats: medianOf(allKLs), klPooledP95Nats: pooledP95,
            pplCandidate: pplPair.candidate, pplReference: pplPair.reference, pplDeltaPct: pplPair.relativeDelta * 100,
            totalPositions: totalPositions, entryCount: corpus.entries.count)
        appendJSONLRecord(ResultRecord(subcommand: "kl", provenance: provenance, payload: payload), to: evidencePath(flags))
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
        case "ctxprobe": await runCtxProbe(flags)
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
                 [--kv-quant <TIER>]          KV-quant tier, RUN BY THE ENGINE: nil/fp16 = exact
                                               triad (fp16 cache); tq2.5|tqB2 / tq3.5|tqB3 select
                                               the TurboQuant KV cache and the lossy triad
                                               (non-crash + non-NaN + canary + engagement), and
                                               REPORT teacher-forced top-1 agreement vs fp16 KV
          bench  --model <PATH>               stream-timed decode bench -> CSV (Release builds only)
                 [--kv-quant <TIER>]          KV tier for the timed decode (fp16|tq2.5|tq3.5)
          kl     --model <PATH>               KLDivergenceMetric vs mlx-lm reference
                 [--kv-quant <TIER>]          CANDIDATE-side KV tier (reference stays fp16 KV)
                 [--reference-model <PATH>]   (defaults to --model: pipeline proof)
                 [--corpus <FILE=corpus/measurement-corpus-v2.json>]
                 [--long-context-sample-positions 128]

        common flags: --python <PY=~/harness-venv/bin/python> --script <scripts/harness_reference.py>
        """)
    }
}
