import Foundation
import HarnessCore
import MLXLMCommon
import SpikeCore

/// Minimal `--flag value` parser (no ArgumentParser dependency), kept pure in HarnessCore so
/// malformed promotion-gate commands are regression-tested off-box.
typealias Flags = CLIFlags

/// Known-good prompts from the spike's equivalence work (see 2026-07-08-swift-spike-verdict.md):
/// "The capital of France is" matched 60/60 on the MoE target model. Still used as the `verify`
/// / `bench` default `--prompt` — the versioned corpus below is `kl`'s measurement input.
let knownGoodPrompt = "The capital of France is"
let benchPrompt = "Explain how continuous batching improves LLM serving throughput."
let knownKVQuantTiers = (
    ["fp16"] + AffineKVTier.allCases.map(\.rawValue)
        + KVarNKVTier.allCases.map(\.rawValue)
        + TurboQuantTier.allCases.flatMap { [$0.harnessSlot, $0.rawValue] }
).joined(separator: ", ")
let kvQuantUsageTiers = (
    ["fp16"] + AffineKVTier.allCases.map(\.rawValue)
        + KVarNKVTier.allCases.map(\.rawValue)
        + TurboQuantTier.allCases.map(\.harnessSlot)
).joined(separator: "|")

/// Strictly parse the CLI tier and normalize explicit fp16 to the engine's nil baseline.
/// A present flag without a value throws instead of silently selecting fp16.
func requestedKVQuantTier(_ flags: Flags) throws -> String? {
    let raw = try flags.strictString("kv-quant", default: "fp16")
    return raw == "fp16" ? nil : raw
}

/// Default prompt for `verify --spec`: a raw-completion repetition shape (the engine feeds
/// prompts untemplated, so an "instruction" would not reliably be followed — a self-continuing
/// repeated structure is). Greedy continuation keeps emitting the repeated line, which is
/// exactly the case PLD drafts from, so the engagement gate (drafting happened) can bind.
let specVerifyPrompt = """
Inventory report, line 1: the warehouse stores red apples, green pears, yellow bananas, and blue plums.
Inventory report, line 2: the warehouse stores red apples, green pears, yellow bananas, and blue plums.
Inventory report, line 3: the warehouse stores
"""

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
    /// Legacy TurboQuant engagement marker (nil for fp16/affine).
    let turboquantTokens: Int?
    /// Native-affine engagement, exact persistent arrays/control, and logical materialization
    /// workspace (nil for fp16/TurboQuant).
    let affineTokens: Int?
    let affinePayloadBytes: Int?
    let affineMetadataBytes: Int?
    let affineControlBytes: Int?
    let affineWorkspaceBytes: Int?
}

/// Evidence record for `verify --spec`: the spec-decode exactness triad — PLD-on vs PLD-off on
/// the SAME engine (byte-identical token streams at temp 0) + the engagement delta (drafting
/// actually happened; a zero-draft run would make the equivalence vacuous).
struct SpecVerifyPayload: Codable, Sendable {
    let prompt: String
    let promptTokens: Int
    let n: Int
    let spec: String
    let ngram: Int
    let maxDraft: Int
    let compiledVerify: Bool
    /// PLD-on and PLD-off token streams are fully identical (not just a prefix).
    let byteIdentical: Bool
    /// Length of the identical prefix (== token count when byteIdentical).
    let identicalPrefix: Int
    let tokensOn: Int
    let tokensOff: Int
    let drafted: Int
    let accepted: Int
    let acceptanceRate: Double?
    let verifySteps: Int
    let normalSteps: Int
    let gateDisabledSteps: Int
    let engaged: Bool
    let triadPassed: Bool
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
    // Spec-decode telemetry (mode == "pld"): totals over the timed (post-warmup) runs.
    // nil on plain runs so pre-spec evidence records keep decoding.
    let specNgram: Int?
    let specMaxDraft: Int?
    let specCompiledVerify: Bool?
    let specDrafted: Int?
    let specAccepted: Int?
    let specAcceptanceRate: Double?
    let specVerifySteps: Int?
    let specNormalSteps: Int?
    /// Steps taken while the yield-gate had PLD disabled — the "gate kept a low-repetition
    /// workload flat" evidence the shape-(c) verdict reads.
    let specGateDisabledSteps: Int?
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
    do {
        let requestedSpec = try flags.strictString("spec", default: "")
        if !requestedSpec.isEmpty {
            await runVerifySpec(flags)
            return
        }
    } catch {
        print("verify FAILED: \(error)")
        exit(2)
    }
    guard let modelPath = flags.string("model") else {
        print("usage: fastmlx-harness verify --model <PATH> [--prompt <TEXT>] [--n 60] [--min-prefix 30] [--kv-quant <\(kvQuantUsageTiers)>] [--python <PY>] [--script <REF.py>] [--reference-model <PATH>] [--evidence <FILE=harness-evidence.jsonl>]")
        exit(2)
    }
    let prompt = flags.string("prompt", default: knownGoodPrompt)
    let n = flags.int("n", default: 60)
    let minPrefix = flags.int("min-prefix", default: 30)
    let kvQuantTier: String?
    do {
        kvQuantTier = try requestedKVQuantTier(flags)
    } catch {
        print("verify FAILED: \(error)")
        exit(2)
    }
    guard let cacheKind = KVCacheKind(kvQuant: kvQuantTier) else {
        print("verify FAILED: unknown --kv-quant tier \(kvQuantTier ?? "fp16") (known: \(knownKVQuantTiers))")
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
        var quantizedMarker: String?
        var quantizedTokens: Int?
        var turboquantTokens: Int?
        var affineTokens: Int?
        var affinePayloadBytes: Int?
        var affineMetadataBytes: Int?
        var affineControlBytes: Int?
        var affineWorkspaceBytes: Int?
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
            let marker: String
            switch cacheKind {
            case .fp16:
                throw SwiftEngineDriverError.unsupportedConfig(
                    "lossy triad selected for fp16 KV")
            case .affine:
                marker = "affine_tokens"
            case .turboQuant:
                marker = "turboquant_tokens"
            case .kvarn:
                marker = "kvarn_tokens"
            }
            let cachedTokens = candidate.engagement.counts[marker] ?? 0
            let cacheEngaged = EngagementCheck(
                marker: marker, floor: promptTokens.count + 1
            ).passed(before: 0, after: cachedTokens)
            engaged = engaged && cacheEngaged
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
            if let quantizedRow0 = rows.first, fp16Row0.count == quantizedRow0.count {
                var maxDiff: Float = 0
                var sumSq: Double = 0
                for i in fp16Row0.indices {
                    let d = fp16Row0[i] - quantizedRow0[i]
                    maxDiff = max(maxDiff, abs(d))
                    sumSq += Double(d) * Double(d)
                }
                let rms = (sumSq / Double(fp16Row0.count)).squareRoot()
                var top1 = -Float.infinity, top2 = -Float.infinity
                for v in fp16Row0 { if v > top1 { top2 = top1; top1 = v } else if v > top2 { top2 = v } }
                let fpArg = argmaxIndex(fp16Row0)
                let quantizedArg = argmaxIndex(quantizedRow0)
                print("pos-0 logit perturbation: max|Δ| \(fmt(Double(maxDiff), 3)), rms \(fmt(rms, 4)) (fp16 top-2 gap \(fmt(Double(top1 - top2), 3))); argmax \(fpArg == quantizedArg ? "MATCH" : "FLIP \(fpArg) -> \(quantizedArg)")")
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
            // Teacher-forced top-1 agreement vs the SAME engine at fp16 KV (adjudicated
            // re-spec): the earlier free-running identical-prefix gate was
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
            quantizedMarker = marker
            quantizedTokens = cachedTokens
            switch cacheKind {
            case .fp16:
                break
            case .affine:
                affineTokens = cachedTokens
                affinePayloadBytes = candidate.engagement.counts["affine_payload_bytes"]
                affineMetadataBytes = candidate.engagement.counts["affine_metadata_bytes"]
                affineControlBytes = candidate.engagement.counts["affine_control_bytes"]
                affineWorkspaceBytes = candidate.engagement.counts["affine_workspace_bytes"]
            case .turboQuant:
                turboquantTokens = cachedTokens
            case .kvarn:
                break
            }
            print("prompt: \(String(reflecting: prompt)) (\(promptTokens.count) tokens), n=\(n), temp=0")
            print("equivalence (lossy, kv_quant_tier=\(kvQuantTier ?? "fp16")): produced=\(candidate.tokens.count), all-finite=\(allFinite), canary=\(canaryPassed ? "PASS" : "FAIL") -> \(lossy.passed ? "PASS" : "FAIL")")
            if !lossy.reasons.isEmpty { print("  reasons: \(lossy.reasons.joined(separator: "; "))") }
            verdict = TriadVerdict(equivalenceOK: lossy.passed, engaged: engaged, acceptanceOK: nil)
            if !verdict.passed { print("candidate: \(candidate.tokens)") }
        }

        print("kv_quant_tier: \(kvQuantTier ?? "fp16") (mode=\(mode.rawValue))")
        if let marker = quantizedMarker, let cachedTokens = quantizedTokens {
            print("engagement:  decode counter 0 -> \(decodeCount) (floor 1); \(marker) 0 -> \(cachedTokens) (floor \(promptTokens.count + 1)) -> \(engaged ? "PASS" : "FAIL")")
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
            teacherForcedTop1AgreementRate: teacherForcedTop1,
            turboquantTokens: turboquantTokens,
            affineTokens: affineTokens,
            affinePayloadBytes: affinePayloadBytes,
            affineMetadataBytes: affineMetadataBytes,
            affineControlBytes: affineControlBytes,
            affineWorkspaceBytes: affineWorkspaceBytes)
        appendJSONLRecord(ResultRecord(subcommand: "verify", provenance: provenance, payload: payload), to: evidencePath(flags))

        if !verdict.passed { exit(1) }
    } catch {
        print("verify FAILED: \(error)")
        exit(1)
    }
}

// MARK: - verify --spec (spec-decode exactness triad)

/// The spec-decode equivalence gate: at temp 0, PLD-on must be BYTE-IDENTICAL to PLD-off —
/// speculation changes how many tokens a forward emits, never which. The reference here is the
/// SAME Swift engine with speculation off (not the mlx-lm process): the claim under test is
/// "the spec path is a pure speed transform of this engine's own greedy loop", so the engine
/// is its own reference and the mlx-lm cross-check stays with the plain `verify`.
func runVerifySpec(_ flags: Flags) async {
    guard let modelPath = flags.string("model") else {
        print("usage: fastmlx-harness verify --model <PATH> --spec pld [--kv-quant fp16] [--ngram 3] [--max-draft 8] [--compiled-verify false] [--prompt <TEXT>] [--n 60] [--min-prefix <N=--n>] [--evidence <FILE>]")
        exit(2)
    }
    let spec: String
    let kvQuantTier: String?
    do {
        spec = try flags.strictString("spec", default: "pld")
        kvQuantTier = try requestedKVQuantTier(flags)
    } catch {
        print("verify FAILED: \(error)")
        exit(2)
    }
    guard spec == "pld" else {
        print("verify FAILED: unknown --spec drafter \(spec) (known: pld)")
        exit(2)
    }
    if let rejectedTier = kvQuantTier {
        print(
            "verify FAILED: specDecode=pld with kvQuant=\(rejectedTier) "
                + "(unmeasured combination; use fp16)")
        exit(2)
    }
    let prompt = flags.string("prompt", default: specVerifyPrompt)
    let n = flags.int("n", default: 60)
    // Default gate: the WHOLE run must match (byte-identical is the headline), and it must be
    // at least this long — a 3-token identical run proves nothing about the accept-walk.
    let minPrefix = flags.int("min-prefix", default: n)
    let ngram = flags.int("ngram", default: 3)
    let maxDraft = flags.int("max-draft", default: 8)
    let compiledVerify = flags.string("compiled-verify", default: "false") == "true"
    do {
        let (driver, tokenizer, _) = try await loadSwiftDriver(modelPath: modelPath)
        let promptTokens = tokenizer.encode(text: prompt)
        let offConfig = RunConfig(temperature: 0, maxTokens: n)
        let onConfig = RunConfig(
            temperature: 0, maxTokens: n, specDecode: spec,
            specNgram: ngram, specMaxDraft: maxDraft, specCompiledVerify: compiledVerify)

        let off = try await driver.generate(prompt: promptTokens, config: offConfig)
        let on = try await driver.generate(prompt: promptTokens, config: onConfig)

        let prefix = identicalPrefix(on.tokens, off.tokens)
        let byteIdentical = on.tokens == off.tokens
        let equivalenceOK = byteIdentical && on.tokens.count >= minPrefix
        let drafted = on.engagement.counts["spec_drafted"] ?? 0
        let accepted = on.engagement.counts["spec_accepted"] ?? 0
        let verifySteps = on.engagement.counts["spec_verify_steps"] ?? 0
        let normalSteps = on.engagement.counts["spec_normal_steps"] ?? 0
        let gateDisabledSteps = on.engagement.counts["spec_gate_disabled_steps"] ?? 0
        // Engagement DELTA: tokens were actually drafted AND at least one was accepted — an
        // all-rejected run is still exact but exercises only the bonus-token path.
        let engaged = EngagementCheck(marker: "spec_drafted", floor: 1).passed(before: 0, after: drafted)
            && EngagementCheck(marker: "spec_accepted", floor: 1).passed(before: 0, after: accepted)
        let verdict = TriadVerdict(equivalenceOK: equivalenceOK, engaged: engaged, acceptanceOK: nil)

        print("prompt: \(String(reflecting: prompt.prefix(80))) (\(promptTokens.count) tokens), n=\(n), temp=0")
        print("spec: \(spec) ngram=\(ngram) max-draft=\(maxDraft) verify-forward=\(compiledVerify ? "compiled(fixed-K)" : "uncompiled")")
        print("equivalence (spec-exact): PLD-on vs PLD-off byte-identical=\(byteIdentical), identical-prefix \(prefix)/\(min(on.tokens.count, off.tokens.count)), length \(on.tokens.count) (gate: identical AND >= \(minPrefix)) -> \(equivalenceOK ? "PASS" : "FAIL")")
        if !byteIdentical {
            let onTok = prefix < on.tokens.count ? String(on.tokens[prefix]) : "<end>"
            let offTok = prefix < off.tokens.count ? String(off.tokens[prefix]) : "<end>"
            print("  first divergence at position \(prefix): spec-on=\(onTok) spec-off=\(offTok)")
            print("  spec-on:  \(on.tokens)")
            print("  spec-off: \(off.tokens)")
        }
        let rate = on.acceptanceRate.map { fmt($0 * 100, 1) + "%" } ?? "n/a"
        print("engagement:  drafted 0 -> \(drafted), accepted 0 -> \(accepted) (floor 1 each) -> \(engaged ? "PASS" : "FAIL")")
        print("acceptance:  \(accepted)/\(drafted) drafts accepted (\(rate)); verify-steps=\(verifySteps), normal-steps=\(normalSteps), gate-disabled-steps=\(gateDisabledSteps) [reported, not gated]")
        print("triad: \(verdict.passed ? "PASS" : "FAIL")")

        let (provenance, _) = ProvenanceCLI.build(modelPath: modelPath, referenceVersions: nil, corpus: nil)
        let payload = SpecVerifyPayload(
            prompt: prompt, promptTokens: promptTokens.count, n: n, spec: spec, ngram: ngram,
            maxDraft: maxDraft, compiledVerify: compiledVerify, byteIdentical: byteIdentical,
            identicalPrefix: prefix, tokensOn: on.tokens.count, tokensOff: off.tokens.count,
            drafted: drafted, accepted: accepted, acceptanceRate: on.acceptanceRate,
            verifySteps: verifySteps, normalSteps: normalSteps, gateDisabledSteps: gateDisabledSteps,
            engaged: engaged, triadPassed: verdict.passed)
        appendJSONLRecord(ResultRecord(subcommand: "verify-spec", provenance: provenance, payload: payload), to: evidencePath(flags))

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
        // Validate the requested measurement before the build-mode or model-load gates so a
        // misspelled/missing flag cannot be reported as an unrelated environment failure.
        let kvQuantTier = try requestedKVQuantTier(flags)
        guard KVCacheKind(kvQuant: kvQuantTier) != nil else {
            print("bench FAILED: unknown --kv-quant tier \(kvQuantTier ?? "fp16") (known: \(knownKVQuantTiers))")
            exit(2)
        }
        let requestedSpec = try flags.strictString("spec", default: "")
        let spec: String? = requestedSpec.isEmpty ? nil : requestedSpec
        if let spec, spec != "pld" {
            print("bench FAILED: unknown --spec drafter \(spec) (known: pld)")
            exit(2)
        }
        if spec != nil, let rejectedTier = kvQuantTier {
            print(
                "bench FAILED: specDecode=pld with kvQuant=\(rejectedTier) "
                    + "(unmeasured combination; use fp16)")
            exit(2)
        }
        try assertReleaseBuild()
        let prompt = flags.string("prompt", default: benchPrompt)
        let maxTokens = flags.int("max-tokens", default: 256)
        let runs = flags.int("runs", default: 3)
        let label = flags.string("label", default: "harness")
        // Spec-decode arm (Task 6): `--spec pld` times the SAME decode workload through the
        // speculative path; the CSV/evidence `mode` column records which arm produced the number.
        let ngram = flags.int("ngram", default: 3)
        let maxDraft = flags.int("max-draft", default: 8)
        let compiledVerify = flags.string("compiled-verify", default: "false") == "true"
        let mode: Mode = spec == nil ? .none : .pld
        let (driver, tokenizer, _) = try await loadSwiftDriver(modelPath: modelPath)

        let modelName = URL(fileURLWithPath: modelPath).lastPathComponent
        // Task 5: the model's OWN declared quantization (config.json), not a dirname-substring
        // guess — a mislabeled checkpoint directory can no longer record the wrong tier.
        let quant = ProvenanceCLI.modelConfig(at: modelPath).quant.label
        let hardware = ProvenanceCLI.chipBrand()
        let cell = Cell(workload: .decode, mode: mode, model: modelName, quant: quant, concurrency: 1)
        let nonce = String(Int.random(in: 0..<1_000_000))

        var ttfts: [Double] = []
        var draftedTotal = 0, acceptedTotal = 0
        var verifyStepsTotal = 0, normalStepsTotal = 0, gateDisabledTotal = 0
        let agg = try await BenchRunner().run(cell: cell, iterations: runs + 1, nonce: nonce, basePrompt: prompt) { i, salted in
            let promptTokens = tokenizer.encode(text: salted)
            let result = try await driver.generate(
                prompt: promptTokens,
                config: RunConfig(
                    temperature: 0, maxTokens: maxTokens, specDecode: spec, specNgram: ngram,
                    specMaxDraft: maxDraft, specCompiledVerify: compiledVerify, kvQuant: kvQuantTier))
            guard !result.tokenTimes.isEmpty else {
                print("# run \(i): produced zero tokens -> skipped")
                return nil
            }
            let metrics = DecodeMetrics(submitTime: result.submitTime, tokenTimes: result.tokenTimes)
            let tag = i == 0 ? "warmup (dropped)" : "run \(i)"
            var specNote = ""
            if spec != nil {
                let drafted = result.engagement.counts["spec_drafted"] ?? 0
                let accepted = result.engagement.counts["spec_accepted"] ?? 0
                let gateDisabled = result.engagement.counts["spec_gate_disabled_steps"] ?? 0
                let rate = result.acceptanceRate.map { fmt($0 * 100, 1) + "%" } ?? "n/a"
                specNote = ", drafted=\(drafted), accepted=\(accepted) (\(rate)), gate-disabled-steps=\(gateDisabled)"
                if i > 0 {
                    draftedTotal += drafted
                    acceptedTotal += accepted
                    verifyStepsTotal += result.engagement.counts["spec_verify_steps"] ?? 0
                    normalStepsTotal += result.engagement.counts["spec_normal_steps"] ?? 0
                    gateDisabledTotal += gateDisabled
                }
            }
            print("# \(tag): \(metrics.generatedTokenCount) tokens, ttft=\(fmt(metrics.ttftSeconds))s, decode_tok_s=\(fmt(metrics.decodeTokensPerSecond ?? .nan, 2))\(specNote)")
            if i > 0 { ttfts.append(metrics.ttftSeconds) }
            return metrics.decodeTokensPerSecond
        }
        guard agg.runs > 0 else {
            print("bench FAILED: no measurable post-warmup runs")
            exit(1)
        }
        let avgTtftMs = ttfts.isEmpty ? 0 : ttfts.reduce(0, +) / Double(ttfts.count) * 1000
        let row = BenchRow(
            label: label, workload: .decode, mode: mode, model: modelName,
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
            label: label, workload: Workload.decode.rawValue, mode: mode.rawValue,
            decodeTokS: row.decodeTokS, ttftMs: row.ttftMs, quant: quant,
            kvQuantTier: kvQuantTier ?? "fp16", concurrency: 1,
            specNgram: spec == nil ? nil : ngram,
            specMaxDraft: spec == nil ? nil : maxDraft,
            specCompiledVerify: spec == nil ? nil : compiledVerify,
            specDrafted: spec == nil ? nil : draftedTotal,
            specAccepted: spec == nil ? nil : acceptedTotal,
            specAcceptanceRate: spec == nil || draftedTotal == 0 ? nil : Double(acceptedTotal) / Double(draftedTotal),
            specVerifySteps: spec == nil ? nil : verifyStepsTotal,
            specNormalSteps: spec == nil ? nil : normalStepsTotal,
            specGateDisabledSteps: spec == nil ? nil : gateDisabledTotal)
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
        print("usage: fastmlx-harness kl --model <PATH> --matrix-id <ID> --cell-id <ID> [--reference-model <PATH>] [--positions 24] [--corpus <FILE>] [--long-context-sample-positions 128] [--promotion-evidence false] [--python <PY>] [--script <REF.py>] [--evidence <FILE=harness-evidence.jsonl>]")
        exit(2)
    }
    do {
        let matrixID = try flags.strictString("matrix-id", default: "")
        let cellID = try flags.strictString("cell-id", default: "")
        guard !matrixID.isEmpty, !cellID.isEmpty else {
            throw KVFrontierEvidenceError.invalidIdentifier("matrix-id/cell-id")
        }
        let promotionEvidence = try flags.strictBool(
            "promotion-evidence", default: false)
        let positions = try flags.strictInt("positions", default: 24)
        let longContextSampleSize = try flags.strictInt(
            "long-context-sample-positions", default: 128)
        guard positions > 0, longContextSampleSize > 0 else {
            throw KVFrontierEvidenceError.invalidMetric("positions")
        }
        let referenceModelPath = try flags.strictString(
            "reference-model", default: modelPath)
        // Candidate-side KV tier (Task 8): the CANDIDATE scores with the selected cache;
        // the reference NEVER sees kvQuant (referenceConfig strips it) — it is the baseline.
        let kvQuantTier = try requestedKVQuantTier(flags)
        let requestedKVQuantTier = kvQuantTier ?? "fp16"
        guard let requestedCacheKind = KVCacheKind(kvQuant: kvQuantTier) else {
            print("kl FAILED: unknown --kv-quant tier \(kvQuantTier ?? "fp16") (known: \(knownKVQuantTiers))")
            exit(2)
        }
        let candidateIdentity = try ProvenanceCLI.modelEvidenceIdentity(at: modelPath)
        let referenceIdentity = try ProvenanceCLI.modelEvidenceIdentity(
            at: referenceModelPath)
        let sameWeights = ProvenanceCLI.sameResolvedModelPath(
            modelPath, referenceModelPath) && candidateIdentity == referenceIdentity
        let corpus = try loadMeasurementCorpus(flags)
        let shortEntries = corpus.entries(tagged: .prose) + corpus.entries(tagged: .code)
        let longEntries = corpus.entries(tagged: .longContext)
        guard !shortEntries.isEmpty, !longEntries.isEmpty,
            shortEntries.count + longEntries.count == corpus.entries.count
        else { throw KVFrontierEvidenceError.missingCohortEvidence }
        let (driver, tokenizer, eos) = try await loadSwiftDriver(modelPath: modelPath)
        let reference = referenceDriver(flags, modelPath: modelPath, eos: eos)
        guard reference.modelPath == referenceModelPath else {
            throw KVFrontierEvidenceError.invalidPromotionProvenance("referenceModelPath")
        }
        let config = RunConfig(temperature: 0, maxTokens: positions, kvQuant: kvQuantTier)
        let referenceConfig = RunConfig.greedy(maxTokens: positions)

        print("candidate: Swift engine on \(modelPath) (kv_quant_tier=\(kvQuantTier ?? "fp16"))")
        print("reference: mlx-lm on \(reference.modelPath) (fp16 KV)")
        print("corpus: \(corpus.corpusId) (content hash \(corpus.contentHash), \(corpus.entries.count) entries)")
        if sameWeights, requestedKVQuantTier == "fp16" {
            print("# SAME weights + fp16 KV both sides -> pipeline/noise-floor proof.")
        } else if sameWeights {
            print("# SAME weights; candidate \(requestedKVQuantTier) KV vs fp16 KV reference -> marginal KV-cache loss measurement.")
        } else {
            print("# DIFFERENT weights -> confounded comparison evidence; never promotion-eligible as marginal KV-cache loss.")
        }
        print("# TEACHER-FORCED: both sides score the reference's greedy continuation, so every position is context-locked.")

        var allKLs: [Double] = []
        // One median per ENTRY, not per position — the headline. Pooling raw per-position KLs
        // across entries with wildly different position counts (three ~24-position prompts vs a
        // 128-sampled-position long-context entry) lets the larger entry's positions dominate the
        // pooled median even though it is exactly one measurement among four; per-entry medians
        // give every entry equal weight regardless of how many positions it was scored at.
        var entryMedians: [Double] = []
        var candNLLTotal = 0.0, refNLLTotal = 0.0, totalPositions = 0
        var top1Matches = 0, top1ScoredPositions = 0
        var shortScoredPositionCount = 0, longContextScoredPositionCount = 0
        var shortEntryScoring: [KVEntryScoringEvidence] = []
        var longContextEntryScoring: [KVEntryScoringEvidence] = []
        var longContextMaxDocumentTokens = 0
        var longContextMaxScoredContextTokens = 0
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
            let top1 = try teacherForcedTop1Agreement(
                candidate: s.candidateRows, reference: s.referenceRows)
            allKLs.append(contentsOf: kls)
            entryMedians.append(medianOf(kls))
            top1Matches += top1.matches
            top1ScoredPositions += top1.scoredPositions
            // Perplexity pools NLL over positions from the SAME rows (identical math to
            // teacherForcedPerplexities — one forward pass serves both metrics).
            let n = Double(s.continuation.count)
            candNLLTotal += meanNLL(rows: s.candidateRows, tokens: s.continuation) * n
            refNLLTotal += meanNLL(rows: s.referenceRows, tokens: s.continuation) * n
            totalPositions += s.continuation.count
            shortScoredPositionCount += s.continuation.count
            shortEntryScoring.append(KVEntryScoringEvidence(
                entryID: entry.id, scoredPositions: s.continuation.count))
            print("entry \(entry.id) (\(entry.tag.rawValue)): forced-positions=\(kls.count), teacher-forced-top1-vs-reference=\(top1.matches)/\(top1.scoredPositions), median KL=\(sci(medianOf(kls)))")
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
            guard let deepestPosition = s.positions.last else {
                throw KVFrontierEvidenceError.missingLongContextDepthEvidence
            }
            longContextMaxDocumentTokens = max(
                longContextMaxDocumentTokens, docTokens.count)
            longContextMaxScoredContextTokens = max(
                longContextMaxScoredContextTokens,
                prompt.count + deepestPosition)
            let kls = perPositionKLs(reference: s.referenceRows, candidate: s.candidateRows)
            let top1 = try teacherForcedTop1Agreement(
                candidate: s.candidateRows, reference: s.referenceRows)
            allKLs.append(contentsOf: kls)
            longEntryKLs.append(kls)
            top1Matches += top1.matches
            top1ScoredPositions += top1.scoredPositions
            let n = Double(s.forcedTokens.count)
            candNLLTotal += meanNLL(rows: s.candidateRows, tokens: s.forcedTokens) * n
            refNLLTotal += meanNLL(rows: s.referenceRows, tokens: s.forcedTokens) * n
            totalPositions += s.forcedTokens.count
            longContextScoredPositionCount += s.forcedTokens.count
            longContextEntryScoring.append(KVEntryScoringEvidence(
                entryID: entry.id, scoredPositions: s.forcedTokens.count))
            // The long-context ENTRY headline is the TAIL (p95), not the median: over natural
            // long text the median sits below the same-weights noise floor (easy tokens both
            // quants agree on dominate it), while KV-quant loss accrues in the tail.
            print("entry \(entry.id) (long-context, \(docTokens.count) doc tokens): sampled-positions=\(kls.count)/\(continuation.count), teacher-forced-top1-vs-reference=\(top1.matches)/\(top1.scoredPositions), p95 KL=\(sci(quantile(kls, 0.95))), median KL=\(sci(medianOf(kls)))")
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
        guard top1ScoredPositions == totalPositions, top1ScoredPositions > 0 else {
            throw KVFrontierEvidenceError.invalidMetric("teacherForcedTop1Positions")
        }
        let top1Rate = Double(top1Matches) / Double(top1ScoredPositions)
        print("teacher_forced_top1_vs_reference: \(top1Matches)/\(top1ScoredPositions) (\(fmt(top1Rate * 100, 2))%)")

        let referenceVersions = await reference.versionSink.versions
        let (provenance, _) = ProvenanceCLI.build(
            modelPath: modelPath, referenceVersions: referenceVersions,
            corpus: corpus,
            modelCheckpointManifestHash: candidateIdentity.checkpointManifestHash)
        let candidateFormat: KVFormatGeometryEvidence?
        let storage: KVStorageEvidence?
        let actualControlBytes: Int?
        switch requestedCacheKind {
        case .affine(let tier):
            guard let telemetry = await driver.affineScoringTelemetry() else {
                throw SwiftEngineDriverError.unsupportedConfig(
                    "affine KL run completed without affine allocation telemetry")
            }
            guard telemetry.tier == tier else {
                throw SwiftEngineDriverError.unsupportedConfig(
                    "affine telemetry tier \(telemetry.tier.rawValue) != requested \(tier.rawValue)")
            }
            let format = KVFormatGeometryEvidence(
                kind: .affine, tier: tier.rawValue,
                keyBits: tier.keyBits, valueBits: tier.valueBits,
                groupSize: tier.groupSize, sinkTokens: 0,
                layerCount: telemetry.layerCount,
                kvHeadCount: telemetry.kvHeadCount,
                headDimension: telemetry.headDimension,
                capacityTokens: telemetry.capacityTokens,
                sequences: telemetry.sequences,
                metadataScalarBytes: telemetry.metadataScalarBytes,
                recordAlignment: 1)
            let (evidenceTotalBytes, overflow) = telemetry.dataArrayBytes
                .addingReportingOverflow(telemetry.materializationWorkspaceBytes)
            guard !overflow else {
                throw KVFrontierEvidenceError.storageArithmeticOverflow
            }
            let actual = KVStorageBreakdownEvidence(
                payloadBytes: telemetry.payloadBytes,
                metadataBytes: telemetry.metadataBytes,
                alignmentPaddingBytes: 0,
                fp16SinkBytes: 0,
                fp16TailBytes: 0,
                workspaceBytes: telemetry.materializationWorkspaceBytes,
                totalBytes: evidenceTotalBytes)
            candidateFormat = format
            storage = try format.storageEvidence(actual: actual)
            actualControlBytes = telemetry.controlBytes
            print(
                "# affine storage: payload=\(telemetry.payloadBytes), "
                    + "metadata=\(telemetry.metadataBytes), control=\(telemetry.controlBytes), "
                    + "persistent_total=\(telemetry.totalPersistentBytes), "
                    + "materialization_workspace=\(telemetry.materializationWorkspaceBytes), "
                    + "evidence_total=\(evidenceTotalBytes), "
                    + "capacity=\(telemetry.capacityTokens), layers=\(telemetry.layerCount), "
                    + "kv_heads=\(telemetry.kvHeadCount), head_dim=\(telemetry.headDimension)")
        case .fp16, .turboQuant, .kvarn:
            // These rows remain exploratory until their formats expose the same complete
            // runtime allocation contract. Promotion continues to fail closed below.
            candidateFormat = nil
            storage = nil
            actualControlBytes = nil
        }

        let frontier = KVFrontierEvidence(
            schemaVersion: 1, matrixID: matrixID, cellID: cellID,
            sameWeights: sameWeights,
            comparisonBaseline: sameWeights
                ? .sameWeightsFP16KV : .differentWeightsFP16KV,
            referenceKVQuantTier: "fp16",
            candidateModel: candidateIdentity,
            referenceModel: referenceIdentity,
            candidateFormat: candidateFormat, storage: storage,
            actualControlBytes: actualControlBytes)
        let payload = KLPayload(
            kvQuantTier: requestedKVQuantTier,
            klMedianNats: headlineMedian, klLongContextTailP95Nats: longContextTail,
            klPooledMedianNats: medianOf(allKLs), klPooledP95Nats: pooledP95,
            pplCandidate: pplPair.candidate, pplReference: pplPair.reference, pplDeltaPct: pplPair.relativeDelta * 100,
            totalPositions: totalPositions, entryCount: corpus.entries.count,
            teacherForcedTop1AgreementCount: top1Matches,
            teacherForcedTop1ScoredPositions: top1ScoredPositions,
            teacherForcedTop1AgreementRate: top1Rate,
            frontier: frontier,
            shortEntryCount: shortEntries.count,
            shortScoredPositions: shortScoredPositionCount,
            longContextEntryCount: longEntries.count,
            longContextScoredPositions: longContextScoredPositionCount,
            shortEntryScoring: shortEntryScoring,
            longContextEntryScoring: longContextEntryScoring,
            longContextMaxDocumentTokens: longContextMaxDocumentTokens,
            longContextMaxScoredContextTokens: longContextMaxScoredContextTokens)
        let record = ResultRecord(
            subcommand: "kl", provenance: provenance, payload: payload)
        let outputPath = evidencePath(flags)
        try RequiredKLEvidenceWriter.append(
            record, to: URL(fileURLWithPath: outputPath),
            promotion: promotionEvidence)
        print("# provenance: appended validated KL evidence to \(outputPath)")
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
        case "service-bench": await runServiceBench(flags)
        case "service-cancel-bench": await runServiceCancellationBench(flags)
        case "service-state-poison-bench": await runServiceStatePoisonBench(flags)
        case "service-soak": await runServiceSoakBench(flags)
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
                                               triad; affine-k4v2-g64/g128, affine-k8v2-g64/g128,
                                               affine-k4v4-g128, and tq2.5/tq3.5 select a lossy triad
                                               (non-crash + non-NaN + canary + engagement), and
                                               REPORT teacher-forced top-1 agreement vs fp16 KV
                 [--spec pld]                 spec-decode exactness triad instead: PLD-on vs
                 [--ngram 3] [--max-draft 8]   PLD-off on the SAME engine must be byte-identical
                 [--compiled-verify false]     at temp 0, with an engagement delta (drafting
                                               happened); acceptance rate is reported; fp16 KV only
          bench  --model <PATH>               stream-timed decode bench -> CSV (Release builds only)
                 [--kv-quant <TIER>]          KV tier for timed decode (\(kvQuantUsageTiers))
                 [--spec pld]                 time the speculative decode path (CSV mode=pld)
                 [--ngram 3] [--max-draft 8]   PLD match length / max drafted tokens K
                 [--compiled-verify false]     verify forward: fixed-K compiled step vs uncompiled
          service-bench --model <PATH>        aggregate service frontier (Release builds only)
                 --policy batch-no-spec|solo-pld  exact batch arm or serialized PLD policy
                 --scenario burst             simultaneous admission (initial measured scenario)
                 --concurrency 1|2|4|8         aggregate + per-request TTFT/TPOT/fairness
                 [--max-tokens 128] [--runs 3] [--prefill-chunk 16] [--max-prefill N]
          service-cancel-bench --model <PATH> disconnect SLA + slot-reuse gate (Release only)
                 [--runs 5] [--max-tokens 64] [--prefill-chunk 16]
                 [--keepalive-ms 1000] [--long-repeat 18]
          service-state-poison-bench --model <PATH> exact A/B/A recovery gate (Release only)
                 [--runs 3] [--concurrency 4] [--max-tokens 64]
                 [--prefill-chunk 16] [--keepalive-ms 1000]
          service-soak --model <PATH> --progress <FILE> resident mixed-workload soak
                 [--duration-seconds 86400] [--concurrency 4] [--max-tokens 64]
                 [--max-rss-drift-percent 5] [--responsiveness-ms 30000]
          kl     --model <PATH>               KLDivergenceMetric vs mlx-lm reference
                 --matrix-id <ID> --cell-id <ID> pin the frontier matrix/cell identity
                 [--kv-quant <TIER>]          CANDIDATE KV tier (\(kvQuantUsageTiers));
                                               reference always stays fp16 KV
                 [--reference-model <PATH>]   (defaults to --model: pipeline proof)
                 [--corpus <FILE=corpus/measurement-corpus-v2.json>]
                 [--long-context-sample-positions 128]
                 [--promotion-evidence false] require full storage + clean-SHA coherence gate

        common flags: --python <PY=~/harness-venv/bin/python> --script <scripts/harness_reference.py>
        """)
    }
}
