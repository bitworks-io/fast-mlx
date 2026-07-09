# Harness Hardening Implementation Plan (BLOCKING items)

> **For agentic workers:** implement via superpowers:subagent-driven-development. Steps use `- [ ]`. The detailed per-item spec (fixes + file:line) is the audit: [`docs/superpowers/verdicts/2026-07-09-harness-coverage-audit.md`](../verdicts/2026-07-09-harness-coverage-audit.md) — read it first.

**Goal:** close the 6 BLOCKING coverage/repeatability gaps so **Google TurboQuant (and any close-quant / low-bit comparison) can be honestly and reproducibly quantified.** The crux is teacher-forced KL — without it, the metric compares diverged contexts and collapses at 2-bit.

**Scope:** BLOCKING items 1–6 from the audit. The KV-quant *kernel* itself is TurboQuant engine work (out of scope here); item 4 here is only the harness-side readiness (lossy-tier gate + tier recording). FAST-FOLLOW/DEFER items are separate plans.

**Tech stack / workflow:** same as the harness spine — Swift 6 in `spike/`, pure `HarnessCore` tests off-box (`swift test --filter HarnessCoreTests`), engine parts on `llmbench@192.168.1.252` (rsync + `xcodebuild -skipPackagePluginValidation`; model `~/perf-work/models/...`; Python from `~/harness-venv`). Branch `feat/harness-hardening` off `main`. TDD, honest numbers, Swift 6 clean (zero unsafe hatches), commit per task.

---

## Chunk A — the crux: teacher-forced KL + perplexity (highest leverage)

### Task 1: Teacher-forced logprobs contract + `KLDivergenceMetric` redefinition
**Files:** `spike/Sources/HarnessCore/EngineDriver.swift`, `QualityMetric.swift`; `spike/Sources/fastmlx-harness/{SwiftEngineDriver,ReferenceDriver}.swift`, `spike/scripts/harness_reference.py`; tests in `HarnessCoreTests`.

- [x] **Contract (pure):** add a teacher-forced variant to `EngineDriver` — `logprobs(prompt:forcedContinuation:config:)` returns full-vocab logits at each of the N *forced* positions (index==token id), i.e. feed `forcedContinuation[i]` as the next token rather than argmax, so BOTH sides score the SAME contexts. Keep the free-running `logprobs` too (used elsewhere) or deprecate it — but the KL metric must use the forced one.
- [x] **Metric (pure) — TDD:** redefine `KLDivergenceMetric.measure` to (a) get the reference's greedy continuation for each prompt, (b) score both driver and reference teacher-forced on that continuation, (c) compute per-position KL over ALL N positions (no divergence starvation). Test with `ScriptedDriver`s that a known aligned KL is computed over all positions (the current CLI-only aligned computation becomes the metric's definition).
- [x] **Drivers (engine, on llmbench):** implement forced continuation in `SwiftEngineDriver.logprobs` (~5-line loop change: use forced token, not argmax) and `harness_reference.py --force-tokens`. Verify the ordering spot-check still holds. *(Spot-check re-confirmed: argmax id identical, sampled raw logits within |diff| ≤ 0.19 fp16 noise.)*
- [x] **Verify on llmbench:** re-run the INT4-vs-INT8 Qwen3-32B comparison — now over ALL positions, not 8/72. Expect a STABLE median (the 33× all-vs-aligned distortion gone). Record it. *(All 72 positions: median 1.192e-01 nats, p95 1.223; repeat run byte-identical. Same-weights noise floor 1.314e-03 nats. Old free-running: 14.2 all / 0.428 on 8 aligned.)*
- [x] Commit. *(8e253d4)*

### Task 2: Perplexity-delta metric
**Files:** `QualityMetric.swift` + test.
- [x] TDD: `PerplexityMetric: QualityMetric` — mean NLL (in nats or bits) of the forced tokens, from the same teacher-forced forward pass as Task 1 (nearly free). Delta = candidate ppl vs reference ppl. Test the pure math with `ScriptedDriver` logits. Wire into the CLI `kl`/a `quality` subcommand. Commit. *(5d30a63 — measured: same-weights delta −0.24% (noise, inside the 1% gate); INT4-vs-INT8 +39.59%.)*

---

## Chunk B — trustworthy inputs + records (pure-heavy)

### Task 3: Fixed, versioned measurement corpus with a long-context entry
**Files:** new `spike/corpus/measurement-corpus-v1.json` (checked in), `HarnessCore` loader, CLI wiring.
- [ ] TDD (pure): a corpus loader with a stable `corpusId` + content hash; entries tagged (prose / code / long-context). Include at least one **≥4K-token** entry (KV-quant loss accrues with context — a 12-token KL for a KV tier is near-meaningless). Replace the 3 CLI-hardcoded prompts (`Harness.swift`). For the long-context KL path, **sample/stream positions** rather than materialize full-vocab rows for every position (memory: ~0.6MB/pos × 2 sides). Record `corpusId` in every result. Commit.

### Task 4: Lossy-tier triad variant + KV-quant tier recording (harness-side readiness)
**Files:** `Triad.swift` (+ test), `EngineDriver.swift`/CLI plumbing.
- [ ] TDD (pure): add a `TriadMode` — `exact` (current: prefix≥minPrefix) vs `lossy` (non-crash + non-NaN + short-prefix + a **coherence canary**: a fixed prompt whose temp-0 answer must contain a known substring). At 2-bit, `verify` uses `lossy` (today it just fails minPrefix=30). Add the `kvQuant` tier string to `RunConfig`/records so a tier is recorded even before the engine honors it (the kernel lands with TurboQuant; here we make the harness *ready to receive and record* it). Commit.

### Task 5: Provenance + append-only JSONL evidence
**Files:** `HarnessCore` `ResultRecord`/provenance struct, CLI emit; `harness_reference.py` version header; `SwiftEngineDriver` version/hw capture.
- [ ] TDD (pure): a `Provenance` struct — date, hardware (chip/RAM/OS), harness git SHA, mlx-swift version, reference mlx + mlx-lm versions, model config/weights hash (replacing the dirname-guessed quant, `Harness.swift:140`), `corpusId`, nonce. Every subcommand (`verify`/`bench`/`kl`) appends a JSONL record with full provenance (bench CSV keeps working but gains the columns spec §6.3 mandates). `harness_reference.py` emits its `mlx`/`mlx_lm` versions in the JSON header; the Swift side captures chip/RAM/OS + its mlx-swift version. Commit.

### Task 6: fp16 reference policy
**Files:** ops on llmbench + a note in the run report.
- [ ] Stage fp16/bf16 reference checkpoints for the target models on llmbench (the dial's KL contract is "vs fp16"; the first run used INT8 as a proxy). If a true fp16 reference is impractical for a given model, document the proxy + a stated correction, per comparison. Verify a teacher-forced KL run against a real fp16 reference for one model. Commit + update the first-run verdict.

---

## After this lands
The harness can honestly quantify **Google TurboQuant** (2-bit KV) on the measured KL + perplexity frontier at long context, with reproducible provenance. Then: implement the TurboQuant KV-quant kernel in the engine (its own plan) → run it through this hardened harness → promote to a dial tier or shelve with a dated negative result. FAST-FOLLOW items (spec-decode telemetry for DFlash, sweep runner, memory metrics, seeded sampling) follow as the queue demands.

**Content:** per the standing practice, write a `docs/content/` piece when Chunk A lands — "auditing the instrument that measures everything" (the free-running-KL flaw + teacher-forcing) is a strong, honest story.
