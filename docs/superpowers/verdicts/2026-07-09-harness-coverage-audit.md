# Harness Coverage & Repeatability Audit

- **Date:** 2026-07-09
- **Verdict:** the harness spine is a well-built, honestly-instrumented **skeleton** — but **NOT yet ready to quantify the roadmap.** Of the dial's 8 axes it can fully exercise **none** beyond swap-the-checkpoint weight-quant; both declared axes (`kvQuant`, `specDecode`) are string fields that *throw*; two of three quality layers don't exist; results carry no provenance; the reference path is a hand-built Python venv on one private box.
- **The single most important gap:** the committed `KLDivergenceMetric` is **free-running** (each side follows its own greedy path), so it compares *diverged contexts* — it already distorted the first real dial point **33×** (all-positions 14.2 vs context-aligned 0.43 nats, on only **8/72** usable positions), and **collapses entirely in the low-bit regime Google TurboQuant targets.** Teacher-forced scoring fixes this and gives perplexity nearly for free.

## Coverage — dial axes

| Axis | Status | Gap |
|---|---|---|
| Weight-quant preset | **PARTIAL** | Only as "different `--model` dir"; quant label *guessed from dirname substring* (`Harness.swift:140`) — mislabeled checkpoint recorded wrong; no bits/size recorded → "quality at equal size" not expressible |
| KV-quant tier | **CANNOT** | `RunConfig.kvQuant` throws (`SwiftEngineDriver.swift:106`); only fp16 exists; no 2-bit/TurboQuant tier named; reference has no KV-quant either |
| Spec-decode mode | **CANNOT** | `specDecode` throws; `acceptanceRate` hardwired `nil`; `Mode` enum only `{none,pld,dspark}`. Checks exist + tested, but nothing produces data |
| Chunked-prefill size | **CANNOT** | `runBench` hardcodes `.decode`; no prefill workload runs; no chunk knob |
| Context regime (4K/64K/128K) | **CANNOT** | Whole live corpus = 3 short prompts (≤12 tok); KL over 24 positions. **KV-quant loss accrues with context → a 12-token KL for a KV tier is near-meaningless.** (Memory: full-vocab float32 rows ~0.6MB/pos × 2 sides → long-context needs sampled/streamed positions) |
| Concurrency / batching | **CANNOT** | Always 1; single-stream engine; no drain-before-join equivalence test |
| Model class (dense/MoE/hybrid-SSM) | **PARTIAL** | Any `--model` path works (MoE+dense proven); no sweep runner; single `--min-prefix` (no per-family table); hybrid-SSM untested |
| Model size | **PARTIAL** | Works via path; no sweep; no param-count/size recorded → can't plot "wins scale inversely with size" |

## Coverage — quality stack (spec §4)

- **Layer 1 KL** — **PARTIAL, the key flaw:** free-running; committed metric is the *unaligned* computation (`QualityMetric.swift:35`); context-aligned is CLI-only. Collapses at 2-bit (paths diverge after ~1 token). **Teacher-forced fix is small + well-bounded.**
- **Layer 2 Perplexity** — **ABSENT** (nearly free once teacher-forcing exists).
- **Layer 3 Task benchmarks** (MMLU/GSM8K/HumanEval/RULER) — **ABSENT** (no corpus/scorer/runner).
- **"Spec-decode is exact → speed-only"** — structure yes (triad), plumbing no; engagement counter is trivial (`"decode": n`, always engages); both drivers throw on temp≠0 → can only prove *greedy* exactness.

## Per-enhancement readiness
- **Google TurboQuant (2-bit KV): NOT quantifiable today.** Needs (in order): teacher-forced KL → KV-quant impl + tier plumb-through → long-context corpus → perplexity → lossy-tier triad variant (non-crash+non-NaN+short-prefix+coherence-canary; today `verify` minPrefix=30 just fails at 2-bit) → **memory metrics** (TurboQuant's whole value = capacity at equal quality, invisible to a tok/s+TTFT-only harness).
- **DFlash: NOT quantifiable.** All spec-decode plumbing + telemetry shape: `RunResult` has one `acceptanceRate:Double?` with undefined denominator — needs explicit `rounds`/`acceptedTokensTotal`/per-round length (the DSpark "check the denominator" lesson) for the ~2.3-tok/round breakeven.
- **oQ4e: PARTIAL, closest.** `--model` vs `--reference-model` KL works mechanically; needs size recording, a true fp16 reference (Run 2 used INT8 proxy), teacher-forcing, and a real corpus.
- **Backlog:** GDN/MoE fusion measurable **today** (temp=0 decode + triad). Sampler fusion **CANNOT** (wins live at temp>0/top-p; harness is greedy-only).

## Repeatability
| Criterion | Status | Fix |
|---|---|---|
| Fixed corpus | **WEAK** | Measurement prompts hardcoded in CLI source, unversioned → promote to a checked-in, hashed corpus; record corpus id in output |
| Determinism | **MOSTLY OK @ temp=0** | temp=0 enforced; but free-running aligned-position count is divergence-sensitive (a repeatability problem) — same teacher-forced fix; also `--nonce` for byte-replay |
| Version pinning | **NOT RECORDED** | Swift pinned (`Package.resolved`); Python env pinned nowhere; no result records versions → `requirements.lock` + emit mlx/mlx-lm/mlx-swift versions + harness SHA in every result |
| Env reproducibility | **MANUAL** | Hardware/model-hash captured only in verdict prose → auto-capture chip/RAM/OS + model config/weights hash (replacing dirname-guessed quant). Release-build guard IS enforced (`BenchMatrix.swift:40`) |
| Customer-runnable verify | **DOES NOT EXIST** | `verify`/`kl` require the mlx-lm venv as live reference → ship **sealed references** (recorded reference streams + KL-vs-recorded values in the signed catalog) so a customer runs the Swift binary alone |
| Durable evidence | **PARTIAL** | Bench CSV lacks hw/versions/date/SHA/corpus-id; verify/kl are stdout-only → append-only JSONL with full provenance per subcommand |

## Prioritized hardening list

**BLOCKING (before Google TurboQuant can be honestly quantified):**
1. **Teacher-forced logprobs** — forced-continuation variant in `EngineDriver`; ~5-line change in `SwiftEngineDriver.logprobs` + `--force-tokens` in `harness_reference.py`; **redefine `KLDivergenceMetric` to use it** (make context-locked the metric's definition, not CLI side-reporting). Highest leverage.
2. **Perplexity-delta metric** — trivial after (1); second `QualityMetric`.
3. **Fixed, versioned measurement corpus + long-context entry** (prose/code + ≥4K; position-sampling for memory); record corpus id+hash.
4. **KV-quant axis plumb-through + lossy-tier triad variant** (honor `kvQuant` once the engine has a quantized cache; tier names incl. 2-bit; non-crash+non-NaN+short-prefix+coherence-canary pass condition). *(The KV-quant kernel itself is TurboQuant engine work.)*
5. **Provenance + JSONL evidence** in every result: date, hw (chip/RAM/OS), harness SHA, mlx-swift + mlx/mlx-lm versions, model config/weights hash, corpus id, nonce.
6. **fp16 reference policy** — stage fp16/bf16 reference checkpoints (KL contract is "vs fp16"; Run 2 used INT8 proxy).

**FAST-FOLLOW (for DFlash etc.):** 7. spec-decode telemetry contract (rounds/acceptedTokens + Mode values). 8. pin the reference venv + mlx version parity. 9. sweep runner (catalog × axes). 10. prefill workload + chunk knob. 11. memory metrics. 12. seeded stochastic sampling (unblocks the whole sampler-fusion stack). 13. `--nonce` + echo salt.

**DEFER:** 14. task-benchmark layer. 15. concurrency/batching bench + drain-before-join test. 16. customer-runnable sealed-reference verify. 17. HTTP `EngineDriver`. 18. per-family minPrefix table + hybrid-SSM verification.

## Bottom line
Production-grade *thinking* (triad, logprobs contract, bench methodology, noise-floor validation, throw-on-unsupported honesty) on a skeleton that can't yet measure the roadmap. Close BLOCKING items 1–6 before any enhancement work; item 1 (teacher-forced KL) is the crux and lands first.
