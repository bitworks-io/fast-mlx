# Carry-Forward Performance Backlog (from mlx-serve → fast-mlx)

- **Date:** 2026-07-08
- **Purpose:** Preserve the proven performance work from the (soon-to-be-sunset) `bitworks-io/mlx-serve` Zig engine as fast-mlx's **initial flywheel backlog** (§1 of the platform spec) and as direct input to the implementation plan. Every technique below was implemented and **measured** in the Zig engine; this is the "first batch" fast-mlx ports to Swift + re-quantifies through its own harness.
- **Provenance:** distilled from `~/mlx-serve-dist/perf-case-study.html` (the polished article, not yet published), `mlx-serve/BenchmarkLog.md` (703-line engineering log, 2026-05 → 2026-07), `docs/dspark-*.md`, `docs/superpowers/**` (EAGLE-3), `CHANGELOG.md`. **These source artifacts should be copied into fast-mlx before mlx-serve is deleted — see "Preservation" at the end.**
- **How to use this:** each entry is a candidate for the flywheel. Porting to Swift does **not** inherit the measured numbers — every one must re-pass the equivalence/engagement/acceptance triad and be re-measured on fast-mlx hardware before it earns a dial tier. The value here is *what to try, in what order, what impact to expect, and what already failed.*

---

## The governing insight: optimizations are model-class-specific

The single most important carry-forward lesson — **a lever's payoff depends on the model class, and the wins scale inversely with model size:**

- **Sampler/vocab-bound levers** (L1, L3, L1b, L3b, `sampleTokenLazy`) target **fast / small / MoE / draft** models — fixed vocab-bound absolute cost is a large fraction of a fast 2B decode, a tiny fraction of a bandwidth-bound 27B+.
- **Attention/bandwidth levers** (L2 fused quant-KV) target **large / long-context** models and grow with context — but are dwarfed by weight-read cost at batch=1.
- **MoE-specific** paths (prefill chunking, GDN fusion, batchability gating) behave differently from dense (memory-bound expert streaming vs. compute).

fast-mlx's dial should therefore select *different* optimization stacks per model class — and the harness must measure each on the model class it targets (toy-green/real-red rule).

---

## Shipped wins (proven in Zig; port + re-measure)

### Sampler-fusion stack (small/fast/MoE/draft models)
| ID | Technique | What it does | Measured impact (Zig) | Caveats |
|---|---|---|---|---|
| **L1** | Batched spec-verify accept walk | Run the sampler chain **once** over `[1,1+m,V]` verify logits, gather m draft probs via one `take_along_axis`, walk acceptance on CPU (marginal-preserving, exact RNG order) | gemma-4-e2b 4bit / M5: temp0.7 PLD echo **+11.8%**; temp0 greedy **+7.1% (free)** | Exposed a real bug: `mlx_topk` (no axis) flattens the whole array → silent shared-mask corruption on `[1,S,V]`. Use `mlx_topk_axis`. |
| **L3** | Top-p fast path when top-k active | Fuse top-p into the k values top-k already produced; skip the full 262K-vocab sort | sampled decode **+5.0%**; stochastic PLD echo **+10.8%** | **Invisible if `top_p=1.0`** (disabled). Any bench defaulting to top_p=1.0 reports zero gain. Distribution-preserving. |
| **L1b** | Extend batching to default-on drafter/MTP | `nextDrafter`/`nextMtp` still ran the per-draft loop; wire them onto L1's batched sampler | drafter echo **+2.7%** (scales w/ block size) | Same `mlx_topk`→`mlx_topk_axis` fix (red-on-revert). |
| **L3b** | Fuse `probsAllPos` top-p | L3's fusion into L1's batched sampler (was still one full-vocab sort/round) | **Cumulative L1+L3+L1b+L3b = +27.2%** on stochastic PLD echo (the run's headline) | — |
| — | `sampleTokenLazy` fast path | Direct 3-D `argmax_axis` for greedy, skip a reshape | ~**1–3%** small models | — |

*Scaling check (real, not toy): on Qwen3.6-27B the same stack gave only sampled +1.4% / PLD echo +3.5% — confirms these are small/fast-model levers.*

### Attention / bandwidth (large / long-context models)
| ID | Technique | What it does | Measured impact | Caveats |
|---|---|---|---|---|
| **L2** | Fused quant-KV attention | Read K/V **quantized** in attention instead of dequantizing the whole cache per token; GQA head-expansion via view-only Q-reshape (NOT `mlx_repeat_axis`, which materializes and re-inflates 4-bit K to bf16) | Qwen3-8B-8bit: 15K ctx flat; **32K ctx +3.1%** | **A roofline model predicted 15–40%; real gain +1–3%** because batch=1 decode is weight-read-dominated (8GB weights ≫ KV traffic). Grows with context; GQA-only (Gemma-4 MoE attention has no fused gate). *Lesson: distrust roofline predictions vs. measured; the harness is the arbiter.* |

### MoE / hybrid-specific
| Technique | What it does | Measured impact | Lesson |
|---|---|---|---|
| **GDN (GatedDeltaNet) decay-gate fusion** | Qwen 3.6 hybrid decay gate as one compiled kernel + cached norm/scale scalars (was ~10 dispatches/layer/token) | Qwen3.6-27B hybrid 28.74→29.12 (closed half the gap to mlx-lm); 35B-A3B MoE +3.9% over mlx-lm | Fusion + constant-caching on hot hybrid paths |
| **MoE cold-prefill de-chunking** | Coarsen SSM-checkpoint chunk stride to `max(base, PREFILL_CHUNK)` **for MoE only** (a stride tuned on a 4B model forced a boundary every 256 tokens on every hybrid, re-streaming all expert weights: constant **~+170ms** regardless of size — the diagnostic tell) | Recovered #1 vs LM Studio/oMLX on MoE prefill | **A default tuned on one model size silently regresses another size class** — guard with per-size-class measurement |
| **Continuous batching** (pre-existing) | `--max-concurrent N` batches batchable dense archs | ~**2.8×** aggregate 1→8 streams (Qwen3-8B: 61→174 tok/s) | MoE/hybrid-SSM **intentionally not batchable** (`modelBatchable` gate) — concurrent slots serialize, per-request latency ~2× at conc=2 |

### Speculative-decode self-management
| Technique | What it does | Measured impact | Note |
|---|---|---|---|
| **PLD yield-gate + mid-request re-enable** | 3 parts: yield gate (<0.25 accepted/step over 32 → fall back to pipelined path), mid-request re-enable (`tailMatchFraction` every 32 disabled steps), and a scheduler fix that had permanently pinned PLD off once disabled | novel-preamble-then-echo (realistic agent shape) 178→**222.5 tok/s** | This is the "speculative decoding manages itself mid-request" behavior. Rare-n-gram-match workloads previously paid PLD's cold un-pipelined forward every no-match step (−14% on creative content). |

### System / deployment
| Technique | What it does | Note |
|---|---|---|
| **GPU wired-memory ceiling raise** | `sysctl iogpu.wired_limit_mb` (e.g. ~96GB→115GB on 128GB) via persistent boot LaunchDaemon | Doesn't change decode speed (bandwidth-bound) but is the **prerequisite** to load 70B / DeepSeek-V4-Flash (284B@2-bit, ~81GB) with headroom. Critical for the 256/512GB big-memory targeting. |

### Request-start / cross-request reuse (omitted from the first distillation; restored 2026-07-12)

These were shipped and measured in the optimized Zig incumbent but were accidentally absent
from the initial carry-forward table. They are **not implemented in fast-mlx** today.

| Technique | What it does | Measured incumbent evidence | Swift-port guardrails |
|---|---|---|---|
| **Exact prefix/session cache** | Snapshot actor-owned KV plus recurrent/model state, find the longest reusable token prefix, and continue from it across agent turns | Reusing the prior assistant turn raised the second-turn hit from about **15% to 97%**; interleaved conversation roots were retained under entry/byte budgets | Composite semantic key + positive success-only commit; true retained-byte accounting; explicit hybrid/SSM checkpoints; A/B/A poison test |
| **Template/tokenize cache** | Reuse rendered chat-template/tokenization work on repeated long conversations | A measured 1,813-token warm request dropped **271→35 ms (7.7×)** before the first token | Record template/tool/tokenizer/revision in the key; report separately from model-prefix reuse |
| **Eager model warmup** | Page-fault weights and compile decode/prefill shapes immediately after load | First-request wall on Gemma 4 E4B 4-bit dropped **1097→307 ms (3.57×)** in the incumbent release measurement | Gated for tight-memory systems; cold and warm benchmarks must be separate; set explicit MLX cache policy |
| **Cold SSD prefix tier** | Spill exact completed prefix snapshots below the hot RAM LRU | Operationally shipped as a revisited-prompt latency tier; not a mid-generation context extender | Phase 2 only, after hot-cache correctness; encrypt/protect prompt-bearing state and bound disk bytes |

Source: preserved [`mlx-serve-CHANGELOG.md`](mlx-serve-archive/mlx-serve-CHANGELOG.md) and
[`mlx-serve-CLAUDE.md`](mlx-serve-archive/mlx-serve-CLAUDE.md). As with every carry-forward
number, these are priors to reproduce—not results inherited by the Swift engine.

---

## Big investigations (deep, honest, carry the lessons)

### DSpark — EAGLE-3-style external speculative drafter (the deepest single investigation)
- Full port (checkpoint `dspark_qwen3_8b_block7`): loader (64 tensors), non-causal 7-token-block backbone, multi-layer tap `[1,9,17,25,33]`, rank-256 Markov head, persistent monotonic draft KV cache, propose/verify/accept/rollback/commit, dispatch priority `dspark > mtp > drafter > pld`. **Byte-identical output on/off, verified.**
- **Acceptance "11.4%" was a metering error** (divided by full 7-block instead of effective length); true ~**1.5–1.8 tokens/round**, P(accept≥1)≈66–67% — a *healthy* drafter.
- **Real blocker (batch=1, Qwen3-8B):** target verify forward dominates (~37ms vs ~16ms baseline; 2.3×) → economics need acceptance **>~2.3–2.35** tok/round to win; measured 1.5–1.8 **loses** (net 41–51 vs 62.5 tok/s baseline).
- **Shipped sub-lever — re-forward elimination:** partial-accept was doing a **second full 8B forward on ~97.5% of rounds**; replaced with `KVCache.truncate` (byte-identical). **+24%** on the DSpark path (41→51), but **off by default** (still net-below baseline on this pairing — runtime gate correctly disables it).
- **Rejected by data (not assumed):** bf16-target does NOT raise acceptance (worse: 1.34 vs 1.60, and halves throughput); confidence-gate (can't shrink fixed propose cost); ratio-accept (measured ~break-even via a zero-behavior counterfactual probe *before* building).
- **Where it wins (not measurable at 8B here):** slower/bigger target (27B+/frontier) where the fixed drafter tax is dwarfed — published EAGLE-3 for Qwen3-32B reports ~1.7–1.9× on vLLM/SGLang. **Citation-hygiene catch:** an earlier "mlx-dspark Qwen3-8B ~1.6×" cite was a misattribution (real: Qwen3-4B/M4 Pro). *Nobody has shown a DSpark win at 8B on Apple Silicon.*
- **fast-mlx implication:** DSpark/EAGLE-3 is a **big-dense-model** lever; target it at 27B+/frontier, gate it economically per model, and never trust an acceptance number without checking the denominator.

### EAGLE-3 drafter (in progress in mlx-serve — carry the design)
- `docs/superpowers/specs/2026-07-07-eagle3-speculative-decoding-design.md` + a phase-0 spike plan, targeting `RedHatAI/Qwen3-32B-speculator.eagle3`. This is the concrete next step the article's roadmap names — the big-dense-model speculative win DSpark couldn't deliver at 8B. **Preserve these two docs.**

---

## Negative results — RETAINED so fast-mlx never re-attempts them blind

*(This is itself a carry-forward principle: the flywheel logs dated negative results with the same rigor as wins.)*

| Rejected/reverted | Why | Lesson |
|---|---|---|
| **L5 — batch-over-spec scheduling** (suspend PLD when ≥2 slots active to batch instead) | **Rejected before any code:** PLD-on serialized (514 tok/s) *beats* batched-no-PLD (274@c2, 403@c4) on echo/agentic; PLD's ~2.6× win dominates | Measure the proposal against real workloads before building |
| **Lazy-pipeline gate** (disable async submit-ahead above 25ms/step) | Fired correctly on Qwen27B but **hurt it −4.3%** — overlap helps even when GPU is saturated | Keep the submit-first async pipeline unconditionally |
| **Raw-lever audit (5 knobs)** | llama.cpp FA AUTO already ≈ ENABLED (+ safe fallback — don't hardcode); `n_ubatch` 512→2048 is a metric artifact; `n_threads` no effect (GPU-bound); MLX sliding-window KV already matches mlx-lm; all 16 attn sites already fused SDPA | These knobs are already right — don't re-litigate |
| **Resumable/interleaved prefill** | Built + byte-identical, but **46ms→46ms** (prefill runs ~265K tok/s; a 14K prompt loads <50ms, below decode jitter). Kept (useful on longer ctx / slower HW) but **not shipped as a speedup** | Don't claim a win a measurement doesn't show |
| **bf16-precision DSpark target** | Worse acceptance + half throughput | (see DSpark) |

---

## Methodology lessons (bake into the fast-mlx harness from day one)

- **Warm-vs-warm benchmarking:** adding an 8-token warmup curl reversed an apparent "31B regression" into +11.9%; 5/6 archs cleared +5% from the methodology fix **alone, no code change**. Cold-start effects masquerade as multi-percent regressions *in either direction*. (→ §6.3 warmup-drop rule.)
- **Verify the checkpoint's actual bit-width:** the one case MLX looked slower than GGUF was a "4-bit"-labeled Gemma-4-12B that actually shipped **mixed precision** (144 MLP projections at 8-bit, 10.5GB vs GGUF's uniform 6.5GB); a true uniform 4-bit re-quant got **+17%** at equal footprint. Don't attribute a checkpoint difference to the engine.
- **Report every number at its measured size, including below-prediction ones** (the article's stated ethic — L2's +1–3% vs the predicted 15–40% is *in* the article). This is the precision-loss product's credibility, applied to speed.
- **Silent-bug tells the harness caught:** an all-NaN causal mask (`0 × -inf = NaN`) invisible while the path was dead (the harness had been discarding NaNs); a "correctness test" that ran on an arch never reaching the code path (passed trivially). → real-scale, engagement-gated tests (§6.1/§6.6).

---

## Preservation status (act before mlx-serve is dropped)

**At risk when the mlx-serve GitHub repo (and/or local clone) is removed — recommend copying into fast-mlx `docs/reference/`:**
- `~/mlx-serve-dist/perf-case-study.html` — the polished article (local disk, not git; not yet published). The narrative artifact of the migrating work.
- `mlx-serve/BenchmarkLog.md` (703 lines) — the full engineering log this backlog was distilled from (more history + file:line detail).
- `mlx-serve/CLAUDE.md` — ~40 documented pitfall classes (the richest lessons-learned artifact; much already mined into spec §6/§7).
- `mlx-serve/docs/dspark-port-spec.md` (1,370 lines) + `docs/dspark-acceptance-investigation.md` (233 lines) — the DSpark technical spec + root-cause narrative.
- `mlx-serve/docs/superpowers/specs/2026-07-07-eagle3-speculative-decoding-design.md` + the phase-0 spike plan — the in-progress big-dense-model spec-decode work.
- `mlx-serve/docs/big-model-scale-out-plan.md` + `~/mlx-serve-dist/BIG_MODELS_RESEARCH.md` — frontier/512GB scale-out (DeepSeek-V4, GLM-5.2, Kimi-K2.7).
- `mlx-serve/CHANGELOG.md` — narrative version history with additional quoted speedups.
- Bench harness: `~/mlx-serve-dist/bench_matrix.py` + the July CSVs — the measurement methodology to port into fast-mlx's harness.

**Already safe (different repos):** the model catalog (`bitworks.io-website/concierge/docs/scaling-model-options.md`) and its sibling `concierge/docs/llm-strategy.md` / `infra/mlx-brain.md`.
