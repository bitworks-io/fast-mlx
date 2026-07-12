# Sol optimization-landscape audit — the complete fast-mlx performance backlog

- **Date:** 2026-07-12
- **Owner:** Sol / fast-mlx
- **Scope:** existing plans, verdicts, task inbox, performance intake, preserved Zig evidence,
  current Swift implementation, pinned MLX dependencies, and current primary research
- **Audit-time decision:** run the Qwen3-32B EAGLE-3/DSpark gate first; reorganize everything
  after it into exact decode, concurrent serving, fused long-context memory, and model/quant
  lanes

> **Post-audit execution update (2026-07-12):** the first gate ran. The authenticated
> Qwen3-32B EAGLE-3 head passed parity but failed greedy byte identity on both 4-bit and 8-bit
> targets, so it is [shelved RED](../superpowers/verdicts/2026-07-12-qwen3-32b-eagle3-preflight.md).
> No compatible 32B DSpark/DFlash/MTP control was runnable. The actionable queue now advances
> to continuous batching plus decode-first chunked prefill.

## Bottom line

The queue was directionally right, but it mixed three different states: features already in
fast-mlx, wins proven only in the retired Zig engine, and techniques merely available in an
upstream dependency. That made several "already-have" labels false for the Swift engine and
hid two of the incumbent's largest agent-facing wins: exact prefix/session reuse and eager
request-start warmup.

At audit time, the audit did **not** displace the committed Qwen3-32B EAGLE-3/DSpark cycle,
but source
review changed the reason. Its model card reports `acceptance_length` 2.15 on summarization,
2.29 on code, and 2.49 on math at `k=3`. The official evaluation code defines that metric as
`1 + accepted draft tokens per round`, so the comparable draft counts are about 1.15, 1.29,
and 1.49. The preserved ~2.3 Apple break-even came from a different Qwen3-8B/DSpark cost
profile and cannot be transferred to a 32B target. The checkpoint remains first because it
matches the committed production-size target and its **32B Apple economics are unknown**—not
because a published acceptance headline approaches the old threshold. DFlash now has strong,
directly relevant MLX evidence, so it joins the gate as a reference implementation and
same-target control where a checkpoint exists; it does not silently replace EAGLE-3.

The strongest new quantization lead is **KVarN K4V2-g128**: 4-bit keys, 2-bit values,
Hadamard rotation, and two-axis variance normalization. It directly targets the
autoregressive scale-error accumulation that uniform low-bit KV schemes miss. Its evidence
is new, NVIDIA/Triton-only, and author-reported, so the right action is a bounded Apple/MLX
spike—not a promotion claim. A fused compressed-domain attention path must precede any claim
that a stored KV format improves decode speed.

## User story and proof contract

**Story.** The fast-mlx owner needs one durable portfolio view that says what is shipped,
what is proven elsewhere, what should run next, and what should not consume another cycle,
so the flywheel spends bench time on the highest-value Apple-Silicon frontier rather than on
duplicates or stale claims.

The audit is accepted when:

1. every local plan, verdict, inbox item, intake candidate, and carry-forward optimization
   has a disposition;
2. each new recommendation either strengthens an existing item or has a bounded task seed;
3. rank considers impact, evidence, implementation cost, and Apple/MLX fit;
4. performance claims are linked to primary papers, official implementations, upstream
   documentation, or preserved local measurements and remain labeled as unverified until
   fast-mlx reproduces them;
5. historical negative results stay closed unless a named mechanism changes their economics.

**Happy path:** a primary source sharpens an existing cycle or creates one measurable new
gate. **Failure/edge paths:** duplicates are folded into their parent, weak or vendor-only
evidence stays in research-later, unsupported Apple kernels do not inherit CUDA speedups, and
lossy cache work cannot substitute free-running drift for teacher-forced KL/perplexity.

## Current-state truth table

| Capability | fast-mlx now | Proven in Zig / available upstream | Portfolio consequence |
|---|---|---|---|
| Compiled base decode | **Shipped**, at Zig parity | Upstream model loading retained | Closed; optimize above the base loop |
| PLD | **Promoted**, byte-identical at temperature 0 | Zig precedent | Closed except serving-policy wiring |
| Trained speculation | Not implemented; `dspark` is a harness placeholder | EAGLE-3/DSpark evidence, DFlash MLX ports, pinned Swift speculative machinery | EAGLE-3 is RED; DSpark/DFlash/native-MTP controls are blocked until a compatible product-size checkpoint exists |
| Continuous batching | Not implemented; one actor owns one decoder | Zig measured about 2.8× aggregate from 1→8 streams; current MLX-LM has `BatchGenerator` | Next service-throughput cycle |
| Sampled generation/fusion | Engine is greedy-only | Zig L1/L3/L1b/L3b reached +27.2% on stochastic PLD for a small/fast model | Separate from batching; sampler/RNG contract first |
| Prefix/session reuse | Not implemented; `resetForNewRun()` discards cache state and compiled caches cannot copy/restore | Zig measured a 15%→97% second-turn cache hit and 7.7× warm request-start improvement; MLX-LM has trie/LRU prompt caching | Restore as an explicit exact-cache cycle after scheduler ownership is designed |
| KV quantization | fp16 and shelved TurboQuant only | Zig affine 4/8-bit and fused quant-KV; current research includes KVarN/KVTuner | Build the ordinary/asymmetric baseline and fused read path before exotic promotion |
| Absorbed MLA | Not implemented for DeepSeek V3 | Current Python MLX ships it; pinned Swift has an analogous GLM latent-cache path | Existing task is planning-ready with two code oracles, not greenfield research |
| Request-start warmup | Not implemented as a product policy | Zig measured 1097→307 ms on Gemma 4 E4B; upstream caches/templates exist | Fold eager warmup + tokenize caching into exact request-start work |
| Runtime capacity control | Static capacity model/CLI shipped; no serving admission loop | Platform spec defines the gate | Retain as reliability-critical work |

The code facts behind the table are local and inspectable: `InferenceActor` is single-owner;
`KVCacheKind` has only fp16 and TurboQuant; `CompiledKVCache.copy()` and state restore trap;
and the harness accepts PLD but does not integrate a trained drafter. The pinned
`mlx-swift-lm` revision contains speculative-decoding and prompt-cache machinery, but that is
reference code—not a fast-mlx feature.

## Ranked execution queue

Scores are ordinal, 1 (weak/expensive) to 5 (strong/cheap). **Cost score is inverted:** 5 is
cheap or tightly bounded. External numbers are source-confirmed claims, not reproduced
fast-mlx results.

| Rank | Cycle | Impact | Evidence | Cost | Apple fit | Decision and hard gate |
|---:|---|---:|---:|---:|---:|---|
| 1 | **Qwen3-32B EAGLE-3/DSpark, with DFlash + native-MTP controls** | 5 | 4 | 2 | 5 | **Executed 2026-07-12:** EAGLE RED on byte identity; controls blocked by missing compatible product-size checkpoints. See the post-audit banner and dated verdict. |
| 2 | **Continuous batching + decode-first chunked prefill** | 5 | 5 | 2 | 5 | Port the scheduler design. Preserve drain-before-batch-join, cancellation, fairness, and architecture batchability. Measure aggregate throughput plus p95 TTFT/TPOT. |
| 3 | **KVarN K4V2 + asymmetric affine/KVTuner storage-quality gate** | 5 | 4 | 2 | 3 | New top KV-quality gate. First prove the tile transform and packed bytes/token; compare K4V2/K8V2 and per-layer schedules at equal effective bits. |
| 4 | **Fused compressed-domain KV attention for the winner** | 4 | 4 | 1 | 4 | Phase B of the same lane. Equal-output fp16 oracle first; require an end-to-end 32K/128K win, not a kernel-only headline. No codec earns a speed tier while materializing the full cache. |
| 5 | **Exact hot prefix/session cache + request-start latency stack** | 5 | 5 | 2 | 5 | Design over scheduler/cache ownership. Composite key, positive commit, byte/entry budget, hybrid checkpoints, eager warmup, template/tokenize cache. SSD is phase 2. |
| 6 | **Absorbed MLA for DeepSeek-class models** | 5 | 5 | 2 | 5 | Keep high; move to #2 if DeepSeek deployment becomes near-term. Near-float-noise teacher-forced logit/attention agreement before memory or speed claims. |
| 7 | **Sampled-generation foundation, then sampler fusion** | 3 | 5 | 3 | 5 | Split from batching. Define deterministic RNG ordering and distributions, then port L1/L3/L1b/L3b and measure the small/fast model class. |
| 8 | **Mixed/learned weight-quant frontier** | 4 | 4 | 4 | 5 | Bounded offline sweep: standard affine vs MLX dynamic/DWQ and oQ4e; the Swift loop should consume ordinary MLX checkpoints. Preserve calibration provenance. |
| 9 | **Runtime admission + measured prefill capacity** | 4 | 5 | 3 | 5 | Reliability/capacity work, not a decode multiplier. Enforce explicit `Memory.cacheLimit` whenever the wired ceiling is raised. |
| 10 | **TurboQuant Spike B closure** | 3 | 3 | 3 | 4 | Keep the second-failure stop rule. Broaden the finite sweep to outliers plus K-high/V-low and boundary-layer protection, but rank below fused attention/KVarN. |
| 11 | **PrismML Ternary/Bonsai device tier** | 3 | 3 | 3 | 4 | Real and benchmarkable, but it is a separately trained model family—not post-hoc Qwen compression. Test ternary stock-MLX first; binary stays conditional on fork review/upstreaming. |

Task seeds for ranks 1–8 live in `docs/task-inbox/`; ranks 9–11 retain their existing spec,
task, or intake owner. Queue order is a planning decision, not evidence that any external
technique has passed fast-mlx's promote gate.

## Four portfolio lanes

### 1. Exact decode multipliers

- **PLD — PROMOTED.** Exact and gate-tuned; only serving default wiring remains.
- **Qwen3-32B EAGLE-3/DSpark — EXECUTED / EAGLE RED, CONTROLS BLOCKED.** The compatible EAGLE
  checkpoint passed head parity and failed byte identity on 4-bit and 8-bit targets. DSpark,
  DFlash, and native MTP produced no product-size same-target verdict because no compatible
  checkpoint was runnable. Do not infer a negative result for those untested methods.
- **DFlash — CONTROL / CONDITIONAL PORT.** The paper and current Apple port are real. The
  port reports 2.78–3.06× on Qwen3.6-27B-4bit from 1K–16K output on an M5 Max, but those are
  maintainer results from one prompt. Qwen3.5-27B falls from 2.37× at 1K to 1.34× at 8K,
  proving model/length sensitivity. Use a same-target Qwen3-8B control where possible and a
  product-scale pair separately; never compare raw cross-model tok/s.
- **Native MTP — FOLD INTO THE TRAINED-SPEC GATE.** The pinned Swift dependency contains
  drafters/iterators, but fast-mlx does not. Treat it as a lower-port-cost control, not
  “already-have.”
- **DFlash small-M int4 verify QMM — REUSABLE SUB-SPIKE.** Even if DFlash loses, its
  verify-specialized Metal approach attacks DSpark's measured bottleneck. Benchmark current
  MLX small-M quantized matmul before importing custom kernels.
- **Apple ReDrafter / QuantSpec / Double / SpecVocab — RESEARCH-LATER.** Useful design
  references; checkpoint coverage, training, stochastic exactness, or Apple implementation
  evidence is weaker than the active gate.
- **Shapeless compilation — FOLD INTO PLD/BATCHING.** A cheap probe for variable proposal
  and batch shapes. Shape-dependent cache/mask logic can make it incorrect; API existence is
  not a speed result.

### 2. Concurrent serving and request-start latency

- **Continuous batching — EXECUTE.** Preserve the named solo→batch transition invariant.
  MLX-LM's current decode-first prompt/completion separation and Sarathi's token-budgeted
  chunking are references, while the 2.8× Zig result is the Apple prior.
- **Chunked prefill — FOLD INTO BATCHING.** It is a fairness, cancellation, and TTFT tool,
  not a claim that full-context KV allocation disappears.
- **Exact prefix/session cache — EXECUTE AFTER CACHE OWNERSHIP.** Trie/LRU nearest-prefix
  lookup, explicit conversation checkpoints, byte budgets, and positive success-only commit.
  Key every semantic axis: model/revision, tokenizer/template, tools, KV scheme, position/RoPE
  semantics, architecture state, and drafter state.
- **Eager warmup + template/tokenize cache — FOLD INTO REQUEST-START.** They explain part
  of the incumbent's large warm-TTFT lead and are exact, but need salted/cold benchmarks so
  they cannot contaminate steady-state claims.
- **Cold SSD prefix tier — PHASE 2.** It improves revisited-prompt TTFT, not active-context
  length. Hot cache correctness and memory accounting come first.
- **POD attention, PagedAttention, prefill/decode disaggregation — DEFER.** Profile a
  working single-Mac scheduler first. Disaggregation belongs to future multi-Mac scale-out.

### 3. Fused long-context memory

- **Ordinary affine K/V + fused quant attention — MISSING BASELINE.** fast-mlx should not
  compare a novel codec only to fp16 or call a nominal bit width its size. Record packed
  bytes/token including scales, biases, codebooks, and alignment.
- **KVarN K4V2-g128 — NEW TOP CANDIDATE.** It spends more precision on keys than values and
  normalizes variance across both token and channel axes. The paper establishes the method
  and 2/2-bit quality evidence on smaller models. A newer **official-repository author
  benchmark**, not the paper, reports Qwen3-32B AIME25 parity, roughly 4× KV capacity, and
  throughput above fp16 for a 16K-context burst at TP=2. Apple performance remains unverified.
- **KVTuner — FOLD INTO KVarN/AFFINE PHASE 2.** The Python research plane can select
  per-layer K8V4/K8V2/K4V2 schedules while Swift consumes a small configuration artifact.
  This is likely the cheapest unusual quantizer setup once per-layer cache metadata exists.
- **TurboQuant Spike B — BOUNDED CLOSURE.** Its task owns the omitted outlier recipe and current
  K-high/V-low/boundary evidence, then fully shelve on a second loss. Community MLX kernels
  are implementation leads, not proof.
- **Absorbed MLA — EXECUTE.** Python `mlx-lm` shipped DeepSeek V3 MLA in v0.30.6, and the
  pinned Swift dependency already uses the same latent-cache/absorbed-projection pattern for
  GLM4-MoE-Lite. The old “nobody has shipped this in MLX” statement is contradicted.
- **CommVQ, AQUA-KV/xKV, GEAR, eOptShrinkQ — RESEARCH-LATER.** Interesting codebook,
  cross-layer, low-rank, and residual methods; all add training or kernel complexity before
  the stronger KVarN/affine baseline exists.
- **EpiCache / KVzip / Quest / SnapKV / PyramidKV — WORKLOAD-SPECIFIC LATER.** These forget
  or select context rather than only changing numeric representation. They require agent and
  long-context task scores; teacher-forced KL alone cannot establish retained utility.

### 4. Model and weight frontiers

- **MLX dynamic quantization + DWQ / oQ4e — BOUNDED SWEEP.** Official MLX-LM supports
  sensitivity-based dynamic quantization, DWQ, AWQ, and GPTQ; dynamic can feed DWQ. oQ4e is a
  newer activation-energy/imatrix-weighted alternative whose output remains standard MLX
  safetensors. Compare all claims through fast-mlx rather than inheriting vendor evals.
- **PrismML Ternary/Bonsai — RESEARCHABLE DEVICE TIER.** Official models, demos, and MLX
  artifacts now exist. Ternary 2-bit runs on stock MLX; binary 1-bit currently depends on
  PrismML forks while upstream support is pending. Compare against same-footprint models and
  enforce the coherence/garbage floor.
- **SliceGPT — RESEARCH-LATER.** Dense dimension removal maps better to Apple GPUs than
  unstructured sparsity, but checkpoint conversion and architecture changes make it a second
  weight-frontier study after learned quantization.
- **TEAL / W4A4KV4 SpinQuant — DEFER.** Their wins rely on sparse or low-bit activation
  kernels absent from the current MLX/Metal stack.
- **XGrammar 2 — PRODUCT-ADJACENT INTAKE.** Apple/Metal support and exact structured-output
  masking matter for tools, but it should follow the serving/sampler surface and is not a
  decode multiplier by itself.

## Complete local portfolio disposition

### Completed / promoted

- Swift compiled decode loop and language/lineage gate.
- Harness spine and hardening: teacher-forced KL, perplexity, long-context tail-p95,
  provenance, and lossy-tier floor.
- System-aware capacity model, profiler, and CLI.
- PLD framework and the 2026-07-11 gate tuning.
- Preservation of the retired Zig performance evidence under `docs/reference/`.
- Content pieces for the shipped flywheel cycles.

Unchecked boxes in completed historical plans are stale execution markup, not new work.

### Completed / shelved / blocked

- Uniform TurboQuant v1: `tqB3` lost to affine 4-bit; `tqB2` was catastrophic.
- Free-running precision-loss comparisons from the pre-hardening harness: historical only.
- Qwen3-32B EAGLE-3: shelved on byte identity; DSpark/DFlash/native-MTP controls blocked by
  missing compatible product-size checkpoints and not planning-ready.

### Active or planning-ready

- Continuous batching, chunked prefill, and runtime admission.
- Sampled-generation foundation and sampler fusion.
- KVarN/asymmetric affine/KVTuner storage-quality gate.
- Fused compressed-domain KV attention for the selected format.
- Exact hot prefix/session cache, request warmup, and later cold SSD tier.
- Absorbed MLA.
- Learned/mixed weight quantization.
- TurboQuant Spike B closure.
- PrismML device tier.
- GDN/MoE tuning only after checking the pinned upstream implementation.
- Serving/API driver, conformance, task benchmark layer, soak/recovery, and memory controls
  already specified by the platform design.

### Do not reopen without a changed mechanism

- Batch-over-spec blanket scheduling.
- Disabling the submit-first pipeline when the GPU is busy.
- Hard-coding FlashAttention, changing `n_ubatch`, or retuning CPU thread count without a new
  profile; existing sliding-window/fused-SDPA behavior.
- Resumable/interleaved prefill as a present-day 14K speed claim (retain as infrastructure).
- bf16 DSpark target, confidence gating, or ratio acceptance without changed economics.
- Uniform TurboQuant v1.
- Generic “add FlashAttention” work: upstream MLX already owns optimized SDPA.

## Source-review ledger

The labels review whether a source supports a statement—not whether its benchmark has been
independently reproduced.

| Claim or prior statement | Review | Evidence / correction |
|---|---|---|
| EAGLE-3 model-card acceptance is directly comparable with the 8B Apple break-even | **CONTRADICTED** | The [model card](https://huggingface.co/RedHatAI/Qwen3-32B-speculator.eagle3) reports `acceptance_length`; [official evaluator code](https://github.com/vllm-project/speculators/blob/main/tests/e2e/run_vllm.py#L115) defines it as `1 + accepted drafts/round`. The ~2.3 threshold was pairing-specific to Qwen3-8B/DSpark. |
| DFlash has current Apple/MLX implementations and large reported gains | **CONFIRMED claim / UNVERIFIED result** | [paper](https://arxiv.org/abs/2602.06036), [official code](https://github.com/z-lab/dflash), [optimized MLX port](https://github.com/bstnxbt/dflash-mlx), [independent MLX port](https://github.com/Aryagm/dflash-mlx) |
| Continuous batching is a proven Apple multiplier | **CONFIRMED locally** | [carry-forward evidence](2026-07-08-carry-forward-performance-backlog.md); [current MLX-LM releases](https://github.com/ml-explore/mlx-lm/releases) corroborate implementation feasibility |
| Native MTP is “already-have” in fast-mlx | **CONTRADICTED** | The pinned Swift dependency contains machinery, but the fast-mlx actor/compiler/harness path does not route it |
| Paged/prefix/SSD KV caching is “already-have” in fast-mlx | **CONTRADICTED** | The current actor resets state and compiled caches cannot copy/restore; only the incumbent/upstream ecosystems have it |
| Nobody has shipped absorbed MLA in MLX | **CONTRADICTED** | [MLX-LM v0.30.6](https://github.com/ml-explore/mlx-lm/releases/tag/v0.30.6) shipped DSV3 MLA; the pinned Swift GLM4-MoE-Lite path is an implementation analogue |
| KVarN K4V2 reports fp16-level Qwen3-32B quality and about 4× capacity | **CONFIRMED repository claim / UNVERIFIED Apple result** | The [paper](https://arxiv.org/abs/2606.03458) covers the method and 2/2-bit smaller-model evidence; the [official repository](https://github.com/huawei-csl/KVarN) alone reports the later Qwen3-32B AIME25, 16K-burst, TP=2 K4V2 system result. |
| KVTuner supports per-layer mixed K/V precision | **CONFIRMED** | [paper](https://arxiv.org/abs/2502.04420), [official code](https://github.com/cmd2001/KVTuner); reported throughput varies by paper revision, so no single number is adopted here |
| TurboQuant has no MLX implementation | **CONTRADICTED** | [MLX-VLM](https://github.com/Blaizzy/mlx-vlm), [TurboQuant+](https://github.com/TheTom/turboquant_plus), and community ports exist; upstream/core status and quality remain unsettled |
| MLX offers learned/mixed weight-quant tools | **CONFIRMED** | [official MLX-LM learned-quant guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md) |
| PrismML is only a placeholder | **CONTRADICTED** | [official demo](https://github.com/PrismML-Eng/Bonsai-demo), [ternary MLX model](https://huggingface.co/prism-ml/Ternary-Bonsai-8B-mlx-2bit), [binary MLX model](https://huggingface.co/prism-ml/Bonsai-8B-mlx-1bit); performance is vendor-reported |
| Lossy selective-cache methods are globally safe defaults | **UNVERIFIED / rejected as stated** | [EpiCache](https://machinelearning.apple.com/research/epicache), [KVzip](https://github.com/snu-mllab/kvzip), and [Quest](https://arxiv.org/abs/2406.10774) are workload-sensitive and require task-level gates |

## Standing research cadence

- **Monthly light scan:** MLX/MLX-LM/MLX-Swift-LM releases, Apple research, leading inference
  runtimes, and new quant/speculation papers. Update only source status and intake.
- **Quarterly deep audit:** rerun this reconciliation and re-score the full matrix.
- **Event-driven:** immediately re-open a candidate when it gains an Apple/Metal
  implementation, a product-model checkpoint, or evidence that changes a recorded negative
  result's cost model.

Research never promotes a dial tier. Only a clean-SHA bench cycle through the hardened
harness may do that.
