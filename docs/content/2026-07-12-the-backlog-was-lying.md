# The backlog was lying: when “already have” meant “the old engine had it”

> **Whitepaper themes:** Rapid research integration — the flywheel; Building a
> high-performance MLX inference engine in Swift; The optimization dial — quantified
> precision-loss tuning

A performance backlog looks like a list of future work. In an inference engine, it is also a
model of reality: what the engine can do, which experiments have already failed, and which
unclaimed multipliers remain. If any of those states are wrong, prioritization becomes a
benchmark bug before a benchmark even runs.

Our backlog had drifted in a subtle way. fast-mlx is a new Swift/MLX engine replacing an
optimized Zig incumbent. The preserved notes correctly described years of Zig work, and the
pinned Swift dependency contained increasingly capable upstream code. But some intake labels
collapsed all three systems into “we.” Native multi-token prediction was marked
“already-have.” Paged/SSD prefix caching was also “already-have.” Neither existed in the
fast-mlx execution path. The Swift engine could run exact prompt-lookup decoding, but it was
otherwise a single greedy actor that reset its cache between runs.

That category error hid more than missing checkboxes. It hid product economics.

## Start with code, not the research feed

The audit began by reconciling every plan, dated verdict, task seed, intake candidate, and
preserved Zig result against current code. Four local facts mattered:

- PLD was genuinely shipped and exact.
- trained drafters were not wired, even though a placeholder and upstream machinery existed;
- KV storage offered fp16 and the shelved TurboQuant experiment, not an ordinary affine
  4/8-bit production ladder;
- compiled caches could not copy or restore state, and the actor deliberately discarded it
  before the next run.

The last point restored a missing high-impact lane. The old engine had measured a second-turn
prefix hit rising from roughly 15% to 97%, a 1,813-token warm request falling from 271 ms to
35 ms, and eager warmup cutting a first request from 1,097 ms to 307 ms. Those are not
micro-optimizations. They are the difference between an agent loop feeling continuous and
feeling like every tool call starts a new conversation. Yet the first carry-forward summary
had omitted them.

The correction is not “copy the old cache.” Hybrid linear-attention state can retain far more
memory than a simple KV-byte counter reports. A cache key must encode every semantic axis,
and entries should be committed only after clean success. The implementation task is now
explicit: exact hot state first, true byte accounting and poisoning tests, cold SSD later.

## New research should change a gate, not erase history

The existing top task was EAGLE-3-style speculative decoding on Qwen3-32B. DFlash then
arrived with a block-diffusion drafter and unusually relevant Apple evidence. One current MLX
port reports 2.78–3.06× on Qwen3.6-27B-4bit from 1K through 16K output on an M5 Max. It also
shows why headlines are insufficient: a Qwen3.5-27B pairing declines from 2.37× at 1K to
1.34× at 8K. Model and output length change the result materially.

It was tempting to replace the queue leader. The numbers did not justify that.

The target-compatible Qwen3-32B EAGLE checkpoint reports `acceptance_length` values of 2.15 on
summarization, 2.29 on code, and 2.49 on math. Our first reading put those beside the old
Apple break-even near 2.3 and called the result “too close to call.” Source review caught the
denominator error: the official evaluator defines `acceptance_length` as one plus accepted
draft tokens per round. The comparable draft counts are about 1.15, 1.29, and 1.49. Worse,
the 2.3 break-even belonged to a different Qwen3-8B/DSpark cost profile.

That correction does not prove EAGLE loses at 32B—the larger target has different AR and
verify costs. It changes why the task stays first: a compatible production-size checkpoint
exists and its Apple economics are unknown, not because a published acceptance number
approaches an old threshold. DFlash becomes a same-target control where checkpoints permit
and a product-scale reference otherwise. Native MTP joins the same gate as a lower-port-cost
control.

We also kept a stricter definition of exactness. A DFlash implementation can truthfully say
that it emitted no unverified token while still allowing MLX dispatch differences to change
the greedy byte stream. fast-mlx cannot. At temperature zero, speculative decoding is
byte-identical or it is a bug.

## A quantizer is not a speedup until attention reads it compressed

The first novel fast-mlx quantizer taught us this lesson painfully. We implemented
TurboQuant faithfully enough to reproduce its mathematical distortion table, then measured
it losing to plain 4-bit affine KV on Qwen3-32B. Worse, the cache materialized full-precision
K/V before attention. Even a future codec winner could not claim the intended bandwidth win
through that path.

The audit therefore split format from execution. Phase A selects a storage/quality winner;
phase B makes compressed-domain attention mandatory before that winner can claim decode
speed. The kernel gate is end-to-end at 32K and 128K, not a Metal chart.

At the same time, a stronger format candidate appeared. KVarN's released `K4V2-g128` setup
spends four bits on keys and two on values, rotates channels, and normalizes variance across
both channel and token axes. The paper establishes the algorithm and 2/2-bit smaller-model
quality; a later official-repository author benchmark—not the paper—reports Qwen3-32B AIME25
parity, roughly four times the KV capacity, and throughput above fp16 for a 16K-context burst
at TP=2. That is a high-quality NVIDIA lead, not an Apple result.

So the new quant cycle begins with an intentionally boring matrix: affine K4V2 and K8V2,
group sizes 64 and 128, per-layer KVTuner schedules, and KVarN. The separate TurboQuant-B
closure owns its outlier, boundary, and QJL ablations; clean-SHA rows can be compared when
conditions match. Every row reports actual packed bytes including metadata, teacher-forced
KL/perplexity at long context, reasoning/code/tool task scores, unified-memory pressure, and
decode speed. Only the quality/size winner earns a custom Metal path.

## The useful outcome is a four-lane queue

The audit did not produce one magic optimization. It produced a portfolio that can no longer
confuse provenance:

1. **Exact decode multipliers:** EAGLE/DSpark first, with DFlash and native MTP controls.
2. **Concurrent serving:** continuous batching, decode-first chunked prefill, then exact
   prefix/session reuse and request-start caching.
3. **Fused long-context memory:** compressed-domain attention, KVarN/asymmetric schedules,
   absorbed MLA, and a bounded TurboQuant closeout.
4. **Model and weight frontiers:** official MLX learned quantization, oQ comparison points,
   and real PrismML Ternary/Bonsai device-tier models.

Several exciting papers remain in research-later: selective cache eviction, codebook KV,
cross-layer residual compression, sparse activation kernels, and fused sampling. That is not
timidity. Each lacks a prerequisite, an Apple implementation, or evidence stronger than a
task already queued.

The general lesson is simple: a research flywheel needs garbage collection. Before asking
“what is new?”, ask “what is true here?” Then require every paper either to strengthen an
existing gate, create a bounded new one, or stay out of the execution queue. A complete
backlog is not the longest list of ideas. It is the shortest faithful map from evidence to the
next measurement.

## Sources and durable result

The ranked matrix, source-review ledger, explicit deferrals, and task links live in the
[dated Sol audit](../reference/2026-07-12-sol-optimization-landscape.md). Key primary sources:
[EAGLE-3 checkpoint](https://huggingface.co/RedHatAI/Qwen3-32B-speculator.eagle3),
[DFlash](https://arxiv.org/abs/2602.06036),
[KVarN](https://arxiv.org/abs/2606.03458),
[KVTuner](https://arxiv.org/abs/2502.04420), and
[MLX-LM learned quantization](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md).
