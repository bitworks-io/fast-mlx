# When smaller KV is not faster

**Whitepaper themes:** The optimization dial — quantified precision-loss tuning; Serving big
models on Apple Silicon; Rapid research integration — the flywheel

The KVarN/asymmetric KV-cache gate started with a tempting promise: spend fewer bytes on the
attention cache, fit far more context, and maybe get a runtime speedup too. That is what the
fast-mlx dial exists to absorb: measured quality loss shown plainly enough for deliberate choice.

The Apple result was more useful, and less flattering, than the pitch.

On Qwen3-32B-4bit on a single Apple M5 Max 128 GiB system, fast-mlx measured lossy KV cells at
**2.78x to 4.51x KV-budget capacity**. That finding is model-specific: it is not a broad support
claim for Qwen-family models, Apple Silicon, or KV compression in general. It also did not produce
a runtime speed win through the current implementation path. Every lossy cell still
materializes full-precision K/V before calling attention, so the compressed cache saves storage
but pays dequantization, layout, and workspace overhead on the hot path. Smaller KV was not
automatically faster KV.

That matters because "capacity" is easy to overclaim. The numbers below are **KV-budget
capacity** after counting payload, metadata, sink/recent windows, tails, alignment, and workspace.
They are not a whole-system "this Mac can run 4.5x the model" claim. Model weights, allocator
behavior, activations, tokenizer state, process overhead, and other allocations still exist.

## The measured dial

The same-weights reference is fp16 KV on the same Qwen3-32B-4bit checkpoint. The lossy rows are
therefore marginal KV-cache tradeoffs, not a comparison against a different weight checkpoint.
Quality was measured with context-locked metrics and task/coherence checks; runtime is the current
materialize-then-attend path. No public competitor comparison is part of this result.

| Tier | Cell | KV-budget capacity | Decode tok/s | Prefill tok/s | Quality signal | Task checks |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Transparent baseline | fp16 KV | 1.00x | 28.46 | 376.33 | Same-weights reference | 16/10/20/20 |
| Balanced capacity | affine K4V2-g64 | 4.27x | 27.52 | 360.54 | median KL 0.0756; ppl +4.58% | 16/9/20/20 |
| Max-fit | KVTuner schedule | 4.51x | 27.53 | 319.00 | ppl +10.01% | 15/9/20/20 |
| Max-fit / capacity-only | KVarN i8 | 4.23x | 13.74 | 276.46 | median at floor; ppl/task differ | 9/11/20/20 |

The Balanced row is the cleanest product result: 4.27x KV-budget capacity with a small throughput
regression. Decode moved from 28.46 to 27.52 tok/s, and prefill from 376.33 to 360.54 tok/s.
Median KL of 0.0756 and perplexity delta of +4.58% put it outside the Transparent default bar, but
inside the predeclared Balanced region. Task checks lost one point in the second domain and held
the others: 16/9/20/20 versus 16/10/20/20.

That is not "free." It is a real, user-visible dial position: choose it when 4.27x more KV-budget
context or concurrency is worth a measured, bounded quality cost.

KVTuner earned a different label. Its frozen schedule reached the largest retained capacity point,
4.51x, while decode stayed essentially tied with the Balanced affine cell at 27.53 tok/s. Prefill,
however, fell to 319 tok/s, and perplexity moved to +10.01%. The task checks were 15/9/20/20. That
is above the Balanced quality contract but still above the coherence floor for this measured
model, so it belongs in Max-fit: explicit opt-in for users trying to make a context fit, not a
default for strict code, math, or tool workloads.

KVarN i8 was the most interesting disappointment. It cleared capacity at 4.23x and showed strong
teacher-forced median behavior: its aggregate median KL landed at the fp16 pipeline floor at the
reported precision. That does not make it fp16-equivalent. Its tail, perplexity, and task results
still differed, including +2.86% perplexity and a hard math-task drop to 9/11/20/20. The explicit
uncompiled correctness path was also slow: 13.74 decode tok/s and 276.46 prefill tok/s. That makes
it a **capacity-only Max-fit** candidate for this measured model, not a general tier or speed
optimization.

The negative cells are just as important. KVarN i16 and K8 affine controls did not earn separate
user-facing tiers: they failed to improve the practical frontier once capacity, quality, and
runtime were all counted together. Two aggressive affine cells failed the hard coherence floor.
They are not hidden "expert modes." The dial permits informed loss; it does not offer garbage.

## What "informed consent" means here

The result forced the product policy to be precise. A low-loss default and a useful lossy tier are
different promises.

The **Transparent** setting is the safe baseline: fp16 KV, no marginal KV-cache loss.

The **Balanced** setting is an informed capacity tier. Affine K4V2-g64 is measured and useful:
4.27x KV-budget capacity for a small decode/prefill regression and measured quality drift.

The **Max-fit** setting is for constrained hardware or unusually long contexts. KVTuner and KVarN
i8 do not pretend to be conservative. They show the capacity gained, the loss measured, the task
domains where the model held or slipped, and why a user might still choose them. That is the point
of a dial instead of a binary "pass/fail" switch. Both use separate 256-byte control state in this
packet. It is reported separately from the displayed storage totals and capacity ratios and is
immaterial at their shown precision; it is not a hidden free path.

The hard floor is non-negotiable. Users may choose noticeable loss for fit or context. They may
not choose corrupted state, non-finite logits, incoherent output, catastrophic task collapse, or a
dominated setting that gives up quality without buying capacity or speed.

## The next gate is fusion

This experiment selected the next technical question: **fused compressed-domain attention**. The
storage formats proved they can reduce KV-budget memory. They have not proved that decode gets
faster while attention still reads materialized full-precision K/V.

The next gate has to keep the selected compressed layout compressed through attention reads, then
measure full 32K/128K behavior end to end. A microbenchmark or kernel chart is not enough. The
product question is whether a user sees a better frontier: more context, acceptable quality, and
an actual runtime win after all overheads are charged.

This result is deliberately model-specific. It is Qwen3-32B-4bit on one M5 Max 128 GiB system.
Broad support requires the same gate on a materially different popular model family, because KV
sensitivity, head geometry, task loss, and memory behavior are not universal constants.

The general lesson is simple: compression is a storage result until the execution path proves
otherwise. For this measured model, KVarN and asymmetric KV moved the Apple capacity frontier.
They did not make lossy KV a speed tier yet. That is not a failed cycle. It is the flywheel doing
its job: keep the useful capacity rows, reject the incoherent ones, and spend the next kernel
effort only where the measured frontier says it can matter.

## Sources and measurement method

The phase plan is the
[KVarN/asymmetric KV-cache frontier plan](../superpowers/plans/2026-07-14-kvarn-kv-frontier.md).
The dated decision is the
[2026-07-18 KVarN / asymmetric KV-cache frontier verdict](../superpowers/verdicts/2026-07-18-kvarn-kv-frontier.md),
with compact evidence in
[kvarn-kv-frontier-evidence-2026-07-18.jsonl](../superpowers/verdicts/kvarn-kv-frontier-evidence-2026-07-18.jsonl).
Metric definitions are in [Reading the quality metrics](../reference/quality-metrics-explained.md).
The algorithm/source boundary is recorded in the
[KVarN / asymmetric KV-cache algorithm lock](../reference/kvarn-kv-algorithm.md), and lane rules
are summarized in [How to read technique verdicts](../superpowers/verdicts/README.md).
