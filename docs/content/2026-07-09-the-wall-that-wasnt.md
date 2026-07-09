---
title: "The 7K wall that wasn't: jetsam forensics, a quadratic allocator, and the statistic hiding in the tail"
date: 2026-07-09
whitepaper_theme: The optimization dial — quantified precision-loss tuning
status: draft
---

# The 7K wall that wasn't

Our precision-loss harness had just been hardened — teacher-forced KL, perplexity, a versioned
corpus, provenance records. Then it hit a wall: any measurement past roughly 7,000 tokens of
context died with a SIGKILL and zero output. The findings note blamed the obvious suspect: our
compiled decode core, with its fixed-shape KV buffers, must have a preallocation ceiling. The
engine "couldn't serve past 7K." That was the story we almost carried forward.

It was wrong in every particular. The wall wasn't in the engine, wasn't a ceiling, and wasn't
even one process's fault.

## Read the corpse, not the code

A SIGKILL with no output means the OS killed you, and macOS writes down why. The box's
`JetsamEvent` reports told a story no amount of code-reading had suggested: at the moment of
death, the Swift harness was **112GB resident** and the Python reference process — a separate
process, running the *same* measurement from the other side — was at **88GB**. Two processes,
200GB of anonymous memory, a 128GB machine. The kernel shot the biggest one.

And the memory wasn't GPU-wired — it was compressible, idle, *dead*. That's not a model
working set. That's a leak-shaped something.

## The quadratic allocator

An instrumented probe made the something visible. Teacher-forced scoring ran one forward pass
per scored token, on a cache that grows one token per step. Every step therefore allocates
transient buffers *slightly larger* than the last step's. MLX's allocator keeps freed buffers
in a cache for reuse — but a cached buffer can only serve a request that fits, and every future
request is bigger. So nothing is ever reused, and nothing is ever returned: the cache grows as
the **sum of all step sizes — O(context²)**.

Measured on a 32B model: 0.9GB of dead cache at position 1,000; 8GB at 3,000; **43GB at 6,750**
— while active memory sat flat at 17GB. Both processes did this simultaneously. The "~7K
ceiling" was just the context length at which their combined garbage crossed physical RAM.
One config detail made it worse: we'd raised the GPU wired limit to 115GB for big models, and
MLX's default cache limit *tracks that limit* — so the allocator believed it was entitled to
hoard nearly all of it. Twice.

The compiled decode core we'd blamed? Its fixed-shape buffers are exactly what *don't* have
this problem. It prefilled 32K tokens on the first try.

## Two fixes, both necessary

**Chunked scoring.** Teacher-forcing doesn't need decode-style stepping — the whole point is
that the continuation is known. Feeding it in 512-token chunks gives every position's logits
directly from the prefill math: 24,000 tokens now score in about a minute per side instead of
ten, and the growing-transient churn shrinks by the chunk factor. Honesty requires the next
sentence: chunking alone did *not* flatten memory. The materialized K/V slices still grow chunk
to chunk, and an unbounded probe still hoarded 62GB by 16K context.

**A bounded allocator.** The flat-memory guarantee comes from capping the buffer cache at 8GB
on both sides, evicting the unreusable buffers instead of collecting them. With both layers, a
32K-token scoring pass peaks at 33.8GB — cache pinned exactly at 8GB — and the serving path's
decode speed is untouched: 155.4 tok/s before and after, 60/60 token-identical at temp 0.

## The statistic in the tail

Fixing the wall exposed the second problem: with long context finally measurable, the numbers
looked *too good*. A 4-bit model scored against a true bf16 reference at 24K tokens showed a
median per-position KL near the noise floor. Was quantization loss really invisible?

No — the median was looking in the wrong place. Most tokens in natural text are easy;
a 4-bit model and a bf16 model agree on "the" and "of" all day, and those positions bury the
median. Quantization divergence is a **tail phenomenon**: it lives in the hard positions, and
it *accrues with context depth*. So the long-context headline is now a per-entry p95 — with a
deliberately tail-biased quantile convention (ceiling index, so a divergence tail of exactly
5% mass can't land just under the statistic built to catch it).

The instrument now reads: same-weights tail floor **0.004 nats**; 4-bit vs bf16 tail p95
**0.31 nats at 5.3K tokens → 1.67 nats at 24K** — four hundred times the floor, growing with
depth, exactly where a median said "nothing to see." Perplexity agrees: +21.4% vs bf16. That
pair of numbers — measured against a *real* fp16-class reference, at real long context, with
the git SHA of the instrument in every record — is the baseline row that 2-bit KV-cache
quantization will have to beat.

## The lesson

Root-cause the crash you have, not the one your architecture makes plausible. The "engine
ceiling" story was coherent, flattering to nobody, and completely wrong — the truth (two
processes quadratically hoarding freed buffers under a raised memory limit) was only visible
in the kernel's own postmortem and a memory curve nobody had plotted. And when the wall came
down, the next lie was statistical, not mechanical: a median that averaged away the exact
signal the harness exists to measure. Instruments don't just need to run at the frontier —
they need statistics chosen for what the frontier's failure actually looks like.
