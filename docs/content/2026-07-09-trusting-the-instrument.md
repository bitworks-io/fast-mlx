---
title: Who measures the measurer? Auditing a precision-loss harness that was quietly lying
date: 2026-07-09
whitepaper_theme: The optimization dial — quantified precision-loss tuning
status: draft
---

# Who measures the measurer?

fast-mlx's product isn't raw speed — it's a **dial**: turn up the compression, and see *exactly* how much accuracy you trade. That promise lives or dies on one thing — the instrument that produces the "how much accuracy" number. So before we let that instrument judge any optimization, we audited it. It was lying to us. Subtly, and in the worst possible place.

## The number that looked fine

The dial's cheapest, most sensitive quality signal is **KL divergence**: run a compressed model and a high-precision reference on the same prompts, and measure how far the compressed model's next-token probability distribution drifts from the reference's. Small KL = imperceptible loss; large KL = the model is thinking differently.

Our first real measurement — a 4-bit model vs its 8-bit self — came back at **0.43 nats**. Plausible. We almost moved on.

## The audit

We didn't move on; we audited the harness against its hardest case first. That surfaced the flaw immediately: the metric was **free-running**. Each model generated its *own* greedy continuation, token by token, and we compared them position by position. That's fine — until the two models pick a different token. From that point on, they're continuing *different sentences*, and every "per-position KL" after the split is comparing two unrelated contexts. It's not measuring precision loss anymore; it's measuring divergence, which is a different and much noisier thing.

How bad was it, concretely? Of 72 positions in that 4-bit-vs-8-bit run, only **8** were on shared context. The "0.43 nats" came from those 8 early, biased positions. Computed over all 72, the number was **14.2 nats** — a 33× difference depending on where you stopped looking. And the same-weights control (a model against *itself*, which should read ~0) had a p95 of **20 nats**, pure divergence noise.

Worse, the failure is adversarial to our own roadmap. The most aggressive compression we want to ship — 2-bit KV caches — is exactly where two models diverge *fastest*: often after a single token. There, the free-running metric would have almost no shared positions to measure, and would produce essentially **no signal** precisely where we most need one. The instrument was destined to go blind at the frontier it exists to survey.

## The fix: teacher-forcing

The fix is standard once you've named the problem: don't let the models free-run. Pick **one** continuation — the reference's greedy path — and **feed it to both models**, forcing each to score the *same* token at the *same* position. Now every position is on locked, identical context, and the KL at each one is a clean apples-to-apples comparison of the two distributions. No divergence, no starvation, no dependence on where you stop.

The change was small and well-bounded — a forced-continuation variant of the model's `logprobs` call (feed the forced token instead of the model's own argmax), and a redefinition of the metric to be teacher-forced *by construction* rather than as an optional after-the-fact correction. We wrote the tests first: a spy that proves the metric feeds the reference's continuation to *both* sides and scores every position; guards that **throw** rather than silently return a number when the continuation is empty or a score row is missing. A metric that can quietly produce a meaningless value is worse than one that stops.

## The result

The distortion is gone. The same-weights noise floor dropped from a p95 of 20 nats to a **median of 0.0013 and a p95 of 0.0095** — the metric now reads a model against itself as ~zero, as it must. The 4-bit-vs-8-bit number resolved to a stable **0.119 nats across all 72 positions** — ~91× above the noise floor, and **byte-for-byte reproducible** on rerun at temperature 0. And because teacher-forcing scores the reference's actual tokens, **perplexity delta** fell out of the *same* forward pass for free: same-weights −0.24% (comfortably inside the dial's 1% "unnoticeable-loss" gate), 4-bit-vs-8-bit +39.6% (a decisive fail, consistent with the KL).

One honest footnote we wrote into the record: these particular numbers characterize the *instrument*, not the model — the reference was still 8-bit (a stand-in for true fp16) and the corpus was three short prompts. Those are the next items on the hardening list. But the metric itself is now trustworthy.

## The lesson

The instrument that measures everything has to be measured too — and it won't confess its flaws under gentle use. A precision-loss metric that lets models free-run isn't measuring precision loss; it's measuring divergence, and it fails silently, worst, exactly at the aggressive-compression frontier you built it to survey. You only catch that by pointing the tool at its hardest case and asking whether the number it gives back is the number you think it is. Honest measurement isn't a default you get for free; it's a property you design in, test, and verify — before you trust a single downstream claim built on top of it.
