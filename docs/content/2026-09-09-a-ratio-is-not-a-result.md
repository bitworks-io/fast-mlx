---
title: "A ratio is not a result"
date: 2026-09-09
whitepaper_theme: Serving big models on Apple Silicon
status: published
---

# A ratio is not a result

We had a measured speedup for speculative decoding: 1.31x, honestly obtained, with controls that
passed. It was also, for our purposes, close to useless — because it was measured with sampling
switched off in a way no real request uses.

Fixing that took one afternoon of measurement and produced a better number. What it also produced
was a reminder that a speedup is a fraction, and that changing the experiment can move the bottom
of the fraction while you are watching the top.

## The number that did not transfer

Speculative decoding proposes tokens with a cheap draft head and verifies them against the real
model. Accept enough proposals and you get several tokens for roughly the cost of one. The published
figure — a 1.31x median — came from a run at `top_p=1, top_k=0`: no truncation at all, every token
in the vocabulary eligible.

Real traffic does not look like that. Our recommended preset sends `top_p=0.95, top_k=20`. And
truncation does something interesting to speculation: it sharpens the target distribution onto a
handful of tokens while the draft head's distribution stays spread out. Where the two heads agree,
that *raises* the acceptance rate. Where their surviving token sets are disjoint, acceptance for
that block is exactly zero and the speculation is pure overhead.

Which effect wins is not something you can settle by argument. We tried, in both directions, and
the honest answer each time was that it depends on how peaked the model happens to be at the
positions it is asked about. So: measure it.

## Predeclare the band, then look

Before running anything we wrote down what would count as success: a median ratio of at least 1.20,
with no individual prompt slower than 1.05x. We committed that file before the number existed.

The threshold moved down from the earlier 1.30, and the reason is worth stating because it looks
like moving the goalposts. The 1.30 bar had been set to decide whether to *build* the truncation
work. That cost was already spent. The new bar decides something different and cheaper — whether to
switch on a feature that already exists. Those deserve different thresholds, and saying so in
advance is what keeps it from being a rationalization afterward.

We also wrote down what would make us distrust a *good* result: a composed cost model gave an upper
bound of about 1.61x, and we declared in advance that measuring above that ceiling would be a defect
signal rather than a triumph, since every term the model leaves out is a cost we would be failing to
pay.

## The denominator moved

Here is the part we nearly missed.

A speedup ratio is speculative throughput over baseline throughput. We were changing the sampling
parameters to make the *speculative* arm realistic. But the baseline arm samples too — and in our
inference library, `top_p=1, top_k=0` and `top_p=0.95, top_k=20` are not the same code. The first
picks a sampler that draws directly from the distribution. The second picks one that sorts the whole
vocabulary, walks a cumulative sum, masks, sorts again for the top-k, and *then* draws.

That is roughly 152,000 elements sorted per token, on the baseline, that were not there before.

So both arms got slower, and the ratio between them would have told us nothing about which. Our
control for the baseline was inherited from the earlier run: it checked throughput against a fixed
expected value with a ±15% tolerance. The two instruments involved had previously been shown to
agree to within 0.4%. A ±15% window around a number we could reproduce to 0.4% is not a control; it
is a formality that will pass whether the baseline slows by 1% or by 14%.

We re-centered it on a freshly measured baseline and tightened it to ±5%. And we wrote into the
contract that the result must be reported as **three numbers — baseline, speculative, and ratio —
never the ratio alone.**

## What the measurement said

| | seeded | nondeterministic |
|---|---|---|
| median ratio | **1.2884** | **1.3143** |
| slowest prompt | 1.1363 | 1.2006 |
| speculative tok/s | 33.51 | 34.51 |
| baseline tok/s | 25.84 | 25.71 |

Accept, on both. Realistic sampling cost about 2% of the speedup on one arm and 6% on the other —
much less than the argument had allowed for.

And the baseline had indeed slowed, by 1.22% and 1.61%. Of the 1.75% that the seeded ratio moved,
roughly 1.4 points came from the denominator. Publishing "1.29x, down from 1.31x" without that
decomposition would not have been false. It would have been unattributable — and the next person to
optimize the speculative path would have spent their time chasing a cost that belonged to the
control arm.

## Why acceptance barely moved

We also added instrumentation that recorded, at every position where the model proposed a token, how
many vocabulary entries actually survived truncation.

The median was **one**.

At this preset, on this model, the 0.95 nucleus usually collapses to a single token. The model is
simply confident most of the time. That is the whole explanation for why acceptance held up: the
disjoint-supports catastrophe needs two heads to disagree about where the probability mass lives,
and most of the time there is only one place for it to be.

The same instrument reported something we would otherwise have claimed without knowing: `top_k=20`
bound at **2.8%** of positions. Top-p was doing essentially all the work. We had described the run
as testing our support for both parameters. It tested one of them. That is now written down as a
limit of the result rather than left as an implication of it — if a flatter model shows up, the
top-k path is unqualified and needs its own evidence.

## The corroboration we liked best

The strongest check was free. Two different programs measured this system — one with telemetry on,
one with it off and the instrumentation unwrapped. For the deterministic seed, they should drive a
bit-identical decode.

So we predeclared the first program's per-prompt proposal and acceptance counts as literal constants
and had the second program compare against them: `[220, 210, 247, 221, 199, 214, 225, 213]` proposed,
`[144, 150, 131, 143, 155, 148, 141, 148]` accepted.

All eight matched exactly.

A ±0.05 band on an average would have passed almost regardless. Exact equality across eight prompts
in two independently written instruments is a much harder thing to fake, and it cost nothing but the
decision to write the numbers down first.

## What we would tell someone else

Three things survive from this that are not about speculative decoding at all.

**When a change touches both arms of a comparison, the ratio is the least informative number you
have.** Report the numerator and denominator separately, or you will publish a figure nobody can
attribute — including you, six weeks later.

**A tolerance should be set by the reproducibility you have demonstrated, not by what feels safe.**
±15% around a quantity reproducible to 0.4% cannot fail for any reason you would want to hear about.
Inherited tolerances go stale exactly when the experiment changes, which is exactly when you stop
looking at them.

**"We measured it at the real settings" is a claim, and it has parts.** Ours had two parameters in
it and only one of them was doing anything. The instrument that told us so was a count of non-zero
entries in an array we had already allocated — about as cheap as evidence gets, and the difference
between reporting what we ran and reporting what we knew.
