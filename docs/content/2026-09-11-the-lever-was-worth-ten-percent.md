---
title: "The lever was worth ten percent"
date: 2026-09-11
whitepaper_theme: Serving big models on Apple Silicon
status: published
---

# The lever was worth ten percent

We were about to build a cache.

The sparse-attention path in our long-context decode re-pools its key blocks on every single token.
At a cache length of 32,768 that is thousands of blocks re-derived per layer per step, and twelve
such layers per forward pass. The fix writes itself: keep the pooled blocks, append the one new block
each step, and stop redoing the other few thousand.

It is a good idea. It is also a real correctness surface — a cache that must be invalidated on
speculative-decode rewind, must not be poisoned by a copy-on-adopt transaction, and must hold a
truncation invariant on trim. That is the kind of thing you want to be sure about before you write
it.

So before building it we measured what it was worth. The answer was **10%**, against a bar of 30%.
We did not build it.

## Measuring a lever you have not built

The tempting evidence was already in hand: the sparse-attention layer's per-step cost rises 1.795x
between cache lengths 2,048 and 8,192, while the recurrent layer next to it stays flat at 1.0000
across the same span. Something in that layer scales hard with context, and re-pooling scales with
context, so re-pooling is the thing. Ship the cache.

That reasoning has a hole in it wide enough to lose a month in. **1.795x is a layer aggregate.** At
least four terms inside the layer scale with context, and the cache removes exactly one of them. The
other three — a selection-score contraction, a top-k selection, and a dense O(context) attention mask
— scale just as hard and are untouched.

So we built an instrument that removes the pooling phase *in place*, inside the real graph, and
measures what the layer costs without it. Two arms:

- **the ceiling** — skip the entire pool/normalize/rotary phase, substituting a pre-computed tensor.
  Whatever this saves is the most any cache could ever save.
- **the actual proposal** — do exactly what the cache would do: reuse a stored prefix, compute the
  one new block for real, concatenate.

Both run against the production code path, not a reimplementation, so there is no chance of measuring
a copy that quietly picked different kernels.

Fitted across four cache lengths from 4,096 to 32,768, all of them above the point where the sparse
path engages:

| | share of the context-driven slope |
|---|---|
| ceiling — remove the whole pooling phase | **10.53%** |
| the proposed cache | **10.48%** |

The second row is the one that closed it. **The cache captures 99.5% of the phase it targets.** It is
not a badly designed optimization that a better design could rescue. The phase is just small. There
was no version of this worth the invalidation logic.

## Measuring the instrument before believing it

A rig that reports a small number has two possible meanings — *the lever is small*, or *I cannot see
the lever* — and they call for opposite decisions. So the rig had to prove it could see.

The pooling term at 32,768 is on the order of tens of microseconds. The layer arm it is being
subtracted from takes about 10 milliseconds, and a 3% measurement floor on 10 milliseconds is 300
microseconds. Stated that way, the thing we were trying to detect could have been an order of
magnitude below the noise, and we would have read "invisible" as "small" and killed the idea for the
wrong reason.

The resolution check settled it: the measured gap was **54x** the run's own noise floor, where a
byte-identical repeat of the same arm reproduced itself to within **0.047%**. The instrument could see
it. The lever really is small.

Two further controls earned their keep. A recurrent-layer arm, which has no business scaling with
context, came in flat at 0.10% of the sparse layer's slope — so the rig was not simply reporting
memory pressure that rose with every arm. And the whole run reproduced an earlier measurement's ratio
to within 0.77%, which is what makes it comparable to anything at all.

The first run of this was **formally void**, and we recorded it that way. Four control arms breached
the drift gate. All four run at about a quarter of a millisecond, where a 3% *relative* threshold is
roughly seven microseconds — inside timer jitter. Every arm that mattered passed, the worst at 1.09%.
A gate you explain away after seeing the result is not a gate.

So we fixed the gate rather than the interpretation: it now carries an absolute floor alongside the
relative one, plus a check that fails the run if that floor ever becomes the binding threshold on an
arm that feeds the result. The second run passes cleanly, with nothing failing, and reproduces the
first to within 0.85% on the underlying slope. The numbers above are the clean run's.

The two runs are worth reporting together, because of what they did to the threshold. The first came
in at 9.89%. The second came in at 10.48%. Our predeclared bands put reject below 10% and no-verdict
from 10% to 30%. **The measurement straddles that boundary.** Had we let the decision turn on which
side of 10% a single run landed, it would have been settled by noise.

## The number we were not looking for

The interesting result was in the 90% the pooling phase did not explain.

The per-key slope works out to roughly **155 KB of implied memory traffic per key of cache**, measured
against the same machine's own streaming bandwidth, probed in the same run. The path actually needs a
few hundred bytes per key.

The concrete version is starker. At a cache length of 32,768, the context-driven part of a single
layer's decode step costs **7.98 ms**. Streaming the entire 8.39 MB key state once, end to end, costs
**0.0129 ms**.

That is **618x** — and the voided first run put it at 591x, so this is not a conclusion that depends
on which run you read.

**Long-context sparse-attention decode, on this path, is not a memory-bandwidth problem.** It is
nowhere near the roofline. The cost is materialization — building a dense mask proportional to the
full context and running dense attention across all of it, on every token, to produce a result that
only ever uses a sparse subset.

This quietly reframes a whole line of work. Two of our candidate optimizations attacked the pooling
and copying side of that layer, on the shared assumption that moving key bytes around was the
expense. One of them shipped on its own merits and stays. But the premise underneath both of them
does not survive this measurement, and we would rather find that out from a one-day instrument than
from a month of building.

The successor is better news than the thing we rejected: the engine already contains a compact
selection path that skips the dense mask entirely. It is off by default. The next measurement is not
"should we build something" — it is "what does the code we already have actually buy". That is a much
cheaper question, and this run is the reason we know to ask it.

## What we would keep from this

Predeclare the number that would kill the idea, and write it down before the rig runs. Ours said
below 10% reject, at or above 30% build. The answer came back at 9.89%, and then — same rig, same
host, one corrected gate later — at 10.48%. A threshold your measurement can land on either side of
is not telling you anything, and the honest reading was never a crisp verdict either way. It was that
the ceiling sat three times under the bar, which is the same decision from a direction the boundary
cannot wobble.

And measure the ceiling before you cost the build. The most a change can possibly buy is almost always
cheaper to find out than the change is to write.
