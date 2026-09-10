---
title: "The default nobody chose"
date: 2026-09-10
whitepaper_theme: Serving big models on Apple Silicon
status: published
---

# The default nobody chose

A client that sends no `temperature` gets greedy argmax. That sounds like a reasonable default until
you notice two things: the model's authors shipped a recommended sampler inside the checkpoint, and
greedy is the configuration a thinking model is most likely to loop on.

Our resolver had one line deciding this:

```swift
guard let temperature, temperature != 0 else { return .greedy }
```

It returns before `top_p`, `top_k`, `min_p`, and `seed` are looked at, so a request carrying
`top_p` but no `temperature` had those fields silently discarded. Every serving route delegated to
that resolver. There was no server-side sampling default of any kind, and the loader did not read
the sampling fields in `generation_config.json` at all — the config decoder declared exactly three
keys, none of them sampling.

So the artifact on disk said `do_sample: true, temperature 1.0, top_p 0.95, top_k 20`, and the
server was structurally incapable of reading it. Real clients send minimal parameters. The practical
consequence is that traffic was decoding greedily against the model author's stated intent, and
nobody had chosen that — it was the residue of an early return.

## Reading the artifact instead of hardcoding a preset

The fix is a flag, `--default-sampling generation-config`, that fills *only* absent fields from the
served checkpoint's own `generation_config.json`. Sourcing the values from the artifact rather than
from a hardcoded preset matters for a boring reason: the alternative couples a family-agnostic
serving layer to one checkpoint's numbers, and those numbers then rot the first time the model is
swapped. Reading them from the artifact means the behavior follows the model.

Three things about it are deliberate.

**It is opt-in.** Changing what every existing client gets is not a change to make on by default in
the same week as a deployment.

**It fails closed at startup.** If the flag is set and the artifact is missing, unparseable, or says
`do_sample: false`, the server refuses to start. Falling back to greedy would silently reproduce the
exact defect the flag exists to fix, which is the worst available outcome: a healthy-looking boot
and the old behavior.

**`temperature: 0` still means greedy.** The default only fills `nil`. A caller who genuinely wants
argmax keeps it, and that path has its own pinned test because it is now load-bearing rather than
incidental.

There is also a refusal that exists purely because one decoder cannot honor sampling at all. On a
compiled-fp16 route, the decoder ignores a sampling request; with the flag on, the greedy startup
probe would still pass, the server would boot healthy, and then every real request would fail
mid-stream with an opaque error. That combination is refused at load instead. Worth noting that this
is *not* a rare configuration — it is the dominant resolution for any all-dense-attention
checkpoint, so the refusal earns its keep rather than guarding a hypothetical.

## Measuring it, and the control that made the measurement mean anything

We predeclared the metric and the accept bands before collecting anything, then ran a fixed
ten-prompt corpus twice against the same binary and checkpoint: once with the flag off, once on.
Neither arm sent any sampling parameter, because the thing under test is the *default*.

The primary metric was degeneration — distinct 4-gram ratio and the longest back-to-back repeated
n-gram — since repetition is the specific failure greedy decoding is associated with on a thinking
model.

The interesting methodological point is a band we added before running: **the two arms must actually
differ.** That check is worthless on its own. If the serving path were nondeterministic for any
reason, the arms would differ whether or not the flag did anything, and the check would pass for
free. So we ran the greedy arm *twice* and required the two runs to be byte-identical first. They
were, on all ten prompts. Only then does "the flag arm differs" isolate the flag as the cause.

We wrote the inconclusive branch into the predeclaration too: had the baseline been
nondeterministic, the mechanism claim would have been recorded as inconclusive rather than passed.
Deciding that in advance is cheaper than arguing about it afterwards.

## What it showed

Every measurable band passed. The flag arm was better, not merely non-worse:

| | flag off (greedy) | flag on (artifact preset) |
| --- | --- | --- |
| mean distinct-4-gram ratio | 0.961 | 0.974 |
| worst-row distinct-4-gram ratio | 0.674 | 0.847 |
| worst-row repeat run | 2 | 1 |
| errors | 0 | 0 |
| decode tok/s | 21.3 | 21.0 |

The mean barely moved, and that is the part worth carrying forward. Most prompts degenerate under
neither setting, so **the gain is concentrated in the tail** — the worst row improved from 0.674 to
0.847, and back-to-back repetition disappeared entirely. A mean-only read of this change would have
called it noise. The throughput difference is inside run-to-run variance at this sample size and we
report it as no measurable cost, not as a 1.4% regression.

## Two harness defects, and why the first run was thrown away

The first attempt was discarded, and both faults were ours rather than the product's.

The harness read only `message.content`. The served model is a thinking model that returns both
`content` and `reasoning_content`, and when the token budget is consumed inside the reasoning block,
`content` comes back **null** — present but null, which the parser treated as malformed. Compounding
it, the token budget we had predeclared was small enough to guarantee that outcome. Four of the
first five prompts errored.

Every one of those failures was a `200` with a body the instrument mis-parsed. That distinction is
recorded next to the discarded log, because a future reader finding a run full of errors would
otherwise reasonably conclude the server had a defect. It did not.

The correction was to measure the full generated text — reasoning block and content — and to raise
the budget so content is actually reached. Degeneration inside a thinking block is still
degeneration, and on this model that is where most of the tokens are, so measuring `content` alone
was measuring the wrong thing even when it worked.

## Limits, stated plainly

Ten prompts, one checkpoint, one host, and the sampled arm is unseeded because real default traffic
carries no seed. This is a directional read, not a significance claim, and degeneration is a proxy
for one failure mode rather than a measure of overall quality. No task-accuracy or capability claim
is made or implied.

It is also not, by itself, a decision to turn the flag on anywhere. It is the evidence a decision
would need.

## The transferable part

The bug here was not a wrong value. It was an *absent* one: a default that nobody selected, produced
by an early return, sitting underneath a configuration file that already contained the right answer.
Those are hard to see precisely because nothing is misconfigured — every layer is doing what its
code says, and the code says nothing about the case in question.

Two habits fall out of it. When a config file ships values you never read, that silence is a finding
rather than an absence. And when a comparison's whole meaning rests on two things differing, spend
the extra run proving the baseline is stable first — otherwise the check passes for reasons that
have nothing to do with your change.
