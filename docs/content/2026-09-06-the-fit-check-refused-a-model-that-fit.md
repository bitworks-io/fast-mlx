---
title: "The fit-check refused a model that fit"
date: 2026-09-06
whitepaper_theme: Serving big models on Apple Silicon
status: published
---

# The fit-check refused a model that fit

An earlier note — [*The bytes were derivable. We refused anyway.*](2026-08-19-the-bytes-were-derivable-we-refused-anyway.md)
— set out the rule the fit-check lives by: **fail toward RED.** The number the engine computes before
it loads anything answers one operator question — *will this actually hold, or will it OOM three
thousand tokens in?* — and when the engine cannot model a model honestly, it refuses rather than
printing a green it cannot stand behind.

This week the rule bit us from the other side. The fit-check refused a configuration that genuinely
fit. Fixing it turned out to require correcting two things we had written down and believed.

## A table that isn't there

The model in question keeps a very large n-gram embedding table. Serving it with an offload plan
streams that table's shards from a sealed on-disk row store instead of holding them in memory. The
model loads; the table does not.

The sizer knew none of this. It sized the checkpoint the only way it knows how — every byte on disk —
and produced 105.54 GiB. On a 256 GiB machine that was merely wrong. On a 128 GiB machine, where the
planning ceiling is 75% of RAM, or 96 GiB, the same configuration modelled 105.54 GiB against a 96
GiB ceiling and **refused**. The configuration needed about 73 GiB. It fit comfortably. You could
only serve it by passing `--force`, which is exactly the habit a fit-check exists to prevent: once
operators learn to `--force` past a refusal they don't believe, the refusals that matter stop working.

This is the mirror image of a phantom-GREEN, and it is not obviously the safer failure. A green you
can't stand behind gets you an OOM. A red you can't stand behind gets you an operator who ignores
reds.

## Subtracting the table was the wrong fix

The obvious repair is to subtract the offloaded tensors: whole-file total, minus the bytes that are
no longer resident. That is where the interesting part starts, because subtraction alone would have
been **worse than the bug it fixed**.

The row store has its own memory budget. It streams rows, and it keeps some of them — the plan file
declares a `maxResidentBytes`, and the loader's own documentation warns that an over-large residency
budget "quietly gives back" the memory the offload saved. That budget is operator-authored, and the
plan validator bounds it only as *greater than zero*.

So the two-term formula models a table that is entirely gone. A legal plan with a 16 GiB residency
budget would model 75.7 GiB and then consume far more. That is not a smaller error in the same
direction — it converts a false refusal into a load that passes the check and then OOMs. The false
RED at least fails safely.

The residency budget is therefore a mandatory third term, not a refinement:

```
adjusted = wholeFileTotal - offloadedTensorBytes + plan maxResidentBytes
```

We only noticed because we asked what the row store does with memory, rather than what the checkpoint
does without the table. The two questions have different answers.

## The key that matched nothing

To subtract the offloaded tensors you must first find them, which means matching checkpoint keys. Two
internal records quoted the key shape. Both quoted it the same way. Both were wrong.

The real tensors carry a module namespace the quoted form omits. A matcher built from the written-down
shape matches **zero** tensors — and zero matches is the quiet failure, because it sums to nothing,
applies no reduction, and looks exactly like a checkpoint that simply had no table to offload. Worse,
a synthetic test fixture written from the same record would use the same wrong names and pass.

That is the trap worth naming: **a fixture built from the same source as the code under test cannot
falsify it.** The two agree because they share an ancestor, not because either is right. The only way
out was to read the headers of the actual artifact.

Doing that settled something else. An earlier record had concluded the reduction was *not derivable
from the checkpoint* — that the offloaded figure was a property of the sealed row store, knowable only
by asking the store. It isn't. Safetensors headers carry per-tensor `data_offsets`, and summing the
spans of the matched tensors gives the figure exactly: **32,000,153,600 bytes**, which is also, to the
byte, the size of the row store file on disk. Two independent routes to the same number, where we had
previously recorded that only one route existed.

So the code reads headers. Not the blobs — the shards are about 10 GiB each, and the reader takes the
8-byte length prefix and the JSON header and stops.

## Failing closed on a measurement

A number this load-bearing needs to be wrong loudly or not at all. Before the reduction is applied, a
structural audit requires at least one match, exactly one layer index, shard indices contiguous from
zero, and the same field set on every shard. Any violation throws, and every throw leaves the
conservative full-resident figure standing.

That audit also makes the feature self-limiting without naming a model family. A checkpoint that isn't
this architecture carries none of these keys, so it produces zero matches, so the audit refuses. We
did not have to condition on a family name — a thing this project has removed twice, because family
names are the easiest wrong answer in an inference engine.

An independent review then found four separate ways the figure could still come out **too small**,
which is the only direction that matters:

- **Declared spans were never bounded by the file.** The subtracted term came from header offsets; the
  minuend came from real file sizes. Only one of the two could exceed physical reality, and it was the
  one being subtracted. A shard over-declaring its tensors by 2x would subtract ~64 GB and still pass
  every other check.
- **Duplicate keys across shard files were counted twice.** Matches accumulated in a list while the
  audit checked a set, so a consolidated copy staged beside the sharded set satisfied every structural
  check while doubling the reduction. A sibling tool in the same repository already treated this as a
  hard error; the newer code had not inherited it.
- **The sum used a trapping `+`.** Overflow would crash the server rather than return an error — not
  the conservative direction, in a file whose entire premise is that any error means no reduction.
- **A JSON number of `1e300` silently clamped to the largest integer** instead of being rejected, and a
  clamped-but-positive value sails past every downstream "greater than zero" check.

Each is now a specific refusal with its own test. The duplicate-key guard was then mutation-checked —
removed on purpose, to confirm its test actually fails without it. An assertion you have never seen
fail is not yet evidence.

## What it reads now

On the real artifact:

```
ngram offload fit adjustment: whole-file total 113325274612 B
  - offloaded n-gram bytes 32000153600 B
  + row-store residency budget 100000000 B
  = adjusted weights 81425121012 B
fit-check [GREEN] weights=75.83 GiB (measured) peak=78.47 GiB binding=fits
```

75.83 GiB against a 96 GiB ceiling — 79% of it — where the same artifact unadjusted is 105.54 GiB and
refuses. Actual resident memory during the run was 73.39 GiB, so the model still sits 2.44 GiB *above*
the truth. That gap is unexplained, and we are leaving it there: it is in the conservative direction,
and a fit-check that errs toward "slightly too big" is doing its job. We would rather carry a known
2.44 GiB of pessimism than close it with a term we cannot derive.

One last correction, recorded because it will otherwise be rediscovered. The measurement above sums
header offsets for both terms and lands on 81,324,594,328. The shipped code doesn't compute that: its
first term is a sum of whole *file* sizes, which includes each shard's length prefix and header JSON —
526,684 bytes more across 22 shards. Conservative, and a rounding error at this scale, but it means
the tidy number from the analysis is not the number the code produces, and nobody should pin a
regression test to it.

The engine now refuses this configuration for the right reasons, or not at all.
