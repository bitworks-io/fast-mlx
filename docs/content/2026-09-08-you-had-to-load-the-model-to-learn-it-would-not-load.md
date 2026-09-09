---
title: "You had to load the model to learn it wouldn't load"
date: 2026-09-08
whitepaper_theme: Serving big models on Apple Silicon
status: published
---

# You had to load the model to learn it wouldn't load

An earlier note — [*The fit-check refused a model that fit*](2026-09-06-the-fit-check-refused-a-model-that-fit.md)
— was about the fit-check getting an answer wrong. This one is about not being able to ask it the
question.

We were planning a model swap on a machine already serving traffic, and we needed one number:
what context would the new model get on *that* box? The fit-check computes exactly that. It reads
the checkpoint's `config.json` and the headers of its safetensors shards, models the weights, the
KV cache and the construction peak against the host's ceiling, and prints a verdict. It allocates
nothing. It is pure arithmetic over metadata.

And there was no way to run it on its own.

## Pure, and unreachable

The function was already clean. The problem was where it was called from: two sites, both inside
the routine that prepares the serving backend, each one immediately followed by the weight load.
The verdict was computed and then consumed, in the same breath, by the thing it was supposed to
gate.

So the only way to see the number was to start a real serve — on a checkpoint of a hundred-odd
gigabytes, on a box carrying live traffic. You had to pay for the load to find out whether the
load would work.

There were two dry-run paths already, and neither one fit. One picks the best quantization from
several candidate directories; it structurally cannot take a single model path. The other is a
capacity tool that works from a catalogue of known architectures, and the architecture we cared
about was not in it. Both existed. Neither answered.

This is a shape worth naming, because it does not look like a bug. Every piece was correct and
tested. The computation was pure, the callers were right to call it, and nothing was broken. The
capability was simply not *reachable* from outside — and code review does not flag "this is only
callable in a context that also costs you a hundred gigabytes."

## The tempting fix is the wrong one

The obvious implementation is a new mode near the top: parse the arguments, work out the model
directory, compute the fit, print it, exit. Clean, self-contained, easy to test.

It is also how you build a number that quietly stops matching reality.

The moment that dry run computes its *own* fit, it becomes a second implementation of the same
question — and the two can drift. Worse, they can drift silently and in the dangerous direction.
We had a live example to learn from: the existing quantization dry run rebuilds its arguments
field by field, and in doing so drops a couple of the serving flags. That is fine for what it
does. But one of the flags it drops adjusts the layer count used for KV sizing. A dry run that
inherited that shape would have reported a *smaller* memory requirement than the real serve would
use. It would have failed optimistically — printing green for a configuration that then OOMs.

So the flag does not compute anything. It stops at the two call sites that already exist,
immediately before the load, and reports the verdict that call just produced. One computation,
reported rather than consumed. If the serve's sizing changes, the dry run's answer changes with
it, because it is not a copy — it is the same number.

## A dry run must not carry the flag that silences it

The serve has a `--force` flag, which exists to proceed past a red verdict when an operator knows
better than the model. It is a reasonable escape hatch.

Combined with a fit-check dry run, it is a trap. `--force` suppresses the refusal — so a red host,
asked "do you fit?", would answer by exiting successfully and printing nothing alarming. The flag
whose entire purpose is to *learn* the verdict would be carrying the flag whose purpose is to
*ignore* it.

We refuse the combination outright. Same for combining it with the other dry run (ambiguous), and
with the transport-only scripted mode (no model to check).

## Saying what you have not proven

One more piece, and it is the part we would most likely have skipped a year ago.

When the offload plan is in play, the fit path reads exactly one field out of that plan: the
declared residency budget for the row store. It does not open the row store. It does not check the
plan's integrity records, and it does not consult the approval artifact that authorizes serving
from it. All of that is resolved later, on the load path.

Which means a green verdict here proves the *arithmetic* fits. It does not prove the host is
provisioned to serve. Those are different claims, and the gap between them is exactly wide enough
for an operator to walk into a refusal during a change window, holding a green report.

So when that composition applies, the report now says `offload_path_resolvable=unproven`. It is an
ugly token and it makes the output longer. It is also the single most useful thing on the line,
because it is the one thing the number cannot tell you about itself.

## What it cost, honestly

We should be straight about the size of this. It is not new capability. A green verdict was
*already* obtainable, by starting the serve and killing it in the window between the fit block
printing and the weights beginning to load. It worked. People do this.

What changed is that a race against a process you are trying to kill became a contract: exit 0 with
a machine-readable line on a verdict that proceeds, exit 2 with the refusal reason on one that does
not. "Kill it fast enough" is not a procedure you want in a runbook for a production change window.

That is the whole improvement. It is worth doing and it is not worth overclaiming — and telling
those two apart is most of what an engineering note is for.

## The general shape

If you have a pure function whose only callers are the things it gates, you do not have a tool. You
have an internal detail that happens to be well factored. The distance between those is one seam,
and you usually discover which one you have at the worst possible moment — when someone needs the
answer and the only way to get it is to do the expensive thing the answer was supposed to prevent.
