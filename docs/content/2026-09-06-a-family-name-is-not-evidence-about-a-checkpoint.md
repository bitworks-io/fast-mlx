---
title: "A family name is not evidence about a checkpoint"
date: 2026-09-06
whitepaper_theme: Serving big models on Apple Silicon
status: published
---

# A family name is not evidence about a checkpoint

fast-mlx keeps a served model's chain-of-thought out of the answer a caller sees. The OpenAI-shaped
completion carries two fields: `content`, which is the answer, and `reasoning_content`, which is
everything the model worked through to get there. Get that split wrong and one of two bad things
happens — the reasoning leaks into the answer, or the answer disappears into the reasoning. This
note is about hitting both failure modes in the same feature, in opposite directions, a week apart,
and why the fix that finally held is not the one that looked obvious.

## The bug that was real, and only half of it

A newly onboarded model family reasons the way its incumbent sibling does: it emits a block of
chain-of-thought, closes it with a literal `</think>` marker, and continues with the answer. On a
non-streaming request, fast-mlx's splitter found that marker, cut the response there, and returned
a clean pair — `content` held the answer, `reasoning_content` held everything before the marker.
That worked with no changes at all.

On the streaming path it did not. A streaming request to the same model, same prompt, produced zero
`reasoning_content` deltas. The reasoning went straight out as `content`: the first deltas on the
wire were `{"content":"We"}`, `{"content":" need"}` — the opening words of the model's internal
reasoning, sent to the caller as if they were the answer. The literal `</think>` marker itself later
appeared as visible text in the stream, which is the plainest possible evidence that nothing was
splitting anything.

The two paths disagreed because they are gated differently. The non-streaming splitter is
marker-driven: it looks for `</think>` in the completed text and cuts there, regardless of which
model produced it. The streaming path can't wait for the whole response before deciding whether to
split, so it decides up front, before the first token, based on an explicit per-family
attestation — a list of model families the engine has verified separate cleanly. The new family
was not on that list. Correctly not, in the sense that nobody had yet verified it belonged there —
but the consequence was that every streamed response from it exposed raw reasoning to the caller,
including the marker the splitter would otherwise have consumed.

## The bug that was recorded, and was not there

A week earlier, this same behavior had been logged as a broader defect: chain-of-thought reaching
`content` on this model's serving route, full stop, no path qualifier. That record was wrong, and
the way it was wrong is the more useful part of this note.

The run that produced it capped generation at 48 tokens. At that budget, the model never got far
enough to emit `</think>` at all — there was no close marker anywhere in the output. Since the
non-streaming splitter cuts on marker presence, and there was no marker, there was nothing for it to
split on. The recorded "defect" was two different things wearing one description: a real,
streaming-only gating bug, and a fabricated non-streaming bug that was actually just a sample too
short to contain the evidence needed to observe correct behavior. Raising the token budget made the
non-streaming case pass with no code touched.

Neither correction was safe to skip. Treating the truncation artifact as real would have spent effort
patching a splitter that already worked. Treating the real gating bug as covered by that same
description would have left the actual defect — the one visibly leaking reasoning to every streaming
caller — unfixed while the wrong line item got closed. A truncated sample can manufacture a defect
that was never there, and it can do that convincingly enough to hide the one that was.

## Why the fix is not a name on a list

The mechanical fix for the streaming gap looks like adding the new family's identifier to the
attested set, and stopping there. That was deliberately not what shipped.

The attestation exists because the streaming splitter has to commit to a strategy before it has seen
any output — it cannot look at the finished text the way the non-streaming path can. A bare set of
family names encodes a claim about behavior ("this family reasons and closes with these markers")
that is true today and has no mechanism keeping it true tomorrow. A future checkpoint could ship
under the same family identifier with a chat template that never emits the closing marker at all —
a retrained or restyled release, still reporting the same family, with different reasoning
formatting. Under a name-only gate, every one of that checkpoint's answers would be captured into a
reasoning block that never closes. `content` would come back empty. That is worse than the bug being
fixed: a visible answer with leaked reasoning in front of it is at least a visible answer. An empty
one is silent data loss, and it would pass every check that only asks "is this family on the list."

So the gate that shipped is a three-way conjunction, not a membership test: the request has to be on
the route that performs the split, the model's reported family has to be on the attested set, *and*
the checkpoint actually loaded has to have a chat template that contains the markers the splitter
hardcodes. That third term is the one doing the real work — it reads what the tokenizer's own chat
template renders and refuses to split unless the markers it depends on are actually present. There
is no default for it; a caller that omits the check does not silently get `true`, it fails to
compile. An unreadable or non-attesting template means unchanged passthrough behavior, never a
split — the same fail-closed direction as every other gate in this engine.

Before relying on the new conjunction for anything, it had to be checked against the checkpoint
already shipping. The incumbent's chat template carries the identical marker profile the new family
uses, so the three-way gate does not regress the model already in production — verified, not assumed.

## A second behavior moved with it, and that also got measured

Admitting the new family into the attested set flips a second switch that happens to be defined as
its exact negation: a legacy workaround that disables model "thinking" whenever a tool is attached
to the request, put in place before the engine could reliably separate reasoning from tool calls.
Turning the family's reasoning-split on also turns that workaround off for it — which means thinking
and tool-calling now run together for the first time on this family, and that combination had never
been exercised.

It was not left as an assumption. With thinking left on and a tool attached, the model still produced
a well-formed tool call end to end, on both streaming and non-streaming requests, with `finish_reason`
correctly reported as a tool call and `reasoning_content` populated alongside it. The two behaviors
are coupled on purpose, not by accident of implementation, and both directions are pinned by tests.

## What it reads now

Same model, same prompt, streaming, after the fix:

```
0 -> 100 reasoning_content deltas
then content deltas: "7", "pm"
0 occurrences of the close marker in any content delta
```

Streaming went from zero reasoning deltas and a leaked marker to a hundred reasoning deltas followed
by a clean answer, with nothing from the reasoning block visible in `content`.

The fail-closed contract from the truncated case did not go away, and it should not: a generation
that is cut off before the model reaches its own closing marker still returns `content=null` with a
partial `reasoning_content` and `finish_reason` marked as length-limited. That is not a bug fixed by
this change — it is the documented behavior for an incomplete generation, on this family and on the
one already shipping. An operator who truncates a reasoning model's output too aggressively gets an
empty answer either way, and that is the contract to design around, not a gap to close quietly.

The lesson that generalizes past this one feature: a family identifier is a claim about what a
checkpoint tends to do, made at some point in the past, about some other artifact that happened to
carry the same name. It is not evidence about the checkpoint actually loaded right now. Where a gate
can instead check the loaded artifact directly — the rendered chat template, in this case — checking
it costs little and closes a failure mode that a name-only gate cannot see coming: the same name,
shipping different behavior, on a future day.
