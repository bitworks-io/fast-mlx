---
title: "Every tool call took the path we never tested"
date: 2026-09-08
whitepaper_theme: Serving big models on Apple Silicon
status: published
---

# Every tool call took the path we never tested

We had a test for tool calling. We had a lot of tests for speculative decoding. We had no test for
tool calling *with* speculative decoding — and because speculation is bound when the server loads
the model rather than negotiated per request, that meant **100% of tool calls on a speculative
deployment took a path nothing had ever exercised.**

The coverage looked fine from either end. That is the part worth writing down.

## Two axes, each covered, product untested

Speculative decoding on this serving path is not a per-request option. The flag that enables it
constructs a decoder that owns a target model and a draft model together, once, at load. There is no
branch that says *this request carries tools, decode it the ordinary way.* Every request gets the
speculative decoder, or none does.

Our tool-calling proof, meanwhile, had been run on a server started **without** that flag. It was a
real proof — real weights, real tool call, streaming and non-streaming — and it proved the
non-speculative route. Read the two records side by side and it is easy to conclude tool calling is
covered and speculation is covered. Both statements are true. Neither is the statement that matters.

This is a shape we have now been bitten by more than once, so it has a name in our notes: **the
intersection of two covered axes is uncovered.** Two knobs, each individually tested, does not mean
their product is tested — and the interesting bugs live in the product, because that is where two
sets of assumptions meet without either author present.

The same gap existed one level down. We had a test rendering a chat request with tools and a leading
system message. We had a test rendering a non-leading system message with no tools. The shape an
agent framework actually sends — tools attached *and* a system turn injected mid-conversation,
carrying a tool policy — was covered by neither.

## A green light that means nothing

The obvious way to close this is to start a server with speculation on, send a tool call, and check
you get one back. We did that. All nine cells returned HTTP 200.

That result, on its own, would have been worthless.

Speculative decoding on this path is allowed to decline. If the drafter can't support the request —
most commonly because the request samples at a nonzero temperature and this drafter is greedy-only —
the decoder transparently falls back to ordinary decoding and records why. Correct behaviour, and
invisible from the outside: the response looks identical. So a run where speculation quietly declined
every single tool call would produce exactly the same nine green cells, and would prove precisely
nothing about the combination we were trying to qualify.

So the gate was built to be falsifiable in two directions:

- **Prove the feature was actually running.** Every cell's per-request telemetry was read for
  accepted draft tokens. The tool cells accepted 18 of 18 proposed tokens, 40 of 46, and 36 of 36.
  Speculation was not merely reachable; it was doing the work.
- **Prove the feature changed nothing.** Every cell was run a second time against a control server
  with speculation off, and the two runs were compared on parsed function name and arguments — not
  on "did we get a tool call", but on *the same* tool call. All of them matched.

Plus one cell whose job is to fail: tools attached to a question needing none. If the tool parser
over-fires when text arrives in different-sized pieces — which is exactly what speculative acceptance
can change — that cell manufactures a tool call nobody asked for. It didn't.

Only then does the green mean something. Tool calling under speculation works: streaming and not,
thinking on and off, two calls in parallel, across a full tool-result round trip, and with the
mid-conversation system message that motivated the whole exercise.

## The gate caught the instrument lying

Building it that way had an unplanned payoff.

Reading the per-request telemetry rather than just the HTTP status turned up rows that contradicted
themselves: requests reporting *"generating without speculation"* while also reporting 36 of 36 draft
tokens accepted. Both cannot be true.

The cause is one field. The per-request summary differences its counters against a snapshot taken
before the request — so the counts are genuinely per-request — but it reads the fallback *reason*
absolutely, out of a cumulative field that is deliberately sticky and never cleared. On a serving
decoder, which is built once and reused for every request the process ever handles, the first request
that legitimately declines speculation poisons that field for the lifetime of the process. Every
later request is then labelled unaccelerated while it is in fact accelerating.

The comment directly above the code states the invariant it breaks, in as many words: the delta
exists so the summary reflects only what *this* request contributed, not the lifetime total. One
field in the struct didn't get the message.

Nothing generated is affected — no token, no tool call, no completion. What's affected is an
operator's ability to answer "is speculation actually running?", which fails in the more dangerous
direction: it reports *not running* for requests that are. An operator chasing latency would chase a
ghost, and a future engineer reading those logs could reasonably conclude the feature is inert.

The fix that first comes to mind is wrong, which is worth saying out loud: comparing the reason
before and after the request looks like the natural fix and quietly breaks the case where two
consecutive requests both decline for the same reason — they compare equal, and the second gets
reported as accelerated. The per-request truth is the current decode's own reason, with the
cumulative fallback removed from that read and left intact for the callers that legitimately want it.

## What we'd take from this

A pass/fail gate would have returned nine green cells and told us nothing — not that speculation ran,
not that it changed nothing, and certainly not that the field we'd have quoted as evidence was
inherited from an earlier request. The gate found a real defect because it was built to measure the
quantity that discriminates rather than the quantity that is easy to observe.

And the reason it existed at all is a question worth asking of any default: *what fraction of
production traffic takes this path, and what fraction of our tests do?* When the answer is 100% and
0%, the feature is not covered no matter how green the suites are on either side of it.
