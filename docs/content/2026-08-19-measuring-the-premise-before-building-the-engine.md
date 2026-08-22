---
title: Measuring the premise before building the engine — does MoE routing actually cluster?
date: 2026-08-19
whitepaper_theme: Running models larger than memory — the spike that gates the bet
status: draft
---

# Measuring the premise before building the engine

"Run models larger than your RAM" is a seductive line to put on a box. For a Mixture-of-Experts model
it's even plausible: only a handful of the experts fire on any given token, so — the pitch goes — keep
the hot ones in memory, page the cold ones from SSD, and serve a model whose full weights never fit.
The catch is the failure mode this project exists to refuse: a "larger than memory" claim that in
practice means *fractions of a token per second*. That's the trust-burning wrong number. So before
writing a line of the paging engine — a months-scale, fork-level build — we ran a bounded spike to
answer one question with a number: **does real MoE routing cluster enough for paging to be fast?**

## The bet has exactly two unknowns

Model the paging cost honestly and it collapses to two measurable quantities. First, **SSD random-read
bandwidth** at the size of one expert — because a cold expert is a random read, not a sequential one,
and datasheet "up to X GB/s" numbers are sequential lies. Second, **locality**: of the experts a token
needs, how many were already resident? Multiply the miss count by the expert size and divide by the
bandwidth and you have bytes-per-token over GB-per-second — a tok/s ceiling. Everything else is detail.

We refused to build the instrument on faith, so we built it test-first. A pure residency simulator —
an LRU cache per layer, misses counted against what was resident when the token arrived — pinned by
seventeen unit tests against hand-checkable synthetic traces: a sticky stream that should miss only
once, a cache smaller than the token's own top-k that should miss forever, a uniform-random stream
whose miss rate must land on the analytic null. Writing those tests first caught a real bug in the
frequency counter before it could ever skew a verdict. The instrument that produces the number gets
measured before the number does.

## Bandwidth first, because it might have ended the story

We measured this box's SSD the honest way: 2.5 MB random reads (one 4-bit expert) under concurrency,
with the OS buffer cache defeated (`F_NOCACHE`) so the number is the device and not RAM. It came back
at **~5.5 GB/s**, and — the tell that it's real — it held steady as we grew the backing file past any
plausible residual cache. A cache hit would have scaled with file size; this didn't.

That one number reframed the whole spike. At 5.5 GB/s, even a *pessimistic* locality result — say, a
model that misses enough experts to move 400 MB/token — still models to ~14 tok/s. Bandwidth is not
the bottleneck on this machine. The entire go/no-go now rides on locality, which is exactly the thing
nobody can know from a datasheet and everybody assumes. So we went and measured it.

## The premise, tested on a real model

We instrumented a real MoE checkpoint's forward pass and recorded, per layer per decode token,
precisely which experts it routed to. The trick that makes this trustworthy: the model already hands
the block its chosen expert indices as an argument on the way to the expert matmul, so we capture *that
argument* rather than re-deriving the gating math — no chance of our measurement drifting from what the
model actually did. Then we replayed those traces through the residency simulator.

On a small MoE (a proxy for the target geometry), over a few hundred decode tokens, keeping **half**
the experts resident:

- **81% of expert activations were already in memory** — a miss rate of 24 per token against a
  uniform-random model's 64. Real routing is **2.6× better than chance**. Keep 75% resident and the hit
  rate climbs to 94%.

Routing clusters. The premise the entire bet rests on is now a measured fact on a real model, not an
architectural hope. And a second finding shaped the eventual build before we've written it: a *static*
pinned hot set — the cheap thing, just keep the globally-most-popular experts — missed more than twice
as often as the adaptive LRU. The locality is *temporal*, not merely frequency-skewed. If this gets
built, it needs a real online cache, not a frequency table. Cheaper to learn that from a trace than
from a rewrite.

## What this is, and what it isn't

The honest footnote, written into the record the same day: this is not yet a PROMOTE verdict. The
proxy model shares the target's routing shape but not its expert *size*, so it tells us locality is
strong and adaptive — the direction of the signal — but not the target model's cost in megabytes per
token. That number needs the real target checkpoint, and running a model that large on this box is its
own risk to manage carefully rather than fire off unattended. What the spike *has* done is retire the
two ways this bet dies quietly: bandwidth isn't the wall (measured), and routing isn't random
(measured). The expensive question — build it or shelve it — is now down to one clean number on the
right model, with a validated instrument already waiting to compute it.

That's the whole point of a spike. Not to build the engine, and not to guess whether it would work —
but to spend a day making the guess unnecessary. You measure the premise before you build the thing
that assumes it, because the cheapest place to discover that routing doesn't cluster is a trace file,
and the most expensive place is a shipped feature that runs at half a token per second.
