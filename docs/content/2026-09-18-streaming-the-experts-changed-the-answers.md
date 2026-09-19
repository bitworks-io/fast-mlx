---
title: "Streaming the experts changed the answers"
date: 2026-09-18
whitepaper_theme: Serving big models on Apple Silicon
status: published
---

# Streaming the experts changed the answers

We expected this card to be the easy one.

A mixture-of-experts model activates only a few of its experts for each token. If the machine cannot
hold all of them, a serving engine can keep the rest of the model in memory and read each expert
from SSD when the router asks for it. The router still chooses the same experts and the weights are
the same bytes. Only where they come from changes. That reasoning says expert streaming is
lossless by construction, so its quality card should read **EXACT**, and the only cost left to
state is speed.

We wrote that down as a plan, then measured it before publishing anything. The plan was wrong.

## What we ran

- **The pack:** a DeepSeek-V4-Flash GGUF with 4-bit (Q4_K) routed experts, about 165 GB.
- **The machine:** a 256 GB M3 Ultra, which can hold the whole pack resident, so the resident run is
  available as a reference.
- **The engine:** a third-party serving engine at a fixed revision, run two ways on the same file:
  fully resident, and with its SSD expert-streaming mode on. In streaming mode the expert cache was
  set to half the working set, smaller than the experts themselves, so the read-from-SSD path
  actually ran. The cache size stands in for a smaller Mac.
- **The test:** 40 prompts, 64 greedy tokens each, one request at a time. A prompt counts as
  identical only if its text and token count both match.

**Result: 25 of the 40 prompts came back different.**

The differences are not garbage. Here is one, at the point the two outputs part:

> resident: …provide a real-world **example** where a hash…
> streaming: …provide a real-world **scenario** where a has…

Both are reasonable continuations, and they are different answers.

## It is not noise, and it is not the cache

A single differing run proves little on its own, so we ran controls:

- **The resident engine is deterministic.** Four resident processes, in two sessions about an hour
  apart, produced byte-identical output. One of them sent the prompts in reverse order.
- **The streaming engine is deterministic too.** It just has a different answer. Five streaming
  processes all produced the same output, byte for byte, on every prompt. They covered two
  expert-cache sizes (about 7,300 and 10,500 expert slots), both prompt orders, and a run after the
  file cache had been flushed by an unrelated read.

So the output does not depend on which experts happen to be cached when a token needs them. A
different cache history gives the same streaming answer every time.

We tried to go one step further and run with a cache big enough to hold every expert, so that
nothing could be evicted. Even the larger setting had fewer slots than the model has experts
(10,496 against 11,008), so that attribution failed its own control. We do not claim it.

## Only short prompts diverged

Every one of the 25 divergences was on a prompt shorter than 64 tokens: 25 of the 32 short
prompts. All 8 prompts of 64 tokens or more were identical under streaming. We found this pattern in
the first run and predeclared it before the second, and it held exactly.

Our leading hypothesis is that the streaming mode takes a different prefill path for short prompts.
We have not confirmed it and do not claim a mechanism. For a chat workload the practical point is
the same either way: short turns are common.

## How big is the difference?

We do not know, and we will not guess. This engine returns no token probabilities in this mode, so
the measurement our other cards use cannot be taken here: how often the two setups disagree on the
most likely next token along one shared text. All we can measure is whether greedy decoding picks
the same words, and it often does not. The size of the quality change is **unmeasured**. It is
not zero.

## What it costs in speed

Streaming at the half-size cache ran at **0.49× the resident speed** end to end: 15.1 against
31.1 tokens per second per request.

A bigger cache was not faster. At the larger setting the whole corpus took 1,051 seconds against
485, because the cache crowded the rest of memory and swap grew. On this machine the cache size is
not a speed dial.

## What changes in fast-mlx

- **A fit check is a memory verdict.** Our GGUF fit checker can say a pack fits when its experts are
  streamed from SSD. It now says plainly that GREEN is a memory verdict, not a loadability verdict and
  not a quality verdict. "It fits if you stream it" is a different offer from "it runs the same."
- **No EXACT card for streaming.** The plan to issue one was withdrawn when the measurement came in.
- **No card at all yet.** Our cards are keyed to the pack. A card saying "this pack changes its
  answers" would also attach to the resident launch of the same pack, and that launch is the exact
  reference. The card waits until admission can tell a streaming launch from a resident one.
- **Streaming will not be recommended as free.** Our recommender does not offer streamed
  configurations today. When it does, each one will carry its measured cost. "Bigger than your Mac"
  is a trade, and the card store exists to state trades.

## How we ran it

- Every threshold and control was written down before either run.
- The first run was parked by its own rules, because two controls were voided:
  - A sanity check meant to reject degenerate output fired on one legitimate prompt, whose output
    was compact JSON-like text with almost no spaces.
  - A check that streaming had really engaged could not be satisfied at the engine's smallest cache
    setting.
- Both checks were corrected once, reviewed, and predeclared, and the second run passed every control
  the card depends on.
- The question is now closed.

The scope is narrow on purpose:

- one pack, one engine revision, one machine;
- greedy decoding, one request at a time, prompts under 100 tokens.

This is not a claim about expert streaming in general. It is evidence that streaming has to be
measured before anyone calls it exact.
