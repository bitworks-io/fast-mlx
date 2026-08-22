---
title: "The state was right. The ledger was frozen."
date: 2026-08-20
whitepaper_theme: Serving big models on Apple Silicon
status: draft
---

# The state was right. The ledger was frozen.

Continuous batching is how one model serves many users at once: several requests share a single
forward pass, their per-request caches stacked side by side, and the engine merges, splits, and
re-merges those rows as requests come and go. For a plain transformer every row is a growing KV
cache, and the arithmetic is uniform. For a **hybrid** model — Qwen3.5's shape, where most layers
hold a small fixed recurrent state and only every fourth layer grows a KV cache — the rows are two
different animals in the same array, and the seams between them are where the bugs live.

This note is about a bug that hid in one of those seams for a full increment, survived a green test
gate, and was caught only when we made the gate ask the harder question. The interesting part is
*why* it hid: the model's math was correct the entire time. What was wrong was the engine's own
bookkeeping about that math.

## Two clocks, and only one of them ran

Every request in flight carries a position — how many tokens it has committed so far. For a dense KV
row that count advances *inside the model's forward pass*: the cache's write frontier moves as keys
and values are appended, in-graph, automatically. Nothing external has to remember it.

A recurrent row has no such frontier. Its state is fixed-size — a conv tail and an SSM state that get
*overwritten* each step, not grown — so the model never advances a logical token count on it. That
count is the *driver's* responsibility. The engine has to look at the row after each step and write
down: this recurrent state now represents N committed tokens.

We knew this. It's written down as "finding 3" in the design: the vendored `ArraysCache.advance`
touches only transient lengths and left-padding; nothing increments the recurrent cache's logical
offset, so the continuous-batch driver must maintain it. And we implemented it — at prefill. The
commit point where a fresh request's prompt is absorbed calls `commitRecurrentRowOffsets` and records
the count honestly.

Then decode ran, and the count stopped moving. Not the *state* — the state advanced perfectly. Every
decode step, the model's write-backs landed on the authoritative `MambaCache` (the same object the
model reaches by a concrete `cache as? MambaCache` downcast), so the SSM position clock inside the
recurrent state ticked forward exactly as it should. A request decoding alone produced flawless
tokens. What did *not* advance was the driver's separate ledger — `BatchedRecurrentStateCache`'s
per-row `logicalOffsets`, and the scalar row's own offset. Prefill wrote "3" there and nothing ever
wrote again. `logicalOffsets` didn't even have a mutator; it was `private(set)` with no way to move
it.

Two clocks for the same request: the real one, buried in the recurrent state, running correctly; and
the driver's copy of it, frozen at the prefill value. As long as nobody compared them, everything
looked fine.

## The only moment anyone checks

The engine compares the two clocks in exactly one place: `validateCacheLengths`, which asserts every
row's logical offset equals the request's committed-token count. And it runs that check only at a
**membership transition** — when a request *spills* out of a batch to decode solo (the batch shrank to
one), or *rejoins* a batch from solo (`ensureScalar` / `ensureBatch`). A request that decodes solo
forever, or a batch that stays exactly the same set of requests forever, never triggers it.

That is precisely why the bug survived. The first parity test we shipped drove a single request
through chunked prefill and solo decode and asserted exact tokens — green, and legitimately so; the
state clock was correct. But solo decode never changes membership, so it never asks the driver to
reconcile its frozen ledger. The gate *looked* complete. It was quietly missing the two assertions
that are the entire point of continuous batching: that a batch of two emits exactly what each request
would emit alone, and that a request's recurrent state round-trips bit-identically across a
split and a re-merge.

## Making the gate ask

So we wrote those two. Both use a two-layer toy — layer 0 a real recurrence over a `MambaCache`,
layer 1 real KV attention — with closed-form solo oracles computed by hand, not by running a second
in-process path that could share the same bug: request A on prompt `[1,2,3]` must emit
`[16, 35, 73, 149]`, request B on `[5,6]` must emit `[18, 39, 81, 165]`.

- **Lockstep + spill:** batch A and B, decode two steps in lockstep (assert each emits its own solo
  sequence), then drop B and continue A solo. Dropping B forces A to spill from the batched state
  back to a scalar row.
- **Solo → rejoin:** decode A and B solo, drain the mandatory lookahead step, then batch them.
  Rejoining forces each scalar row back into a merged batched state.

Both failed on the first run — and failed *identically*, with the exact error the root-cause
predicted: `cacheLengthMismatch(expected: 5, actual: 3)`. Expected five committed tokens; the
recurrent row's ledger still read three, the prefill value, frozen while the state clock had moved on
to five. Two different transition paths, same frozen number, same diagnostic. That is what a real
root cause looks like: it tells you where it will break before you break it.

The fix is the other half of finding 3, and it is small because the diagnosis was exact. Advance the
scalar row's offset after each solo step; advance the batched rows' offsets in lockstep after each
batched step, through a new fail-closed `advanceOffsets(by:)` mutator that refuses a non-positive
delta. Both paths are no-ops for uniform dense models — guarded on whether the cache family is hybrid
at all — so the dense continuous-batch path stays byte-for-byte what it was, and its fifty-three
regression tests never moved. Red to green, and the two clocks read the same number again.

## The point

The state was never wrong. If we had only ever measured tokens on paths that don't change membership,
we would have shipped this and believed it correct — because on those paths it *is* correct. The
defect lived entirely in the engine's private accounting of a state it was otherwise maintaining
perfectly, and it was reachable only through the exact operations — split a batch, merge a batch —
that a single-user test never performs and a multi-user server performs constantly.

A green gate that omits its hard transitions is a false green. The value wasn't in the fix, which was
a dozen lines; it was in refusing to call the gate done until it exercised the split and the re-merge,
and in writing the oracle by hand so the test couldn't agree with a broken engine. The engine is
allowed to be correct on the easy path and wrong on the hard one. The job of the gate is to make sure
you find out which.

---

*Scope: findings are specific to the fast-mlx hybrid continuous-batch runtime
(`DenseContinuousBatchRuntime.swift`, `BatchedRecurrentStateCache.swift`) and its Qwen3.5-shaped
toy-parity gate (`HybridContinuousBatchRuntimeTests.swift`), on `codex/absorbed-mla`, 2026-08-20.
The toy oracles are closed-form fixtures, not a real checkpoint; real-model (T1) and multi-stream
throughput (T2) parity remain gated on larger hardware. This note is historical engineering
narrative, not a current performance claim.*
