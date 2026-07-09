---
title: "One formula, wrong for a third of the catalog: a KV-memory model that refuses to lie"
date: 2026-07-09
whitepaper_theme: Serving big models on Apple Silicon
status: draft
---

# One formula, wrong for a third of the catalog

We wanted the engine to answer a simple operator question: *before* you raise a model's context
window from 32K toward its maximum, will **this** Mac actually hold it? Answering it needs one
number — how many bytes of KV cache a model burns per token — and there is a textbook formula for
it, printed in every inference blog post:

```
kv_bytes_per_token = n_layers × n_kv_heads × head_dim × 2 (K and V) × bytes_per_element
```

We almost shipped it. Then we checked it against the fourteen models in our catalog — their real
`config.json` files, and the actual vendored Swift architecture code that caches their K/V. The
formula was **wrong for five of them. By 4× to 71×.**

## The formula assumes an architecture most modern models abandoned

The formula bakes in one assumption: *every layer runs full attention and grows its cache every
token.* That was true in 2023. It is false for a third of what people actually serve now.

- **Hybrid-linear attention** (Qwen3.6-35B, Ornith-397B): only one layer in four is real
  attention. The other three are GatedDeltaNet layers with a **fixed-size** recurrent state that
  doesn't grow with context at all. The vendored arch code proves it — its cache factory hands
  back a `MambaCache` for the linear layers and a growing cache only for the 1-in-4 attention
  layers. The naive formula over-counts by **4×**.
- **Interleaved sliding-window** (Gemma-3-27B): five of every six layers are local, capped at a
  1024-token window — their cache stops growing almost immediately. Only the global layers grow
  with full context. Over-counted by **~6×**.
- **Mamba2 hybrids** (Nemotron-3-Ultra): most layers are SSM state (O(1)) or pure MoE feed-forward
  with no attention to cache at all.

And then the one that breaks the other direction — the formula catastrophically *under*-counts:

- **MLA, as actually implemented** (DeepSeek-R1): everyone "knows" Multi-head Latent Attention has
  a tiny KV cache — that's its whole selling point, a compact latent of ~2 GiB at 32K. But the
  inherited Swift port (a faithful port of the reference `mlx-lm`) **decompresses** K and V back to
  full per-head form *before* writing the cache. So R1, as this engine caches it today, stores
  **4.88 MiB per token — 152.5 GiB of KV at 32K context.** Not 2 GiB. The famous MLA savings are a
  property of an *absorbed*-cache implementation that nobody has shipped here. Trust the textbook
  MLA number and you're off by **71×** — and you'd cheerfully promise a context length that OOMs a
  512GB machine.

## The fix: dispatch the formula, don't unify it

The correction is the same shape as what the arch code already does. It dispatches cache *type*
per layer; our memory model dispatches the *formula* per architecture class — uniform-GQA,
hybrid-linear, interleaved-SWA, MLA-as-implemented, Mamba2-hybrid. Each class computes what its
layers actually cache.

We wrote it as pure, dependency-free logic and pinned every branch to a real configuration:
Qwen3-30B reconciles to **3.0 GiB** at 32K, Qwen3.6-35B to **0.625 GiB**, Gemma-3-27B to **2.91
GiB** (global growth plus the fixed local-window term), Qwen3-32B to **8.0 GiB**, DeepSeek-R1 to
**152.5 GiB**. The test that matters most doesn't check a number — it collapses the dispatch back to
the naive formula and asserts the hybrid/SWA/MLA cases *break* by exactly the expected 300–450%.
If the class label stops changing the math, the test fails.

## The harder half: refusing to answer

Two models had **unconfirmed** geometry. Nemotron-3-Ultra's count of growing attention layers
isn't pinned down in any config we could read; DeepSeek-V4-Flash uses a novel compressed attention
that isn't MLX-servable at all yet. The tempting engineering move is to plug in a plausible
placeholder and emit a number — a dashboard hates a blank cell.

We did the opposite. The model carries an `isKVDerivable` flag, and the capacity classifier checks
it *first*: for a model whose KV cost we can't honestly derive, it returns `.kvNotDerivable` —
not green, not a guess. Because the entire value of a capacity advisor is trust, and nothing
destroys that faster than a confident under-count that paints a 275GB model as a comfortable fit.
"I can't derive this — confirm the config" is a more useful answer than a number that's silently
one attention-layer's worth of a much larger truth.

The same discipline caught a smaller thing on the way: our own spec had written R1's per-token cost
as "4.88 MiB." The honest geometry (128 heads × (64+128+128) dims × 2 bytes × 61 layers) is 4.766
MiB — the 4.88 was a KiB-vs-KB rounding slip in our prose. The implementer flagged it and refused
to bend the geometry to match the document. The geometry reproduces 152.5 GiB at 32K to the byte;
the prose was what was wrong.

One last inversion worth naming: **Phi-4's maximum context (16K) sits *below* the 32K default** we
wanted to ship. "Default 32K, raise toward the model's max" quietly assumes every model clears 32K.
One doesn't. The effective default has to be `min(32K, model_max)`, or the very first thing the
feature does is over-promise.

## The lesson

The formula everyone copies encodes a 2023 architecture, and the models worth serving in 2026 have
moved on in five different directions — four that make the cache *smaller* than the formula claims,
one that makes it 71× *larger* than the theory promises. You only find that by checking the formula
against real configs and the code that actually allocates the cache, one architecture at a time.
And when you build the thing whose job is to tell an operator the truth about their hardware, its
integrity lives as much in the cells it leaves blank as in the numbers it fills — a planning tool
that can't say "I don't know" will eventually, confidently, be wrong.
