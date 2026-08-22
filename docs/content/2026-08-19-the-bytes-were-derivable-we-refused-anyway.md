---
title: "The bytes were derivable. We refused anyway."
date: 2026-08-19
whitepaper_theme: Serving big models on Apple Silicon
status: draft
---

# The bytes were derivable. We refused anyway.

An earlier note — [*One formula, wrong for a third of the catalog*](2026-07-09-one-formula-wrong-for-a-third-of-the-catalog.md)
— described the fit-check: the number the engine computes *before* it loads a model, answering the
one operator question that matters on a fixed-memory Mac — *will this context window actually hold,
or will it OOM three thousand tokens in?* The whole design rule was: **fail toward RED. When you
can't model a model honestly, refuse to load it — never print a green "fits" you can't stand behind.**

This week two hybrid architectures arrived at the frontier of what the fit-check can size. They make
a clean pair, because one we now model exactly, and the other we deliberately still refuse — and the
refusal is the more interesting half.

## Falcon-H1: the parallel hybrid

The hybrids we already supported are *interval-select*: in Qwen3.6 or Jamba, only some layers run
full attention and grow a KV cache; the rest hold a small, fixed recurrent state. Count the attention
layers, size their growth, add the fixed term once for the linear layers. Done.

Falcon-H1 breaks that shape. Reading the vendored Swift arch (`FalconH1.swift`, the cache factory at
lines 799–801), every single layer is handed *both* a growing attention cache **and** a Mamba-2
conv+SSM recurrent state. It's a **parallel** hybrid: nothing is either/or. So the sizer has to do
something it never did before — count the growing term over *all* layers, and add the fixed recurrent
term over *all* layers too, not over "the linear remainder" (which here is zero).

The recurrent term itself needed a source audit, not a guess. Two details a blog-post formula would
get wrong:

- The **conv state** width isn't `mamba_expand × hidden`. In Falcon-H1 the inner dimension *is*
  `mamba_d_ssm` (the arch sets `intermediateSize = args.mambaDSSM` directly). On the real
  Falcon-H1-34B that's the difference between 4,096 and 10,240 — a 2.5× error hiding in a variable
  name.
- The **SSM state** is stored at *activation* precision (2 bytes), not fp32. We only claim that
  because the arch proves it: `SSM.swift` sets `stateType = state?.dtype ?? x.dtype` and casts the
  next state to it every step, so the slot follows the compute dtype forever. Where a sibling family
  (Jamba) left that dtype unaudited, we used the conservative 4-byte width instead. You earn the
  smaller number by reading the code.

The result: on Falcon-H1-34B (72 layers) the fixed recurrent state is **146 MiB per sequence** —
small next to the growing KV, but not zero, and now counted. And one more refusal baked in: if a
checkpoint omits `mamba_d_ssm`, we **throw**, we don't fall back to the vendored default of 1536. A
model that relied on that default would otherwise be sized against a guessed inner dimension — a
phantom number wearing a real number's clothes.

## Baichuan-M1: the one we could size but wouldn't

Then Baichuan-M1. We audited it the same way — vendored `BaichuanM1.swift`, real published
`config.json` — and here is the uncomfortable finding: **the KV bytes are fully derivable from the
config.** Every field we need is present and unambiguous. We could print a number today.

We refuse to.

The reason is representational, not epistemic. Baichuan-M1 has **dual head geometry**. Its global
attention layers use one head shape (2 KV heads × 256 dims → a product of 512); its sliding-window
layers use a *different* one (`num_swa_key_value_heads` / `num_swa_attention_heads` → 8 × 128 → a
product of 1024). Same model, two head geometries, twice over. The vendored arch reads a different
head count depending on which layer class it's building.

Our capacity model, like almost every KV-memory model in the wild, carries **one** `n_kv_heads` and
**one** `head_dim`, and multiplies that single product into both the growing term and the
window-capped term. For Baichuan-M1 no single product is correct for both:

- Use the **global** geometry, and the sliding-window cache is under-counted by **~320 MiB per
  sequence** — and that's the *phantom-green* direction, multiplied by every concurrent stream. This
  is precisely the failure the fit-check exists to prevent.
- Use the **sliding-window** geometry instead, and you 2× *over*-count the dominant growing term —
  safe, but now the instrument is lying in the other direction, doubling the flagship number an
  operator reads.

There is a correct answer — it just isn't a scalar. Sizing Baichuan-M1 honestly requires teaching
the profile to hold *two* geometries and the capacity model to spend each in the right term. That's a
real change with real blast radius (every consumer of "the" head count), so it gets its own careful
pass, with the exact formula and the real numbers written down first. Until that lands, Baichuan-M1
stays refused — not because we don't know its size, but because our *representation* can't state it
without lying in one direction or the other.

## The point

It would have been easy to ship a Baichuan-M1 number. It parses; it looks plausible; nobody would
notice until a 32K context quietly OOM'd a machine that the tool had called safe. "We can compute
*a* number" is not the same as "we can compute *the* number," and an instrument that rounds the
difference away is worse than no instrument, because people trust it.

Two hybrids, one commit apart: one we now size to the byte, one we can size but won't. The engine is
allowed to say *I don't model this honestly yet.* That sentence is the whole product.

---

*Scope: findings are specific to the vendored MLX-Swift architectures (`FalconH1.swift`,
`BaichuanM1.swift`) and the published `tiiuae/Falcon-H1-34B-Instruct` and
`baichuan-inc/Baichuan-M1-14B-Instruct` configs audited on 2026-08-19. Byte figures are per-sequence
model estimates from those configs, not a universal guarantee. This note is historical and not a
current performance claim.*
