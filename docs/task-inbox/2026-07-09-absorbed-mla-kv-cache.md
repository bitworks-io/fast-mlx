# Absorbed-MLA KV cache for DeepSeek-class models (71× KV reduction)

- **Captured:** 2026-07-09
- **Status:** backlog (not for immediate implementation)
- **Task type:** engine capability / KV-memory optimization
- **Priority:** high-leverage, but gated (own design/plan required)
- **Owner project:** fast-mlx engine

## Raw finding (from the system-aware-context-operability research, 2026-07-09)

The vendored `mlx-swift-lm` port (`DeepseekV3.swift`, a direct port of `mlx-lm`'s `deepseek_v3.py`) **decompresses K/V via `kv_b_proj` *before* writing the cache**, so the cache stores full per-head K/V (128 heads × (qk_rope 64 + qk_nope 128 + v 128)). Result: DeepSeek-R1 caches **4.88 MiB/token → 152.5 GiB KV at 32K context** — which does **not fit any box, including the 512GB M3 Ultra**, once weights (~335 GiB) are added.

The famous MLA memory savings are a property of an **absorbed** cache implementation (cache only the compressed latent `kv_lora_rank 512 + qk_rope 64` per layer, re-project per query at attention time). Absorbed = **68.6 KiB/token → 2.14 GiB @32K — 71× smaller**, and it should be **exact/lossless** (an algebraic reformulation, not an approximation). **Nobody has shipped absorbed-MLA in MLX / mlx-swift-lm** (confirmed absent from the vendored rev). This is the single biggest KV-memory lever in the whole catalog.

## Why it matters

It is the *only* thing that makes DeepSeek-R1 (and the `deepseek_v3` arch family) viable at serious context length on Apple Silicon at all. Without it, R1 is a weights-only demo — any real context OOMs. With it, R1 long-context serving on the 512GB M3 Ultra becomes feasible.

## Light triage / open questions

- Is the absorbed reformulation expressible with stock MLX ops + the existing `CompiledKVCache`, or does it need a custom Metal path (like the fused-attention question in the TurboQuant work)?
- Interaction with the compiled decode core (fixed-shape KV buffers) — the latent cache is smaller/differently-shaped.
- Verify losslessness through the hardened harness (teacher-forced KL vs the as-implemented decompressed baseline should be ~0 / float-noise).

## Next planning step

Own design note + implementation plan (do not fold into the context-operability feature). Gate: measure against the as-implemented baseline through the harness. Referenced from the operability spec §7 (backlog) and §8 (DeepSeek-R1 honesty case): `docs/superpowers/specs/2026-07-09-system-aware-context-operability.md`.
