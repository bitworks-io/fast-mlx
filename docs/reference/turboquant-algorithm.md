# Google TurboQuant — implementable algorithm reference

Design input for the TurboQuant KV-quant implementation (the first flywheel customer). Primary source: **arXiv:2504.19874** (Zandieh/Daliri/Hadian/Mirrokni; QJL companion arXiv:2406.03482). **CONFIRMED** = from the paper; **INFERRED** = standard fill-in where the paper omits a formula; **VERIFY** = flagged for PDF re-check before hard-coding. Baseline it must beat (from our hardened harness): 4-bit affine KV vs bf16 = tail-p95 **1.665 nats @ 24K**, ppl **+21.4%**.

## The algorithm (CONFIRMED unless noted)

**Setup (global, once — not per token):**
- `Π ∈ R^{d×d}`: dense Haar-random orthogonal (QR of an i.i.d.-Gaussian matrix). Fixed, reused for all vectors. `d = head_dim`. **No fast/structured (Hadamard/SRHT) alternative is given in the paper** — real O(d²) apply cost per token (small at d=64–128, but non-zero vs our Hadamard scheme's O(d log d)).
- For `_prod`: `S ∈ R^{d×d}`, i.i.d. N(0,1), also fixed/global.
- **Lloyd-Max codebook** `{c_1..c_{2^b}}`: solved once offline for the coordinate density and cached as a LUT. Post-rotation coordinates are Beta((d−1)/2,(d−1)/2)-shaped → **≈ N(0, 1/d)** for practical d, so: **solve Lloyd-Max on N(0,1) once per bit-width, then rescale levels by 1/√d** (levels are dimension-independent in shape). Confirmed values: b=1 `{±√(2/π)/√d}`; b=2 `{±0.453/√d, ±1.51/√d}` (= the classical Gaussian Lloyd-Max constants — a good self-check). Decision boundaries = midpoints of sorted centroids. [INFERRED for other b: standard Lloyd-Max iteration — boundaries at midpoints, centroids = conditional means, iterate to convergence.]

**`_prod` per-token (Algorithm 2 — the KV variant):**
```
Quant(x):                            # x = one K or V head-vector (dim d)
  y   = Π · x                        # dense matmul
  idx = argmin_i |y_j − c_i|  ∀j     # nearest Lloyd-Max centroid per coord, (b−1) bits/coord
  r   = x − Πᵀ·dequant(idx)          # residual in original space
  qjl = sign(S · r)                  # 1 bit/coord
  store: (idx, qjl, γ=‖r‖₂)          # NO per-group scale/zero
Dequant(idx, qjl, γ):
  return Πᵀ·dequant(idx)  +  (√(π/2)/d)·γ·(Sᵀ·qjl)
```
- Bit tiers: **2.5-bit** = 2 base + 1 QJL (with outlier channels: e.g. 32 ch @3b + 96 @2b — VERIFY the exact split; the 2.25-vs-2.5 arithmetic didn't reconcile from the extraction). **3.5-bit** = 3 base + 1 QJL (paper: **3.5-bit matches full precision** on Llama-3.1-8B LongBench-v1: 50.06 vs 50.06; 2.5-bit 49.44).
- Unbiased inner-product in expectation over S's draw (Thm 2); with S fixed/reused, per-entry bias is small + concentrates with d (state precisely, don't overclaim exact per-vector unbiasedness).
- **Online / data-free** for the core quantizer; the **outlier-channel selection** may need calibration (paper borrows outlier handling from prior work, doesn't state selection method) — treat "fully calibration-free" as qualified for the outlier recipe.

## mlx-swift implementation approach

- **MLX native `quantize` = affine/mxfp4/mxfp8/nvfp4 only** — cannot represent a non-uniform LUT quantizer. mlx-swift-lm's `QuantizedKVCache` is affine (what we already have, mislabeled "turbo"). MLX issue #3404 requests TurboQuant-in-SDPA — **open, not shipped**. → fully custom build; we'd be first.
- **Rotation (Π, S):** plain `MLX.matmul` against a small fixed matrix (one Π, one S per `head_dim`; O(d²) memory trivial). No custom kernel.
- **LUT quantize/dequant:** `MLXFast.metalKernel` (gather/lookup — a comfortable fit) or plain `mx.take`/gather at some perf cost.
- **Attention read — the hard part:** MLX exposes no custom-quant FlashAttention path; a fused dequant-inside-SDPA kernel = hand-rolling a FlashAttention-class Metal kernel. **v1 = materialize-then-attend:** dequant stored K/V (LUT gather + Πᵀ / Sᵀ matmul) to full precision, then existing `MLXFast.scaledDotProductAttention`. Keeps the **KV-cache storage-size win** (the point); the attention **bandwidth** win needs true fusion → stretch goal.
- **Integrate with `CompiledKVCache`:** it stores fp16 K/V today; add a TurboQuant-quantized cache variant storing (codes, qjl signs, per-token ‖r‖₂ + rotation state as globals), dequant-on-read. Mind the compiled-step retrace + the 8GB allocator-cache bound from the long-context work. Ships behind the `kvQuant` tier the harness already records ("tq2.5"/"tq3.5").

## Validation path (through the hardened harness)
Run `verify` (lossy-tier triad + canary at 2.5/3.5-bit), and `kl`/perplexity/**long-context tail-p95** vs the **bf16** reference on corpus v2 (incl. the 24K entry). **Promote to a dial tier only if it beats the 4-bit affine baseline (tail-p95 1.665 @24K) at equal-or-smaller KV size**; else shelve with a dated negative result. Also record KV bytes/token (the storage win) once the memory-metrics FAST-FOLLOW lands.

## Open gaps to resolve during implementation (paper-unspecified)
1. Fast rotation alternative (dense is heavy) — could we substitute a randomized-Hadamard rotation and re-fit Lloyd-Max? Empirical question; the paper doesn't, but it's the obvious perf lever.
2. Lloyd-Max levels for b≥3 (solve numerically; only b=1,2 tabulated).
3. Rotation/QJL granularity for KV (global vs per-head_dim — INFERRED: per head_dim).
4. Outlier-channel selection (fixed vs calibrated) + the exact 2.5-bit channel split (VERIFY).
5. K vs V same/different bit-widths (unstated).
6. `_prod` distortion constant 3π² and the norm/‖r‖ bookkeeping when input isn't unit-norm (VERIFY vs PDF).

No official Google impl; ~10 unofficial community repos (unverified — re-derive constants from the paper, don't trust them). QJL sub-component: author repo amirzandieh/QJL.
