# Google TurboQuant — implementable algorithm reference

Design input for the TurboQuant KV-quant implementation (the first flywheel customer). Primary source: **arXiv:2504.19874** (Zandieh/Daliri/Hadian/Mirrokni; QJL companion arXiv:2406.03482). **CONFIRMED** = from the paper; **INFERRED** = standard fill-in where the paper omits a formula; **VERIFY** = flagged for PDF re-check before hard-coding. Measured reference row from the hardened harness: Qwen3-32B 4-bit weights with fp16 KV vs bf16 weights with fp16 KV = tail-p95 **1.665 nats @ 24K**, ppl **+21.4%**. No ordinary affine-KV quality row was measured in this cycle; the KVarN/asymmetric gate owns that comparator.

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
Run `verify` (lossy-tier triad + canary at 2.5/3.5-bit), and `kl`/perplexity/**long-context tail-p95** vs the **bf16** reference on corpus v2 (incl. the 24K entry). **Promote to a dial tier only if it establishes a useful measured quality/size point against the same-weights fp16-KV row; do not claim an affine-KV comparison without a separately measured teacher-forced affine row.** Also record actual KV bytes/token once the memory-metrics FAST-FOLLOW lands.

## Spike A resolution (2026-07-09, on-box d=128 property tests + arXiv HTML v1 re-check)

- **Dequant scale CONFIRMED verbatim:** `x̃_qjl = (√(π/2)/d)·γ·Sᵀ·qjl`, no shrinkage factor. Our implementation's regression slope of `⟨q, r̂⟩` on `⟨q, r⟩` = **1.0066** (unbiased, correctly scaled), and the flipped matmul convention (`signs·Sᵀ`) measurably degrades (0.038 vs 0.029) — the `sign(r·Sᵀ)`/`signs·S` row-convention pairing is right.
- **Theorem 2 metric clarified (this was the VERIFY item):** unbiasedness `E[⟨y,x̃⟩] = ⟨y,x⟩` plus a **worst-case-query** distortion bound `D_prod = E[(⟨y,x⟩−⟨y,x̃⟩)²] ≤ (√3·π²·‖y‖²/d)·4^{−b}` (b = total bits; the earlier "3π²" note was the wrong constant — HTML v1 reads √3·π²). Paper's empirical table: `d·D_prod ≈ {1.57, 0.56, 0.18, 0.047}` for b = 1..4. **Our codec reproduces it: 0.175 @ 3 total bits, 0.0514 @ 4 total bits** (d=128, 4096 unit-norm pairs).
- **Sharp caveat discovered (derived, then measured):** for **independent** random q,k the unbiased QJL correction has variance `≈ (π/2)·‖r‖²/d` — a factor π/2 *worse* than simply dropping the residual (measured 0.0294 vs 0.0240 mean-abs). `_prod`'s advantage is the removal of the `_mse` shrinkage *bias* `⟨q,r⟩ ≈ ‖r‖²·⟨q,x⟩`, which dominates only when q correlates with x: at ⟨q,x⟩=1 the measured errors are 0.0282 (`_prod`) vs 0.1159 (`_mse`) — 4.1× better; crossover ≈ ⟨q,x⟩ ~ 0.2. Attention is the correlated regime (softmax cares about the high scores), which is why the paper's end-to-end results hold. **Any property test for `_prod` must use correlated or worst-case queries, or test unbiasedness/the Theorem-2 table — a mean-abs test on independent random pairs provably favors `_mse`** (0.122/d < 0.18/d from the paper's own numbers).

## Open gaps to resolve during implementation (paper-unspecified)
1. Fast rotation alternative (dense is heavy) — could we substitute a randomized-Hadamard rotation and re-fit Lloyd-Max? Empirical question; the paper doesn't, but it's the obvious perf lever.
2. Lloyd-Max levels for b≥3 (solve numerically; only b=1,2 tabulated).
3. Rotation/QJL granularity for KV (global vs per-head_dim — INFERRED: per head_dim).
4. Outlier-channel selection (fixed vs calibrated) + the exact 2.5-bit channel split (VERIFY).
5. K vs V same/different bit-widths (unstated).
6. ~~`_prod` distortion constant 3π² and the norm/‖r‖ bookkeeping when input isn't unit-norm (VERIFY vs PDF).~~ **RESOLVED** — see "Spike A resolution" above (constant is √3·π², worst-case-query metric; scale/convention confirmed). Non-unit-norm bookkeeping: γ=‖r‖₂ is stored explicitly so the residual estimate is norm-correct; Theorem 2 is stated for x on the unit sphere — K/V vectors are not unit-norm, so the *base* codebook's 1/√d scaling assumes unit-ish norm and per-vector norm handling for the base quantizer remains an integration-time check (Phase 2).

No official Google impl; ~10 unofficial community repos (unverified — re-derive constants from the paper, don't trust them). QJL sub-component: author repo amirzandieh/QJL.
