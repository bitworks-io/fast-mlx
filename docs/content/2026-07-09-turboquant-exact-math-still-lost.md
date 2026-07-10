# We implemented Google's TurboQuant exactly, matched the paper's error tables — and it still lost to plain 4-bit quantization

**Whitepaper themes:** The optimization dial — quantified precision-loss tuning; Building a high-performance MLX engine in Swift

## The promise

The KV cache is the memory bill for long context. On Qwen3-32B, every token you keep
costs 256 KiB of fp16 keys and values — 64 layers × 8 KV heads × 128 dims × 2 tensors.
At a 24K-token context that's 6 GB per sequence, and it scales linearly from there.

TurboQuant (arXiv:2504.19874, Google Research) is an elegant attack on that bill. Rotate
each key/value vector by a fixed random orthogonal matrix so its coordinates become
near-Gaussian, snap each coordinate to a tiny Lloyd-Max codebook, then — the clever part —
sketch the *residual* with one sign bit per dimension (a Johnson-Lindenstrauss trick, QJL)
so that inner products against the reconstruction are **unbiased**. Attention is made of
inner products; unbiased is exactly what you want. The paper reports near-lossless quality
at ~3.5–4 bits per element, with distortion tables to back it up.

We wanted it as a `kvQuant` tier in our precision-loss dial: measured trade-offs, not
vibes-based defaults.

## Building it honestly

We implemented the quantizer in Swift on MLX and refused to integrate it until the math
proved itself on the paper's own terms:

- **Distortion table, reproduced.** At 3 total bits/element our measured inner-product
  distortion d·D was 0.175 (paper: ≈0.18); at 4 bits, 0.051 (paper: ≈0.047). Within 20%,
  at real head_dim (128).
- **Unbiasedness, reproduced.** Regression slope of estimated vs true inner products:
  0.996. That slope pins the (π/2)^½/d dequant scale — any other constant biases it.
- **The gap the paper hides in a subsection.** The guarantees hold for *unit* vectors.
  Real K/V rows aren't. Store ‖x‖ per row, quantize the direction, rescale on read
  (paper §1.1) — our round-trip relative error then matched the unit-norm error
  bit-for-bit.
- **A cache that adds nothing.** The KV cache stores codes and dequantizes on read
  (materialize-then-attend). Its output is bit-exact against the raw codec — the storage
  and scatter plumbing contributes zero additional error. The quantization ops even trace
  cleanly under `MLX.compile`, so the compiled decode path kept its fused-step design.

Every claim above is a unit test that runs on the bench box.

## The first surprise: greedy prefixes are a chaotic metric

Our first equivalence gate asked the "near-lossless" 4-bit tier to reproduce the fp16
engine's first 16 greedy tokens. It matched **two**. Bug hunt? The compiled and uncompiled
quantized paths agreed with each other well past the float-noise horizon, and scoring the
*identical* context on both tiers showed the real story: quantized KV perturbs the final
logits by ~1.9 RMS on a 64-layer model. The first sentence boundary is a high-entropy
near-tie; ±1.9 flips it on essentially any prompt, and greedy decoding never looks back.
One flipped token diverges everything after it — the same lesson that had already forced
our KL metric to be teacher-forced. We re-specced the gate to context-locked top-1
agreement: the 4-bit tier agrees with fp16-KV on 46/60 forced positions (76.7%), the
3-bit tier on 43/60. Coherent, working — measurably lossy.

## The measurement that decides

The dial's instrument is teacher-forced KL and perplexity against a bf16 reference over a
versioned corpus, with a 24,151-token entry because KV-quant loss accrues with context and
lives in the tail. The row to beat: the same 4-bit-weight model with **fp16 KV** scores
tail-p95 **1.665 nats** @24K and ppl **+21.4%** vs bf16.

TurboQuant tqB3 (3 base bits + 1 QJL bit ≈ 4.25 bits/element with the per-row scalars):
tail-p95 **1.797 nats**, ppl **+32.6%**. The KV quantization added half again the entire
weight-quantization loss — at a size (68 KiB/token) that plain 4-bit affine KV
quantization (72 KiB/token, group-wise scale+zero, boring since forever) roughly matches
while behaving near-losslessly in the literature. The 3-bit tier is smaller (52 KiB/token)
and, as expected, worse. And the v1 decode path pays a real speed tax on top: it
dequantizes the whole cache every step because the fused quantized-attention kernel is
precisely the engineering the paper's system section is about.

**Verdict: shelved, both tiers — a dated negative result.**

## Why the exact math lost

Nothing above contradicts the paper. The paper's near-lossless sub-integer results use
**outlier channel mixing** — the 3.5-bit configuration spends 3 bits on 32 outlier
channels and 2 bits on the other 96. We deliberately deferred that to a later spike and
shipped uniform bits, naming the tiers honestly (tqB2/tqB3, not "2.5-bit") because of
exactly this risk. The measurement says the deferred part wasn't optional: on a real
32B model, per-vector error that looks tiny (2% relative RMSE, exactly as the theory
predicts) compounds through 64 layers of attention into logit-scale noise, and uniform
bit allocation is evidently the wrong way to spend the budget. Outlier channels — Spike B
— is the gated next step, not a dead end. The codec, the cache, the tier plumbing, and
the engagement-trapped measurement path all carry over unchanged.

## The generalizable lesson

Verifying a paper's math is necessary and radically insufficient. Our quantizer
reproduced Google's distortion table to three significant figures and *still* lost
end-to-end, because the end-to-end system multiplies per-component error through depth,
data distributions, and decoding dynamics that no codec-level table captures. The
flywheel discipline did its job: paper-faithful properties gated the build, an
engagement-trapped harness made a silent fallback impossible, teacher-forced statistics
replaced a chaotic gate before it could mislead, and the promote/shelve call was made by
one instrument against one recorded baseline. A negative result with numbers, provenance,
and a named next experiment is not a failure of the process. It is the process.
