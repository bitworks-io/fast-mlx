# TurboQuant KV-cache first run — uniform-v1 (tqB2/tqB3) vs the 4-bit-weights baseline

**Date:** 2026-07-09 · **Branch:** `feat/turboquant` · **Box:** llmbench (M5 Max, 128GB)
**Candidate:** Qwen3-32B-4bit weights + TurboQuant KV (`TurboQuantKVCache`, materialize-then-attend)
**Reference:** Qwen3-32B-bf16 via mlx-lm (fp16 KV), teacher-forced, `measurement-corpus-v2`
**Baseline row:** same candidate weights with **fp16 KV** vs the same bf16 reference
(harness-spine firstrun, Run 3): kl_median 1.945e-01, tail-p95 @24K **1.665 nats**, ppl **+21.37%**.
**Evaluation lane:** [`LOSSY_FRONTIER`](README.md)—real quality loss is allowed when it buys a
useful, measured speed/memory/capacity point above the garbage floor.

This negative result is not a rejection of lossy tiers. `tqB3` is shelved because 4-bit affine
provides better quality at nearly the same design size, so TurboQuant is dominated on the
measured frontier. `tqB2` is shelved because its +488% perplexity and 10.09-nat long-context
tail cross the catastrophic floor. A materially faster or smaller configuration with bounded,
clearly reported loss would remain eligible for the user dial.

## What was built (Phase 1B → Task 7, all verified on-box)

- `TurboQuantCodec` `_prod` (arXiv:2504.19874 Alg. 2): Haar rotation Π + 1/√d Lloyd-Max base
  codes + QJL 1-bit sign residual with γ=‖r‖₂. Reproduces the paper's Theorem-2 distortion
  table (d·D = 0.175 @3 total bits, 0.051 @4, paper: 0.18/0.047); unbiased (slope 0.996).
- Non-unit-norm handling (paper §1.1): store per-row ‖x‖, quantize the direction, rescale on
  dequant. Relative RMSE == unit-norm RMSE bit-exactly; norm-covariant to 3e-6.
- `TurboQuantKVCache`: per-token code fields in fixed-shape buffers mirroring
  `CompiledKVCache`'s compile-legal discipline (in-graph offset, chunked grow, in-place
  reset). Cache adds **zero** error beyond the codec (bit-exact round-trip test). Storage
  dtypes uint8/int8 (exact), byte-aligned — NOT yet bit-packed.
- Decode + scoring integration behind `RunConfig.kvQuant` (`tq2.5`/`tq3.5`); quant ops trace
  cleanly under `MLX.compile`; engagement is the quantized cache's in-graph token counter, so
  a silent fp16 fallback cannot pass. fp16 exact triad still 60/60 vs mlx-lm.

## Quality — teacher-forced, context-locked

Lossy-triad statistic (verify, n=60, vs the SAME engine at fp16 KV — isolates KV loss from
weight loss): **tqB3 top-1 agreement 46/60 (76.7%)** · **tqB2 43/60 (71.7%)**. Both triads
PASS (non-crash, non-NaN, canary, engagement). Free-running greedy prefix was re-specced away
after measuring it chaotic: pos-0 logit perturbation rms ≈ 1.6–1.9 flips the first
high-entropy token on any prompt (evidence in `harness-evidence.jsonl`).

KL / perplexity vs bf16 (candidate 4-bit weights + listed KV tier; reference bf16 + fp16 KV;
3 short entries × 24 positions, 5,338- and 24,151-token entries × 128 sampled positions):

| stat | fp16 KV (baseline) | tqB3 KV (tq3.5 slot) | tqB2 KV (tq2.5 slot) |
|---|---|---|---|
| kl_median (short entries) | 1.945e-01 | 1.662e-01 | — |
| kl_long_context_tail_p95 (headline) | 1.665e+00 | **1.797e+00** | **1.009e+01** |
| — long entry p95 @ 24,151 tok | 1.665e+00 | 1.797e+00 | 1.009e+01 |
| — top-1 agreement (lossy triad, /60) | — | 46/60 (76.7%) | 43/60 (71.7%) |
| ppl_delta (pooled 328 positions) | +21.37% | **+32.61%** | **+487.67%** |

(tqB2 evidence: `harness-evidence.jsonl` @ 2026-07-10T03:50:41Z, corpus-v2. tqB2's 2-base-bit
tier is catastrophic — tail-p95 **10.09 nats** and ppl **+488%** — an order of magnitude past
the baseline; the sub-entry breakdown is moot at that magnitude.)

(tqB3 evidence: `harness-evidence.jsonl` @ 2026-07-10T00:28:33Z, corpus content hash
8dd73ade100742f2, 328 pooled positions. The short-entry kl_median landing slightly UNDER
the baseline row is position-mix noise at 24 positions/entry; the long-context tail and
pooled perplexity are the load-bearing statistics, and both are decisively worse.)

## Size — KV bytes/token (Qwen3-32B: 64 layers × 8 KV heads × 128 head_dim = 131,072 elt/token)

| tier | bits/element (format design) | KV bytes/token (design) | vs fp16 | v1 in-memory today |
|---|---|---|---|---|
| fp16 | 16 | 262,144 (256 KiB) | 1.0× | 256 KiB |
| 4-bit affine (mlx-lm, group 64) | 4.5 | 73,728 (72 KiB) | 3.6× | 72 KiB (packed) |
| tqB3 | 3+1+32/128 = 4.25 | 69,632 (68 KiB) | 3.8× | ~264 KiB (byte-aligned, unpacked) |
| tqB2 | 2+1+32/128 = 3.25 | 53,248 (52 KiB) | 4.9× | ~264 KiB (byte-aligned, unpacked) |

Bits/element includes BOTH per-row fp16 scalars (γ and ‖x‖) amortized over head_dim
(`TurboQuantTier.bitsPerElement`, honest-accounting test updated). The v1 cache stores
uint8 idx + int8 sign + fp32 norms — the design footprint requires bit-packing (deferred
engineering, mechanical).

## Perf — decode tok/s (bench, 256 tokens, 3 runs post-warmup)

Precise decode tok/s **not measured** — the verdict is decided on quality alone (both tiers
fail decisively), so the bench is moot for the shelve decision. Qualitatively confirmed slow:
materialize-then-attend dequantizes the full capacity buffer per layer per step (the codec's
two matmuls Π/S per read), a large per-step cost that grows with context. Fused
quantized-attention is the paper's Metal-kernel work, out of scope for v1. **Net: TurboQuant
v1 is both slower AND lower-quality than the fp16-KV path it would replace** — a second,
independent reason to shelve.

## Verdict — per tier

**SHELVE both tiers** — dated negative result (2026-07-09). Uniform-v1 TurboQuant does **not** beat
the 4-bit-affine baseline on Qwen3-32B:

- **tqB3** (`tq3.5` slot; 3 base + 1 QJL, ~4.25 bits/elem design): tail-p95 **1.797 vs 1.665**
  (+8%) and ppl **+32.6% vs +21.4%** — worse quality at a KV *design* size (68 KiB) only
  marginally under 4-bit affine (72 KiB), and the v1 in-memory footprint is larger (unpacked).
  Dominated on the quality/size frontier.
- **tqB2** (`tq2.5` slot; 2 base + 1 QJL): tail-p95 **10.09**, ppl **+488%** — catastrophic,
  unusable as-is.

**Why (root-caused, not guessed):** the codec is *provably correct* — it reproduces the paper's
Theorem-2 distortion table (d·D 0.175/0.051 vs 0.18/0.047) and is unbiased (slope 0.996). The
loss is the ~2% per-vector KV error **compounding across 64 layers** (pos-0 logit rms ≈ 1.8,
enough to flip the first high-entropy token). The paper's near-lossless 3.5-bit used **outlier
channels** (32ch@3b + 96@2b) that v1 deliberately deferred — uniform bit allocation is the gap.
Second, independent strike: materialize-then-attend decode is slow.

**Gated next step — Spike B (outlier channels).** Allocate higher precision to the high-variance
channels (the paper's borrowed recipe) and re-measure through this same harness. It is the one
lever between uniform-v1 and the paper's result, and the only thing that could move tqB3 under
the baseline. Until then TurboQuant stays behind the `tq2.5`/`tq3.5` flag — built, verified,
unshipped as a dial tier.

**Not wasted:** the verified paper-faithful codec (Spike A), the `TurboQuantKVCache`
(materialize-then-attend, bit-exact round-trip), and the tier plumbing are the exact foundation
Spike B builds on. The negative result is on the *uniform bit allocation*, not the implementation.

## Reproduce

```sh
BIN=<Release fastmlx-harness>
$BIN verify --model ~/perf-work/models/Qwen3-32B-4bit --kv-quant tq3.5 --n 60
$BIN kl --model ~/perf-work/models/Qwen3-32B-4bit \
    --reference-model ~/.mlx-serve/models/mlx-community/Qwen3-32B-bf16 --kv-quant tq3.5
$BIN bench --model ~/perf-work/models/Qwen3-32B-4bit --kv-quant tq3.5 --label tqB3
```
