# TurboQuant Spike B — outlier channels (the gated next step after the uniform-v1 shelve)

- **Captured:** 2026-07-09
- **Status:** backlog (gated next step from a dated negative result)
- **Task type:** quantization / flywheel follow-up
- **Priority:** bounded closure — below fused compressed-domain attention and KVarN
- **Owner project:** fast-mlx engine (`feat/turboquant` foundation, merged behind the `tq2.5`/`tq3.5` flag)

## Why (the negative result this follows)

Uniform-v1 TurboQuant was built, verified paper-faithful (reproduces the Theorem-2 distortion table, unbiased), and **measured — it loses to 4-bit affine on Qwen3-32B**: tqB3 tail-p95 1.797 / ppl +32.6% and tqB2 10.09 / +488%, vs the baseline 1.665 / +21.4%. Verdict: `docs/superpowers/verdicts/2026-07-09-turboquant-firstrun.md`. Root cause: ~2% per-vector KV error compounds across 64 layers; the paper's near-lossless 3.5-bit used **outlier channels** (allocate higher precision to high-variance channels — e.g. 32ch@3b + 96@2b for its 2.5-bit) that v1 deliberately deferred. Uniform bit allocation is the gap.

## The spike

Add outlier-channel bit allocation to the codec: identify the high-variance channels (post-rotation, or per the paper's borrowed recipe) and quantize them at higher precision, the rest lower. Open questions the paper leaves (see `docs/reference/turboquant-algorithm.md` gaps #4): fixed vs calibration-derived channel selection; the exact split; whether it reconciles the "2.5-bit" label. Re-measure tqB2/tqB3-with-outliers through the **same** hardened harness (teacher-forced KL/ppl/tail-p95 vs bf16 on corpus-v2) against the 1.665@24K baseline.

**Audit update, 2026-07-12:** keep the paper-faithful outlier arm, but make this a small
closure matrix rather than betting the whole second cycle on one recipe. Current community
implementations provide three cheap, explicitly unverified ablations: spend more bits on K
than V (`K4/V2`, `K8/V2`, `K8/V4`), protect the first/last attention layers, and compare
QJL against a no-QJL/MSE control. [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) and
[TurboQuant+](https://github.com/TheTom/turboquant_plus) are implementation leads; their
numbers do not replace fast-mlx evidence. Compare actual packed bytes, not nominal tier names.

## Gate

Promote to a dial tier only if one bounded recipe **beats** both the 4-bit-affine baseline and
the new KVarN/asymmetric frontier at equal effective bytes. If none does, shelve TurboQuant
fully (a second dated negative result). No decode-speed promotion is possible while the path
materializes the full cache before attention; the fused compressed-domain kernel is a
separate, higher-priority prerequisite.

## Foundation (already built, do not redo)

The verified codec (`TurboQuantCodec`, Spike A passed), `TurboQuantKVCache` (materialize-then-attend, bit-exact round-trip), and the tier plumbing (`RunConfig.kvQuant`, the `CompiledCache` protocol) are in place behind the `tq2.5`/`tq3.5` flag. Spike B extends the codec's bit allocation only.
