---
status: captured
type: dependency-performance-gate
priority: medium-high
created: 2026-07-14
source: user-supplied-omlx-digest-plus-primary-source-review
planning_ready: true
implementation_ready: false
---

# MLX-Swift 0.32 small-batch kernel uplift gate

## Raw Capture

Re-evaluate fast-mlx when an official MLX-Swift release contains the MLX 0.32 Metal core.
The strongest Apple lead is `qmv_wide`: it amortizes quantized-weight reads across the small
`M=2...8` batches used by speculative target verification and potentially continuous batching.
Do not infer an end-to-end fast-mlx multiplier from upstream kernel microbenchmarks.

## Light Triage

fast-mlx currently pins `mlx-swift` 0.31.6, and the official MLX-Swift release list still ends
there. Execution is blocked until a compatible official release exists, unless a separate review
explicitly approves a source/core revision pin.

Relevant sourced leads:

- MLX `qmv_wide` targets speculative verification batches and reports Apple kernel-level gains
  across chips, widths, and quantization modes. Whether the current fast-mlx graphs select it and
  improve service throughput is unverified.
- Metal GEMV becoming JIT-compilable and RMSNorm register caching are plausible secondary gains,
  but neither is a fast-mlx token/s result.
- The cited RoPE-without-copy change is CUDA-only and is not an Apple-Silicon roadmap claim.
- The asymmetric-SDPA result applies only to its declared Q/K/V shape and must not be generalized.

User/operator: a fast-mlx user who should receive upstream runtime gains without losing exactness,
stability, or a trustworthy old-versus-new comparison.

Acceptance signals:

- pin the exact MLX-Swift release and embedded MLX core revision in evidence;
- compare old/new clean-SHA builds on identical weights, prompts, thermals, and cache limits;
- measure base batch-1 decode, service C=2/4/8, and target-verification widths M=2/4/8 on the
  M5 Max and offered M3 Ultra when available;
- prove temperature-zero byte identity for every exact path and preserve the speculative decoder's
  strict exactness contract;
- report kernel engagement or a defensible trace; a dependency bump with no selected hot path is
  not a speed claim;
- keep per-shape latency, throughput, peak memory, and compile/warmup costs so a regression cannot
  hide inside one aggregate.

## Official Sources

- [MLX 0.32.0 release](https://github.com/ml-explore/mlx/releases/tag/v0.32.0)
- [MLX-Swift releases](https://github.com/ml-explore/mlx-swift/releases)
- [`qmv_wide` small-batch quantized matvec](https://github.com/ml-explore/mlx/pull/3764)
- [Metal GEMV JIT compilation](https://github.com/ml-explore/mlx/pull/3705)
- [Metal RMSNorm register caching](https://github.com/ml-explore/mlx/pull/3754)
- [CUDA-only RoPE copy removal](https://github.com/ml-explore/mlx/pull/3704)
- [Shape-specific asymmetric SDPA](https://github.com/ml-explore/mlx/pull/3637)

## Next Planning Step

Watch for the first official MLX-Swift release containing the new core, then source-lock its exact
revision and write the A/B matrix before changing `Package.resolved`.
