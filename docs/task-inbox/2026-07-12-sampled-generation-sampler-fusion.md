---
status: captured
type: engine-feature
priority: medium
created: 2026-07-12
source: carry-forward-reconciliation
planning_ready: false
implementation_ready: false
---

# Sampled-generation foundation, then sampler fusion

## Raw Capture

Split sampler fusion from continuous batching. fast-mlx is currently greedy-only, while the
retired Zig engine's L1/L3/L1b/L3b gains depend on temperature, top-k/top-p, exact RNG order,
and batched speculative accept sampling. Define that distribution contract first, then port
the proven fusion stack.

## Planning Seed

User/operator: a small/fast/MoE or draft-model user running stochastic chat and speculative
decode without paying repeated full-vocabulary sorts.

Desired outcome: distribution-preserving sampling with a measurable fast path for the model
classes where sampler/vocabulary cost is material.

Acceptance signals:

- seeded scalar and batched paths consume RNG in the same documented order;
- greedy, temperature, top-k, top-p, min-p, penalties, and speculative acceptance each have
  distribution/equivalence tests appropriate to their contract;
- `top_k` operates along the vocabulary axis for `[B,S,V]` and cannot flatten positions;
- L1/L3/L1b/L3b are measured separately and cumulatively on small/fast and 27B+ controls;
- Qrita or fused LM-head sampling is considered only if profiling shows top-p/logit
  materialization remains material after the known fusion.

Known failure cases: silent shared masks across positions, changed RNG ordering, top-p=1.0
benchmarks that never engage the optimization, exact-greedy tests that say nothing about
stochastic distributions, and large-model results too small to justify complexity.

## Next Step

Specify sampler behavior and seed plumbing in `HarnessCore`; write the red tests before adding
MLX sampling to the actor.
