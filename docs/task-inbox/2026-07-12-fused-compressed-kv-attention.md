---
status: captured
type: metal-kernel-spike
priority: high
created: 2026-07-12
source: carry-forward-plus-sol-audit
planning_ready: false
implementation_ready: false
---

# Fused compressed-domain KV attention

## Raw Capture

Eliminate the materialize-then-attend bottleneck. The current TurboQuant cache reconstructs
the full-precision cache before attention; a better storage codec therefore cannot earn a
decode-speed tier. Build a compressed-domain Metal path only after profiling the stock MLX
primitives and defining an ordinary affine K/V baseline.

## Planning Seed

User/operator: a long-context operator who needs KV compression to reduce both unified-memory
pressure and attention traffic.

Desired outcome: attention consumes packed K/V without full-cache dequantization and retains
the exact fp16 oracle for a lossless storage mode.

Acceptance signals:

- fp16/fused control agrees with stock attention within the predeclared numeric tolerance;
- GQA, masks, head dimensions, cache growth, rollback, and batching layouts are covered;
- the batch-compaction poison case is exact: merge unequal rows, remove the longest/zero-padding
  boundary row, append again, then prove physical end, mask width, survivor bytes/logits, and
  dense-control parity;
- packed bytes/token include scales, biases, alignment, and workspace;
- end-to-end 32K and 128K decode beats materialize-then-attend and the base path; a kernel-only
  microbenchmark cannot promote the feature;
- MLX-coupled tests use Xcode on the bench box and preserve actor confinement.

Known failure cases: materializing head expansion, silent mask errors, inferring the write end from
currently active padded offsets after a row removal, non-power-of-two head dimensions, temporary
workspace erasing memory gains, Metal maintenance burden, and a large kernel speedup that remains
end-to-end negative. Structural cache corruption is a correctness bug, never quantified dial loss.

## Sources / existing foundation

- [Zig L2 result](../reference/2026-07-08-carry-forward-performance-backlog.md)
- [MLX fused-attention integration discussion](https://github.com/ml-explore/mlx/issues/3404)
- [Open-TQ-Metal paper](https://arxiv.org/abs/2604.16957) (external numbers unverified)
- [oMLX batched TurboQuant physical-end corruption fix](https://github.com/jundot/omlx/pull/2201)

## Next Step

Profile fp16, affine materialize-then-attend, and current MLX quantized primitives at
8K/32K/128K; write a kernel contract only if attention traffic is the measured bottleneck.
