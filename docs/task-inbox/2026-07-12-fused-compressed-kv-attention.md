---
status: active
type: metal-kernel-spike
priority: high
created: 2026-07-12
source: carry-forward-plus-sol-audit
planning_ready: true
implementation_ready: true
---

# Fused compressed-domain KV attention

## Raw Capture

Eliminate the materialize-then-attend bottleneck. The current TurboQuant cache reconstructs
the full-precision cache before attention; a better storage codec therefore cannot earn a
decode-speed tier. Build a compressed-domain Metal path only after profiling the stock MLX
primitives and defining an ordinary affine K/V baseline.

2026-07-18 selection update: this is now the top engine gate after the KVarN/asymmetric KV
frontier verdict. Carry forward KVarN i8 plus the shared affine/KVTuner storage primitives
selected or retained by the dated KVarN evidence. Do not overclaim the prior cycle: it promoted
capacity tiers and selected fused-kernel inputs, but measured no lossy speed win.

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
- the full 32K and 128K proofs include quality, actual storage, peak memory, decode/prefill, and
  runnable user/operator workload evidence for the selected KVarN i8 and affine/KVTuner paths;
- a second materially different popular model family validates the frontier before broad support
  claims or shared default policy;
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

Continue Phase 0 of the dated
[`2026-07-18-fused-compressed-kv-attention.md`](../superpowers/plans/2026-07-18-fused-compressed-kv-attention.md)
plan. The authenticated plan/evidence contract and deterministic MLX fixture are complete; add the
fresh-output probe CLI/evidence producer, then profile fp16, affine materialize-then-attend, and the
pinned Swift-LM quantized-attention helper at 8K/32K/128K.
Write a portable runtime/kernel contract only after measured evidence identifies the bottleneck.
