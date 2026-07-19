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
capacity tiers and selected fused-kernel inputs, but measured no lossy speed win. The frozen
KVTuner bundle is Qwen-only unless independently calibrated and authenticated for another family.

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
- end-to-end decode at checkpoint-valid contexts beats materialize-then-attend and the base path;
  a kernel-only or synthetic-geometry microbenchmark cannot promote the feature;
- the full proofs include quality, actual storage, peak memory, decode/prefill, and runnable
  user/operator workload evidence for the selected KVarN i8 and affine paths, with KVTuner limited
  to Qwen unless separately calibrated;
- Qwen3-32B qualifies only at 8K smoke/32K because its max context is 40,960; the 128K Qwen cell is
  an authenticated refusal;
- staged Llama-3.3-70B qualifies at 8K smoke, 32K, and near-128K only when prompt+output <= 131,072;
- Qwen and Llama sharing Q64/KV8/D128 can support a same-geometry dense-GQA claim only. Broad
  support claims or shared default policy require another popular materially different attention
  geometry;
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

Continue Phase 1 of the dated
[`2026-07-18-fused-compressed-kv-attention.md`](../superpowers/plans/2026-07-18-fused-compressed-kv-attention.md)
plan. Phase 0 is complete at clean `07219679280abd2f7cefbeef86b71bbec018a1c2`. Its synthetic
decode evidence selected the existing quantized-matmul route with independent K/V geometry and
deferred custom Metal; it streamed checkpoint bytes only for content authentication and did not
instantiate or execute them as MLX model tensors, so it did not prove prefill, end-to-end, model,
family, or dial performance. Define and TDD the smallest model-generic packed-attention router seam,
including independent K/V geometry, exact mask/GQA behavior, frozen-schedule authentication,
unsupported architecture and non-finite refusal, and a portable pinned dependency strategy. Then
integrate actor-confined affine K4V2-g64 and frozen Qwen KVTuner scalar runtime before KVarN or
batch compaction.
