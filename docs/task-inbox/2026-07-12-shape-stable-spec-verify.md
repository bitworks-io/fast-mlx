---
status: captured
type: correctness-performance-spike
priority: conditional
created: 2026-07-12
source: qwen3-32b-eagle3-verdict
planning_ready: false
implementation_ready: false
---

# Shape-stable target verification for exact speculative decoding

## Raw Capture

Investigate whether fast-mlx can make multi-token target verification produce the same greedy
argmax and retained KV state as its one-token autoregressive path. The Qwen3-32B EAGLE-3 gate
found an immediate one-token-versus-batched argmax difference on the 8-bit target and retained
state drift after rejected-future processing on the 4-bit target.

## Light Triage

User/operator: the fast-mlx owner deciding whether any trained speculative decoder can satisfy
the byte-identical temperature-zero contract on MLX.

Acceptance signals:

- identical target argmax and retained per-layer KV state for one-token and `K+1` verify shapes
  across the authenticated 4-bit and 8-bit Qwen3-32B targets;
- byte-identical long-output base/spec streams at `k=1` and `k=3`;
- any sequential commit, cache repair, or shape-stable kernel charges all added target work and
  still clears the same-target throughput baseline;
- no relaxation of the byte-identity invariant and no redefinition of the baseline.

Open questions: whether the divergence is avoidable with a stable target kernel, requires
replaying retained tokens, or changes in a newer MLX/runtime conversion. Upstream numerical
documentation is explanatory context, not proof of a fix.

## Next Step

Continuous batching is complete; keep this deferred behind the current ranked queue. Re-plan
only when a bounded shape-stable kernel hypothesis or a new compatible product-size
trained-drafter pairing exists. Use the 2026-07-12 EAGLE verdict's four-history replay as the
regression oracle.
