---
status: captured
type: engine-feature
priority: high
created: 2026-07-12
source: carry-forward-reconciliation
planning_ready: true
implementation_ready: false
---

# Exact prefix/session cache and request-start latency stack

## Raw Capture

Restore the optimized incumbent's exact multi-turn speed path: hot RAM prefix/session
checkpoints, nearest-prefix lookup, positive success-only commit, eager model warmup, and a
chat-template/tokenize cache. Add cold SSD snapshots only after the hot path is correct and
memory-accounted.

## Planning Seed

User/operator: Concierge and coding-agent users who repeatedly extend a long system prompt or
conversation and need warm turns to avoid full re-prefill.

Desired outcome: revisited prompts resume from exact actor-confined model state without
cross-request poisoning or an unbounded unified-memory resident set.

Acceptance signals:

- A/B/A and multi-turn agent-memory tests prove cache reuse cannot alter output or tool state;
- the key covers model/revision, tokenizer/template, tools, exact KV config, position/RoPE
  semantics, architecture/recurrent state, and drafter state;
- only clean successful generations commit; image/errored/pad-only/zero-token paths fail closed;
- entry and true retained-byte budgets evict LRU state and reserve capacity up front;
- second-turn cache-read tokens, cold/warm TTFT, template/tokenize time, and apparent/physical
  prefill rates are reported separately;
- hybrid/SSM checkpoints are explicit boundaries; arbitrary trim is not assumed.

Known failure cases: shared-buffer undercounting, template/tool/config cross-contamination,
vision embedding reuse, recurrent-state snapshots larger than reported, cache-hit benchmark
contamination, copy/restore breaking compiled array identity, and SSD snapshots containing
sensitive prompt state.

## Sources / local precedent

- [Platform cache contract](../superpowers/specs/2026-07-08-fast-mlx-platform-design.md)
- [Preserved incumbent changelog](../reference/mlx-serve-archive/mlx-serve-CHANGELOG.md)
- [MLX-LM v0.31.2 cache changes](https://github.com/ml-explore/mlx-lm/releases/tag/v0.31.2)
- [SGLang RadixAttention](https://arxiv.org/abs/2312.07104)

## Next Step

Plan this against continuous-batching slot ownership, then TDD snapshot semantics and the
positive commit gate with pure fake-cache state before touching MLX arrays.
