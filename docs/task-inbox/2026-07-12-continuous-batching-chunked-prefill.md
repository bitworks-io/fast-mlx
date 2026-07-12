---
status: captured
type: engine-feature
priority: high
created: 2026-07-12
source: carry-forward-plus-sol-audit
planning_ready: true
implementation_ready: false
---

# Continuous batching with decode-first chunked prefill

## Raw Capture

Port the retired Zig engine's largest remaining service-throughput multiplier into the
single-owner Swift inference actor. Separate decode and prefill scheduling, admit bounded
prefill chunks without stalling active decodes, and preserve the platform's
drain-pipeline-before-batch-join invariant.

## Planning Seed

User/operator: a server operator handling concurrent Concierge/agent requests.

Desired outcome: higher aggregate throughput without corrupting a solo stream when another
request joins, starving short requests behind a long prefill, or exposing non-Sendable MLX
state outside the actor.

Acceptance signals:

- single-stream tokens remain byte-identical before and after a mid-generation batch join;
- aggregate 1/2/4/8-stream throughput, p50/p95 TTFT, p50/p95 TPOT, and fairness are recorded;
- cancelled/disconnected requests release their slot within one configured keepalive
  interval, matching the platform spec's disconnect-cancel SLA;
- dense architectures batch; unsupported MoE/hybrid state fails closed until proven;
- PLD/trained speculation disables in the batched arm unless a separate exact contract lands;
- after the warmup sample, a 24-hour mixed-workload soak holds RSS drift below 5%, matching
  the incumbent's measured stability gate.

Known failure cases: duplicate KV position on solo→batch transition, ragged-mask corruption,
cache merge/extract aliasing, cancelled slot retention, prefill starvation, and shape-retrace
thrash. A small `shapeless` compile probe belongs inside this plan, not as an assumed fix.

## Sources / existing foundation

- [Platform scheduler invariants](../superpowers/specs/2026-07-08-fast-mlx-platform-design.md)
- [Zig carry-forward result](../reference/2026-07-08-carry-forward-performance-backlog.md)
- [Current MLX-LM releases](https://github.com/ml-explore/mlx-lm/releases/tag/v0.31.2)
- [Sarathi-Serve](https://arxiv.org/abs/2403.02310)

## Next Step

Design slot/cache ownership and the pure scheduler state machine first; TDD the transition,
cancellation, admission, and fairness rules before connecting MLX batch caches.
