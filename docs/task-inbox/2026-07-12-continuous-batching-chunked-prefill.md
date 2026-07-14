---
status: completed
phase: promoted-measured-service-building-block
type: engine-feature
priority: high
created: 2026-07-12
completed: 2026-07-14
source: carry-forward-plus-sol-audit
planning_ready: false
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
- direct runtime cancellation releases a request slot within one configured keepalive
  interval; real client-disconnect propagation is tracked by the separate production-route
  task and remains required before a service default;
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

## Outcome — 2026-07-14

**PROMOTED as an exact, measured dense-Qwen3 service-policy building block.** The corrected
same-workload Qwen3-32B frontier selects solo PLD at concurrency one and batch-no-spec for the
tested simultaneous bursts at concurrency two, four, and eight. Final-SHA transition probes,
byte-denominated dense KV admission, cancellation/reuse, byte-identical A/B/A recovery, and the
full 86,400-second post-warmup soak all pass. Peak RSS drift was 2.2444%; every one of 33 soak
predicates passed 3,519/3,519 cycles.

See the [execution plan](../superpowers/plans/2026-07-12-continuous-batching-chunked-prefill.md),
[dated verdict](../superpowers/verdicts/2026-07-14-continuous-batching-chunked-prefill.md),
[compact evidence](../superpowers/verdicts/continuous-batching-phase3-evidence-2026-07-14.jsonl),
and [content piece](../content/2026-07-14-the-fastest-request-wasnt-the-fastest-service.md).

The promotion does not wire a production serving/API route or dynamic policy default. Dense
families beyond `qwen3`, sampled generation, MoE/hybrid/recurrent/vision state, real network
disconnect propagation, and speculation inside a shared batch require separate gates. The
serving route and disconnect gate are captured in
[`2026-07-14-continuous-batching-serving-route.md`](2026-07-14-continuous-batching-serving-route.md).
