# Exact speculation was not fast enough

**Whitepaper themes:** Building a high-performance MLX inference engine in Swift; Rapid research
integration — the flywheel

Speculative decoding has an attractive promise: predict several future tokens cheaply, verify them
with the target model, and skip some serial decode steps without changing the answer.

fast-mlx already had a strong prompt-lookup decoder for repetitive text. The next question looked
almost mechanical: when an OpenAI-compatible request is the only active stream, let it speculate;
when another request arrives, drain the speculative state and join the continuous batch.

The implementation worked. The product decision was still no.

## The hard part was the transition

A standalone speculative decoder owns one sequence. A continuous service owns a changing set of
sequences whose requests may arrive, disconnect, or finish at any point. Combining them creates a
state transition that must be exact:

1. one request runs a bounded prompt-lookup round;
2. accepted tokens become ordinary target-model state;
3. a queued request appears;
4. any in-flight speculative pipeline drains without publishing an extra token;
5. both requests enter shared decode; and
6. speculation remains disabled until the service is solo again.

That drain must not wait for downstream publication capacity, because it produces no visible
output. It must reject stale scheduler plans before mutating cache state. Cancellation must work
before drafting, during verification, and while draining. Lossy KV state must not combine with the
exact speculative path merely because both features exist.

fast-mlx implemented this as an actor-confined incremental session. Pure tests cover bounded draft
history, stale plans, outputless drain ordering, cancellation, and shared-batch rejection. MLX
tests retain the existing mask, GQA, cache-lifecycle, hostile-compaction, and non-finite-input
contracts.

## Exactness passed

The fresh clean Qwen3-32B proof compared the solo speculative result with an independent no-spec
control. The token and byte results matched. Telemetry showed that prompt lookup engaged and
accepted draft tokens.

The same loaded model then admitted two requests together. Both routed through shared no-spec
decode, both matched the independent control, and the execution trace contained no shared
speculative operation. After completion, active requests, coordinator slots, and reserved KV bytes
were all zero.

The selected loaded test passed in 22.459 seconds on the authenticated Apple boundary. The full
Apple regression also passed: 140 harness tests, 194 engine tests, and 50 serving-adapter tests,
plus a Release build.

If correctness had been the only gate, this would have been a success.

## The speedup was 1.39%, not 5%

The promotion gate required at least 5% more C=1 completion throughput on the identical real-HTTP
workload. The comparison used the same source-locked model, prompt, 128-token output budget, warmup,
two measured bursts, and exact output hash.

| Solo policy | No-spec | Dynamic PLD | Improvement |
| --- | ---: | ---: | ---: |
| retained ngram-3 | 21.7851 tok/s | 22.0880 tok/s | **1.3902%** |
| ngram-2 follow-up | 21.8526 tok/s | 22.0981 tok/s | **1.1236%** |

The ngram-3 run drafted 12 tokens and accepted 10 across its measured bursts. We suspected that
shorter guesses might reduce rejected work. Ngram-2 drafted 14 and accepted 12—but delivered an
even smaller improvement.

That falsified the convenient hypothesis. Engagement was not the bottleneck. The accepted spans
were simply too short to repay incremental verification and transition overhead on this workload.

At higher concurrency, the service correctly disabled speculation and used shared batching. Those
measurements validate the safety policy, not a speculative speed claim. The C=8 workload also
reached physical decode width six because staggered prefill limited simultaneous decode slots; it
would be misleading to relabel it as a fixed-width B=8 kernel result.

## Why we kept the code but removed the mode

The actor-confined session and its correctness tests remain useful. They are a reusable research
seam if a future profile reveals a materially better verification strategy or a workload with
longer accepted spans.

The shipping CLI does not expose dynamic PLD. fast-mlx retains the explicit
`continuous-batch-no-spec` route that already passed its loaded disconnect, survivor, cohort, and
resource-release proof. Dynamic results are marked diagnostic-only, non-promotable, and forbidden
from speed aggregation.

This distinction matters. “Exact” answers whether an optimization is safe. “Fast enough” answers
whether users should pay its complexity and operational surface. A technique must pass both gates
in an exact performance lane.

The general lesson is simple: do not ship an inference policy because its internal counter moved
in the intended direction. Count accepted work, but measure the operator-facing rate on the real
route. When the benefit is 1.39% against a 5% gate, preserve the proof, close the unchanged retry,
and move the roadmap forward.

The technical verdict and authenticated criterion mapping live in
[`2026-07-24-continuous-serving-solo-pld.md`](../superpowers/verdicts/2026-07-24-continuous-serving-solo-pld.md).
