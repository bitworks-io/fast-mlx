# Sol optimization-landscape audit — find the next fast-mlx multipliers

- **Captured:** 2026-07-11
- **Status:** backlog
- **Task type:** portfolio audit + deep research
- **Priority:** ongoing discovery; do not displace the committed DSpark queue without evidence
- **Owner:** Sol / fast-mlx

## Raw request

> Identifying new optimizations, enhancements, or unique quantizer setups, and anything else
> that can improve the engine is always being searched for. Let Sol analyze the full set of
> existing tasks, identify anything that is worth investigating, and perform deep research
> online.

## Light triage

First reconcile the existing task inbox, plans, verdicts, performance-technique intake, and
carry-forward backlog so the research does not recreate a shelved technique or duplicate an
active task. Then use current primary sources—papers, official implementations, and upstream
engine documentation—to surface evidence-backed candidates across decode, prefill, batching,
memory/KV, kernels, scheduling, and unusual weight/KV quantizer combinations.

**Open questions:** desired research cadence; whether candidates should be ranked only for
Apple Silicon today or also for the planned NVIDIA/cloud research plane.

**Next planning step:** produce a deduplicated candidate brief that links every recommendation
to the existing task it strengthens or to a new technique-intake entry; do not implement or
reprioritize the queue until the owner reviews it.
