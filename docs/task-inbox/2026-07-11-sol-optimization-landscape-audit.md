# Sol optimization-landscape audit — find the next fast-mlx multipliers

- **Captured:** 2026-07-11
- **Status:** completed 2026-07-12
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

**Resolved assumptions:** rank for Apple Silicon and the current Swift/MLX product first;
retain CUDA/NVIDIA evidence only as an algorithm or reference-kernel lead. Run a monthly
light scan, a quarterly full reconciliation, and an event-driven review when a candidate
gains an Apple implementation or product-model checkpoint.

## Outcome

The full portfolio was reconciled and the ranked queue was updated. The audit preserves the
committed Qwen3-32B EAGLE-3/DSpark preflight because a target-compatible checkpoint exists
and its 32B Apple economics are unknown; source review rejected a first-pass attempt to
compare its inclusive `acceptance_length` directly with the old 8B DSpark break-even. DFlash
and native MTP are explicit controls. The audit also restores missing incumbent work
(continuous batching, exact prefix/session reuse, request warmup, sampler fusion), ranks the
KVarN/asymmetric storage-quality gate and fused compressed-domain attention ahead of a second
TurboQuant cycle, and resolves the absorbed-MLA and PrismML status drift.

- [Dated audit and complete disposition](../reference/2026-07-12-sol-optimization-landscape.md)
- [Updated technique intake](../reference/performance-technique-intake.md)
- [Content piece](../content/2026-07-12-the-backlog-was-lying.md)

No engine technique was promoted by research. Each external result remains unverified until
its task runs on the clean-SHA bench harness.

**Execution update:** the first ranked task has now run. Qwen3-32B EAGLE-3 is shelved after
clean-SHA 4-bit and 8-bit byte-identity failures; the queue advances to continuous batching.
