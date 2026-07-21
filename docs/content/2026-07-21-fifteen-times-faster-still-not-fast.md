# Fifteen times faster still was not fast

**Whitepaper themes:** Building a high-performance MLX inference engine in Swift; The optimization
dial — quantified precision-loss tuning; Rapid research integration — the flywheel

Compressed KV attention had one clear job: keep the cache compressed while attention reads it.
The previous fast-mlx KVarN path saved memory but reconstructed full K/V tensors on the hot path.
That made it a useful capacity feature and a poor speed feature.

The direct implementation fixed that structural problem. On a diagnostic Qwen3-32B-4bit 8K run,
direct KVarN decoded at 7.18 tokens per second while the same compressed storage followed by full
materialization decoded at 0.46. Removing materialization made decode about 15.6 times faster.

That sounds like a win until the real control enters the table. The same fp16-KV workload decoded
at 23.48 tokens per second. Direct KVarN reached only 30.6% of fp16. Prefill was further away:
63.26 tokens per second for direct KVarN versus 533.73 for fp16.

These are engineering diagnostics, not promotion benchmarks. The direct row failed the retained
thermal-equality contract and never became promotable evidence. We use it here for the question it
can answer: is one more focused optimization likely to cross the frozen gate? It cannot support a
user-facing speed claim.

## The gate prevented a victory lap

The gate was fixed before the run. A speed candidate must beat both fp16 and its same-storage
materialize control by at least 5% in decode, while prefill may regress no more than 5%.

For this workload, direct KVarN would need at least 24.65 decode tokens per second and 507.04
prefill tokens per second. That means roughly 3.43 times the observed decode rate and 8.02 times
the observed prefill rate.

A 15.6x local improvement did not matter because it optimized the worse control. The product
control was fp16.

The source map explained why another small patch was unlikely to close the gap. Long prefill uses
512-token graph boundaries, while KVarN packs completed tiles on the host and runs an
eight-iteration normalization transform. Direct attention then performs capacity-wide packed key
and value work at every layer. There was no single synchronization, allocation, or loop removal
with an honest path to both a 3.4x decode gain and an 8x prefill gain.

## The profiler failed safely too

We attempted a bounded Metal System Trace to attribute the remaining work. The host was already in
High Power Mode, on a 140 W adapter, with Foundation reporting Low Power Mode off and nominal
thermal state before and after.

The recorder reported reaching its 240-second limit, produced more than 16 GB of raw trace data,
then remained live while its RSS rose above 100 GiB. A save watchdog terminated it rather than
letting a diagnostic consume the machine indefinitely. `xctrace export` then reported
`Document Missing Template Error`, so the trace was preserved as failure evidence and excluded
from all conclusions about individual Metal kernels.

This distinction matters. A failed profiler does not create permission to guess. The wall-time
envelope and source map were enough to answer the engineering-scope question, so we stopped there.

## The clean matrix closed the remaining escape hatch

We still owed the affine and frozen KVTuner paths a proper test. A fresh five-cell matrix ran each
direct path, its same-storage materialize control, and fp16 once in every position across five
blocks. Every retained row proved a 60-second nominal admission window, AC power, Low Power Mode
off, stable before/after state, exact model and workload identity, and the direct operation that
actually engaged. All 25 rows authenticated.

The direct paths again fixed a real local cost. Affine direct decoded at a median 24.19 tokens per
second versus 19.64 for affine materialize. KVTuner direct reached 23.19 versus 19.58 for its
materialize control. But fp16 decoded at 23.32, so affine's median advantage was only 3.7%, below
the frozen 5% gate, while KVTuner was slightly slower. Position-by-position evidence was stricter:
neither candidate beat fp16 by 5% in every block.

Prefill made the verdict unambiguous. Both direct candidates prefilled at roughly 333 tokens per
second versus 531 for fp16, a regression of about 37% at the median. The allowed regression was
5%. Faster decode than a materialize control could not offset slower prompt processing and higher
time to first token on the complete workload.

The result is clean negative evidence, not a near miss promoted by averaging. Both direct paths
remain useful implementation and capacity research, but neither earns an 8K speed label or a
default change.

## Keep the capacity, drop the speed story

KVarN remains useful. Its previously qualified i8 cell provides a model-scoped capacity-only
Max-fit choice with measured quality and task warnings. The direct implementation and lifecycle
tests remain valuable research infrastructure. What is shelved is the speed role for this cycle.

The 8K affine/KVTuner speed gate is now closed negative. The next evidence boundary is 32K, where
long-context runtime and capacity scaling are separate questions. KVarN will be measured there as
capacity/runtime context only, with no speed label implied.

The general lesson is less obvious than “optimize the bottleneck.” Relative speedups are only as
useful as the baseline they beat. A new path can be fifteen times faster than the implementation it
replaces and still be three times too slow for the product. Freeze the product gate first, preserve
negative evidence, and stop when the remaining multiplier requires a new research program rather
than one more patch.
