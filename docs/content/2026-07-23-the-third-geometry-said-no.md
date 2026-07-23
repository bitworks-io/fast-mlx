# The third geometry said no

**Whitepaper themes:** Serving big models on Apple Silicon; The optimization dial — quantified
precision-loss tuning; Rapid research integration — the flywheel

The third-family gate was not supposed to reward novelty. Qwen and Llama
had already exercised large uniform-GQA shapes, but both shared the same Q64/KV8/D128 attention
geometry. A broader product claim needed a materially different model family with different head
shape, different registry identity, and different failure modes.

The candidate was a source-locked Phi3-family model with Q24/KV8/D128 geometry. The model proved
that the fast-mlx registry, load path, and direct affine KV path
were portable beyond the earlier geometry. The engine did not merely pass synthetic shape tests.
It loaded the locked model, routed the real architecture, and executed affine-direct decode without
falling back to a full-cache materialization path.

That sounds like a success if the only question is architecture coverage. It is not enough for a
speed tier.

## The 8K run split decode from prefill

At 8K, affine direct did something the previous model family had not done: it won decode.

| Cell | Prefill tok/s | Decode tok/s | What it means |
| --- | ---: | ---: | --- |
| fp16 | ~3,541 | 124.80 | same-model baseline |
| Affine materialize | 3,538.09 | 90.57 | compressed storage, full materialization |
| Affine direct | 2,011.61 | 148.54 | compressed-domain read path |

The direct affine row decoded at 148.54 tokens per second, above fp16 at 124.80 and far above the
materialize control at 90.57. If the gate had been a decode-only microbenchmark, this would have
looked like the missing speed result.

The full request said otherwise. Prefill collapsed to 2,011.61 tokens per second against the
roughly 3,541-token-per-second control band, about a 43% regression. The product gate cannot hide
that cost by celebrating the last-token loop. Users pay prefill on real prompts, and long-context
users pay it heavily. A path that accelerates decode while cutting prompt processing almost in
half is not a default speed tier.

So the 8K verdict stayed narrow: registry/load/direct-path portability was proven, and the decode
kernel showed a real positive signal. The end-to-end speed claim failed.

## Fit did not become speed

The longer runs were useful for a different reason. Affine direct fit 32K, and it also fit a
131,072-total-token run. Those facts matter for capacity planning: they show the compressed path
can carry much larger retained contexts for this source-locked Phi3 geometry.

They do not repair the 8K speed failure. The long-context rows were retained only as capacity
context, with speed aggregation forbidden. That keeps the frontier honest. A capacity lane
answers "can this workload
execute within memory?" A speed lane answers "is this path faster than the admissible control
under the frozen measurement contract?" Mixing the two would turn a fit result into a throughput
claim it did not prove.

That separation is especially important for compressed KV work. Storage savings, direct attention
reads, thermal behavior, allocator pressure, prompt length, and decode-loop throughput can all move
in different directions. Capacity is not a consolation speed metric.

## Finite logits were not enough

The quality replay added a more nuanced result. Teacher-forced affine replay scored 328 positions,
with the deepest scored position at 27,145 tokens. It was finite and cleared fast-mlx's frozen
non-garbage hard floor, so the measured option was not rejected as incoherent. It was also far
from transparent:

| Metric | Affine replay |
| --- | ---: |
| Median KL | 0.214878 |
| Pooled p95 KL | 1.42325 |
| Perplexity delta | +33.13% |
| Top-1 agreement | 77.44% |

Those numbers are usable evidence precisely because they are not hand-waved away. Median KL above
0.21, pooled p95 above 1.42, perplexity up by a third, and top-1 agreement under 78% describe a
model whose logits remain well-formed while its distribution has moved materially. The
non-garbage floor preserves room for an explicitly lossy option; it is deliberately lower than a
transparent-quality claim. A user-selectable tier still needs the separate paired task gate.

The task lane closed even harder, but in an unusual way: the fp16 task baseline itself was invalid.
It scored 8/20 math, 1/20 code, 15/20 structured, and 20/20 long, with only 15/20 syntax-valid
structured outputs. That is not a stable control for measuring the candidate's task degradation.

The tempting mistake would be to weaken the floor until both rows could be compared. fast-mlx did
the opposite. The candidate task comparison was refused. A bad baseline does not license a softer
product contract; it invalidates that particular task comparison. The teacher-forced measurement
remains valid and explicitly lossy, but no Phi quality tier is exposed without the paired task
evidence.

## The verdict

For this source-locked Phi3 geometry, fast-mlx keeps four separate facts:

- architecture portability passed;
- 8K affine direct produced a decode win but failed the end-to-end speed gate because prefill
  regressed by about 43%;
- 32K and 131,072-total-token affine runs are capacity context only, with no speed aggregation; and
- teacher-forced evidence cleared the non-garbage floor with material measured drift, while the
  invalid fp16 task baseline prevented a paired task comparison and therefore any quality tier.

That is not a wasted third-family cycle. It is the point of requiring independent gates. An
architecture can load without earning speed. A model can fit without proving throughput. A replay
can clear a non-garbage floor without earning a quality tier. A task suite can run without
producing an admissible comparison.

The general lesson is the one compression work keeps forcing back into view: architecture
portability, fit, speed, and quality are independent product claims. The cleanest evidence is not
always the evidence that opens a tier. Sometimes the third geometry says no, and closing the claim
honestly is the result.

## Sources and measurement method

This piece reports fast-mlx's own source-locked Phi3-family measurements only. The candidate was
selected from the third-geometry gate recorded in the
[fused compressed KV attention plan](../superpowers/plans/2026-07-18-fused-compressed-kv-attention.md).
