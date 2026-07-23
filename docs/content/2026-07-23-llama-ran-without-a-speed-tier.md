# Llama ran, but it did not earn a speed tier

**Whitepaper themes:** Serving big models on Apple Silicon; The optimization dial — quantified
precision-loss tuning; Rapid research integration — the flywheel

The second-family gate for fast-mlx was deliberately simple: take a materially different popular
model family, bind the exact engine source, checkpoint, and tokenizer, then let the same evidence
rules decide whether compressed KV becomes a speed feature, a capacity feature, or neither.

For this pass the model was `mlx-community/Llama-3.3-70B-Instruct-4bit`, locked to model revision
`de2dfaf56839b7d0e834157d2401dee02726874d`, checkpoint `5083c6af…be1400`, and tokenizer
`da67fb22…be58ab`. Loaded speed, generalized capacity, and sealed quality/task each bind their own
clean fast-mlx source boundary (`c8a56ef`, `3493314`, and `9033725`). The answer was useful because
it was negative: fast-mlx did not earn a Llama speed tier, and it did not get a default-setting
claim.

## The 8K run closed the speed question

| Cell | Prefill tok/s | Decode tok/s | What it means |
| --- | ---: | ---: | --- |
| fp16 | 270.06 | 11.87 | same-model baseline |
| Affine materialize | 268.58 | 10.51 | compressed storage, full materialization |
| Affine direct | 199.41 | 11.74 | compressed-domain read path |

The direct path avoided full-cache materialization. But the product gate is not “better than the
worst path.” It has to beat fp16 and preserve prompt processing. Affine direct decoded 1.10% below
fp16 and prefilled 26.16% slower, so the speed gate failed.

That result is not a near miss promoted by interpretation. It is a retained negative. For this
locked Llama snapshot, affine direct is not a speed tier and not a default route.

## 32K separated capacity from speed

The 32K speed lane failed closed before any speed table could exist. The fp16 control moved from
nominal to fair thermal state, so the harness retained zero rows. There is no 32K speed claim to
average, rescue, or relabel.

A separate capacity lane produced useful context. Affine direct ran a 32,640-token prompt plus 128
generated tokens with all 80 layers engaged, zero materialization bytes, and 92.61 prefill / 9.96
decode tokens per second. The measured physical footprint was 130,257,818,032 bytes, and process
maximum RSS was 83,292,012,544 bytes.

That is a capacity/runtime fact, not a speed fact. It says fast-mlx executed this measured 32K
Llama workload through the direct affine path without rebuilding the full KV cache. It does not
say that the path is faster than fp16 under the frozen speed contract, because the speed contract
has no admissible 32K control row.

## Near-128K stopped at a reviewed memory no-go

No near-128K run launched. Scaling the authenticated 32K physical footprint, packed storage, and
direct workspace conservatively projected 150,344,865,136 bytes against 137,438,953,472 bytes of
physical memory. That is already 12,905,911,664 bytes over the host before extra safety margin.

This is not a benchmark and not a synthetic refusal result. It is a decision not to launch a
knowingly impossible soak from an authenticated lower-context measurement.

## Quality was measurable, but promotion still did not clear

The sealed fp16 replay stayed near the numerical floor: median KL `9.6688e-7`, pooled median
`4.2539e-7`, perplexity delta `+0.0025966%`, and top-1 `327/328`.

Affine direct was finite but visibly different: median KL `0.0069743`, tail KL `2.5286453`, pooled
p95 `1.0595306`, perplexity delta `-8.8987%`, and top-1 `285/328`. It used the direct split with
zero materialization and zero normalization.

| Cell | Math | Code | Structured | Long | Structured validity |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp16 | 10/20 | 7/20 | 20/20 | 20/20 | 20/20 |
| Affine direct | 10/20 | 6/20 | 14/20 | 20/20 | 14/20 |

The affine task run completed all 80 cases, but its 14/20 structured-validity score missed the
frozen 18/20 floor. The run therefore ended as authenticated negative evidence, not a quality
pass. Long-depth promotion independently remained incomplete: the maximum scored context was
22,541 tokens, below the required 24,000-token bar.

## The verdict

For this model cycle, fast-mlx keeps the evidence and rejects the claim:

- no speed tier;
- no broad Llama-family or default-setting claim;
- affine direct is negative and dominated for speed;
- the 32K affine direct run is retained only as capacity/runtime context;
- near-128K has no executed row; and
- long-depth promotion remains incomplete while the paired task hard floor also failed.

Phi-4-mini's Phi3 geometry is the current third-family candidate. It is not accepted by naming it:
the registry/load path, exact source boundary, runtime, quality, and failure behavior must all be
proven independently.

## Sources and measurement method

The exact source boundary is recorded in
[`2026-07-23-llama3-70b-source-lock.md`](../superpowers/verdicts/2026-07-23-llama3-70b-source-lock.md).
The runtime, capacity, quality, and task boundaries are recorded in
[`2026-07-23-llama3-70b-runtime.md`](../superpowers/verdicts/2026-07-23-llama3-70b-runtime.md).
Both report fast-mlx evidence only.
