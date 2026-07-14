# Continuous batching with decode-first chunked prefill — PROMOTE

**Date:** 2026-07-14 · **Final engine/harness evidence:** `7a775f6f1db9495d60eecdf030bf63d752f936e0` (clean)
**Box:** M5 Max, 128 GB · macOS 26.5.2 · **Build:** Release · `Memory.cacheLimit = 8 GiB`
**Model:** Qwen3-32B-4bit, dense `qwen3`, `int4:group=64` weights, fp16 KV
**Runtime:** MLX Swift 0.31.6 · mlx-swift-lm `702e5a0eaf990e1f6d3db2b6e7d8872858a44055`
**Identity:** config `b3f033c21f563996` · checkpoint manifest `33827ddf1b497615`
**Evaluation lane:** [`EXACT`](README.md)—scheduling may change when target work runs, never
the temperature-zero greedy result
**Decision:** **PROMOTE the exact dense-Qwen3 continuous runtime as a measured service-policy
building block; the measured performance and stability gate is clear.**

This decision promotes a bounded runtime and policy result, not a deployed service default.
The continuous path remains reachable through explicit harness/probe surfaces; no serving or
API route selects it yet. The measured policy is: prefer solo PLD for an isolated request, and
prefer no-speculation continuous batching for the tested simultaneous bursts at concurrency
two or greater. Dynamic routing is separate product work, and PLD remains unavailable inside
a shared batch.

## What passed the exact gate

The final-SHA real-model probes exercise the cache histories that ordinary single-request
tests miss:

| Gate | Result |
| --- | --- |
| Solo → shared batch → solo | Qwen3-32B-4bit B1→drain→B2→B1; both streams token- and byte-identical to independent scalar baselines. |
| Middle cancellation | Qwen3-32B-4bit B3→B2; both survivors remain exact and the cancelled request equals its scalar prefix. |
| Adversarial chunk boundary | Qwen3-4B-4bit with `prefillChunkSize=1`; decode interleaves with prefill and every stream remains exact. |
| Fail-closed architecture scope | `model_type=qwen3_moe` is rejected before entering the dense runtime. |
| Shared-batch engagement | Every expected shared-batch histogram is present and exact; batched speculation is zero. |

The load-bearing transition remains drain-before-join. A solo compiled decoder may hold one
lookahead token whose forward has already modified KV state. Before another row joins, the
runtime commits that pending state and only then forms the shared decode batch. The exact
probe proves the resulting stream, rather than inferring correctness from queue state.

This `EXACT` classification creates no quality-loss dial axis. The user-visible choice is an
operational frontier—aggregate service throughput versus TTFT, TPOT, and completion
fairness—not permission to change model output. That rule is deliberately narrower than the
product's treatment of intentional approximations: a `LOSSY_FRONTIER` technique may still be
promoted with real, teacher-forced quantified loss when it remains useful and above the
coherence floor.

## The preliminary frontier was invalid

The first policy sweep cannot support a verdict. Each policy process generated its own random
salt nonce, so rows presented as direct comparisons had run different prompt bytes. Those
preliminary rows are **disqualified**, not blended into the result below.

The corrected runner pins one workload identity across all eight cells:
`workloadNonce=frontier-20260714-final`. For a given run and request index, batch-no-spec and
solo-PLD therefore tokenize the same salted prompt. The two arms also produced the same prompt
and visible-output counts for every paired run. Each cell drops one warmup and aggregates three
measured runs. Aggregate rate uses the complete client-observed burst makespan, including
queueing and prefill; TPOT is computed per request after its first token. These are service
metrics, not decode-only tok/s.

Raw corrected frontier artifacts:

- JSONL SHA-256 `9404c0aa699ecc37a5575ceb5e13dd2ec75908cc47e06d7b1ee0cf7251c1f70c`
- CSV SHA-256 `122d5638ee687873941c1fbe98ce8136b707f86423293ee9715fad3321861d68`

## Corrected same-workload service frontier

| Policy | C | Aggregate tok/s | TTFT p50 / p95 (ms) | TPOT p50 / p95 (ms) | Jain mean / min |
| --- | ---: | ---: | ---: | ---: | ---: |
| batch-no-spec | 1 | 26.72 | 348.4 / 348.6 | 35.0 / 35.0 | 1.0000 / 1.0000 |
| batch-no-spec | 2 | 42.70 | 729.2 / 740.0 | 41.5 / 41.8 | 1.0000 / 1.0000 |
| batch-no-spec | 4 | 51.44 | 1370.0 / 1379.3 | 66.5 / 69.9 | 1.0000 / 1.0000 |
| batch-no-spec | 8 | 56.56 | 2628.1 / 2711.2 | 120.1 / 125.3 | 1.0000 / 1.0000 |
| solo-PLD | 1 | **28.30** | 170.6 / 182.5 | 35.0 / 35.5 | 1.0000 / 1.0000 |
| solo-PLD | 2 | 29.29 | 4482.9 / 4844.3 | 34.9 / 35.5 | 0.9081 / 0.9019 |
| solo-PLD | 4 | 32.44 | 8039.9 / 13879.0 | 33.2 / 35.5 | 0.7940 / 0.7754 |
| solo-PLD | 8 | 32.37 | 15386.6 / 28083.8 | 33.2 / 35.0 | 0.6263 / 0.5765 |

Batching is the wrong arm for one request: 26.72 versus 28.30 tok/s, **−5.6%**. At the measured
simultaneous bursts, the policy reverses: batch-no-spec beats queued solo PLD by **+45.8% at
C=2, +58.6% at C=4, and +74.7% at C=8**. Batch throughput scales **2.12×** from C=1 to C=8.

The latency distributions explain the rate rather than hiding it. Serialized PLD preserves a
roughly 33–35 ms TPOT once a request runs, but queueing drives C=8 TTFT to 15.4 s p50 / 28.1 s
p95 and Jain fairness to 0.6263 mean. Batching makes each row's TPOT slower as concurrency
rises, but advances the rows together: C=8 TTFT is 2.63 s p50 / 2.71 s p95 and Jain fairness is
1.0000. The product policy is therefore workload- and concurrency-aware, not “batch always” or
“PLD always.” This one corrected burst frontier does not establish a universal crossover for
other prompt shapes.

## Byte admission, cancellation, and recovery

Admission is now expressed in dense-Qwen KV bytes rather than only logical tokens. It includes
allocation-chunk rounding, per-row metadata, and a conservative five-copy membership-transition
envelope covering the old batch, extracted scalar rows, tail-zero inputs, padded-row
concatenations, and the rebuilt batch. That is exact array geometry under a deliberately
conservative reservation policy; it is **not** a claim that the reservation equals whole-process
RSS or generalizes to another architecture's state.

The Qwen3-4B cancellation bench on clean harness
`a2bf00a7f85e5396f106c1c7eb4bfed78915cfdf` ran five measured repetitions. Runtime slot removal measured
19.542 µs p50 and 21.042 µs p95/max, below the configured 1 s keepalive boundary. The slot was
removed and reused, the survivor completed, and reservations returned to zero. Raw artifact
SHA-256 is `963feb1a1ebebf13f35c19a3246fbfa4f09b23adc5d5648a587dca5cdd9367df`.

The separate Qwen3-4B A/B/A recovery bench ran three measured repetitions. A known-good burst,
a hostile disconnect/cancellation cycle, and the repeated known-good burst produced
byte-identical before/after output per request. Cancellation p95/max was 12.583 µs and resource
reservations again returned to zero. It used the same clean `a2bf00a…` harness; raw artifact
SHA-256 is `b9f21fd01a315a1a2b0430392493e77b0ae5f1c61df1c1a72842aa0e7ef420dc`.

These are runtime cancellation measurements. An actual network client's disconnect-to-runtime
cancellation propagation remains part of the unwired serving/API boundary and is tracked in
the [production serving-route task](../../task-inbox/2026-07-14-continuous-batching-serving-route.md).

## The 24-hour resident soak

The required soak used Qwen3-32B-4bit on clean harness
`0aed28087a3f495e1582977e263bc30e0986703d`. It completed a warmup, then
measured for **86,412.8508 seconds** against the 86,400-second target. There were 3,519 total
cycles and 3,518 measured post-warmup cycles. All 33 predicates recorded 3,519 passes and zero
failures, including exact A/B/A output, shared-batch engagement, decode-first interleave,
no batched speculation, cancellation cleanup, replacement-slot reuse, and released KV
reservations.

| Stability gate | Result |
| --- | ---: |
| Warmup RSS baseline | 21,291,824,760 bytes |
| Terminal RSS | 21,769,271,000 bytes |
| Peak sampled RSS | 21,769,697,008 bytes |
| Terminal RSS drift | +2.2424% |
| Peak RSS drift | **+2.2444%**, below the 5% gate |
| Runtime cancellation | 13.789 µs mean; 28.833 µs max, below 1 s |
| Responsiveness | 344.469 ms max, below 30 s |

No watchdog-timeout record was produced, and the wrapper cleaned its lock and resident process.
The raw soak artifact SHA-256 is
`dcc7e446234cf6defc8746964bdefe3eb80fabe543e93078a096eeb3b9452b22`. The memory verdict uses the completed warmup
as its baseline and gates the **peak**, not merely the terminal sample.

The artifact intentionally retains full detail for the warmup and 288 measured checkpoint
cycles rather than all 3,519 cycles. Its final aggregate contains all 33 counters; the
companion run log contains one A/B/A PASS line per cycle. Thus the full-run claim is supported
by the live aggregate plus complete cycle log, while independent field-by-field reconstruction
is possible for the 289 retained detailed cycles.

## Final verdict — PROMOTE, with a named product boundary

**PROMOTE** exact dense-Qwen3 continuous batching with decode-first chunked prefill as a
measured service-policy building block:

- temperature-zero transition and cancellation streams remain exact;
- simultaneous C≥2 bursts occupy a useful, non-vacuous service frontier against queued solo
  PLD, while isolated requests retain the faster PLD arm;
- cancellation, A/B/A recovery, conservative byte admission, responsiveness, and the 24-hour
  peak-RSS gate pass; and
- unsupported state layouts and shared-batch speculation fail closed.

This clears the measured performance/stability gate. It does **not** claim that a runtime
default changed, a serving/API route was wired, or every dense architecture was validated.
Current scope is dense `qwen3`, fp16 KV, greedy temperature zero, Qwen3-32B-4bit on one M5 Max,
with Qwen3-4B used for bounded cancellation and chunk-boundary probes. Sampled generation,
MoE, hybrid/recurrent state, vision, other model families, other Apple boxes, prefix reuse, and
speculation inside a shared batch remain outside this verdict.

Broadening the promotion requires each new architecture/state layout to prove merge/extract and
temperature-zero equivalence, then remeasure its own same-workload service frontier and resident
stability. Production routing additionally needs the serving/API boundary, real client
disconnect propagation, and an explicit policy implementation; it must not be inferred from
this probe-only result.

## Evidence artifacts

| Artifact | Provenance |
| --- | --- |
| Corrected frontier JSONL | SHA-256 `9404c0aa699ecc37a5575ceb5e13dd2ec75908cc47e06d7b1ee0cf7251c1f70c`; clean harness `7a775f6f1db9495d60eecdf030bf63d752f936e0` |
| Corrected frontier CSV | SHA-256 `122d5638ee687873941c1fbe98ce8136b707f86423293ee9715fad3321861d68` |
| Final-SHA exact probes | SHA-256 `5db4bb5fecc853f04fd5869326c730b702b19511b3d28a271063a9c336b9d0de`; clean harness `7a775f6f1db9495d60eecdf030bf63d752f936e0` |
| Cancellation bench | Raw SHA-256 `963feb1a1ebebf13f35c19a3246fbfa4f09b23adc5d5648a587dca5cdd9367df`; clean harness `a2bf00a7f85e5396f106c1c7eb4bfed78915cfdf` |
| A/B/A recovery bench | Raw SHA-256 `b9f21fd01a315a1a2b0430392493e77b0ae5f1c61df1c1a72842aa0e7ef420dc`; clean harness `a2bf00a7f85e5396f106c1c7eb4bfed78915cfdf` |
| 24-hour soak | Raw SHA-256 `dcc7e446234cf6defc8746964bdefe3eb80fabe543e93078a096eeb3b9452b22`; clean harness `0aed28087a3f495e1582977e263bc30e0986703d` |
| Compact committed extract | [`continuous-batching-phase3-evidence-2026-07-14.jsonl`](continuous-batching-phase3-evidence-2026-07-14.jsonl), SHA-256 `6c188275f2cad49010e0473356dacf3629d066b75b9670f3270a4771fcb97167` |
