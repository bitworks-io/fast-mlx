# Continuous serving solo-PLD verdict — exact, engaged, and not fast enough

- **Date:** 2026-07-24
- **Lane:** exact temperature-zero serving acceleration
- **Classification:** EXACT
- **Engine source:** `520a106708f4f0d47cf8bc9f08a188078c4915d8`
- **Decision:** **SHELVE dynamic solo PLD for this serving cycle. Retain its internal
  actor-confined correctness implementation, keep it out of the shipping CLI, and continue with
  the explicit `continuous-batch-no-spec` route.**

## Operator story and hard gates

A dense-Qwen operator may begin alone and later share the resident decoder with other requests.
The solo phase may use prompt-lookup decoding only when it remains byte-identical at temperature
zero, drains into canonical scalar cache state before batch membership, disables speculation for
every shared decode, and releases cancelled or completed work without leaking slots or KV.

Correctness alone was necessary but not sufficient. Promotion also required the identical
real-HTTP C=1 workload to improve completion throughput by at least 5% over explicit no-spec.
Shared C=2/4/8 requests had to retain the existing exact no-spec frontier. Unknown state,
non-finite values, stale plans, lossy KV plus PLD, and partial evidence had to fail closed.

The happy path is solo admission with bounded PLD rounds, followed by an outputless drain before a
second request joins. Failure recovery returns to the explicit no-spec route; it never publishes a
dynamic/default mode on incomplete evidence.

## What the implementation proved

The smallest actor-confined path is complete. Pure tests cover bounded draft history, stale-plan
rejection, outputless drain ordering under publication backpressure, cancellation in PLD states,
shared-batch speculation rejection, and resource release. The MLX path preserves cache lifecycle,
masks, GQA, hostile compaction, and finite-input checks.

Clean Apple verification at `520a106` passes:

| Proof | Result |
| --- | --- |
| Focused Phase 4 pure contracts | 136/136 |
| `FastMLXHarnessTests` | 140/140 |
| `SpikeCoreTests` | 194/194 |
| `SpikeServingAdaptersTests` | 50/50, four expected loaded-environment skips |
| Release and build-for-testing | succeeded |

The fresh loaded boundary is terminal `COMPLETE` at
`/Users/llmbench/perf-work/results/continuous-serving-phase4-loaded-exact-520a106-v1`.
Exactly one selected Qwen3-32B test passed in 22.459 seconds. It proves:

- the solo PLD result equals an independent explicit-no-spec control;
- solo speculation engages and accepts draft tokens;
- two simultaneous requests route to shared no-spec decode;
- the shared execution trace contains no speculative operation;
- both shared outputs equal the control; and
- final active requests, coordinator slots, and reserved KV bytes are zero.

The launcher binds clean source and Phase 4 files, the loaded test binary and xctestrun, all model
shards and tokenizer/config files, High Power/140 W AC/Foundation state, and explicit 96-GiB MLX,
8-GiB cache, and 16-GiB aggregate-KV limits. Peak XCTest RSS was 18,261,232 KiB. Artifact and
xcresult manifests reauthenticate; no watchdog, retained lock, or orphan remains.

## The measured frontier

The real-HTTP diagnostic used the same source-locked Qwen3-32B prompt, 128 completion tokens, one
dropped warmup, two measured bursts, and exact output SHA-256
`e2bd50d266a2af3f7913eb8ad8b6c5842d4131f62fa3d3bc6dcc98ce93a7270d`.
It is diagnostic-only, non-promotable evidence; a negative can close a route, but it cannot
authorize one.

| Solo policy | Explicit no-spec | Dynamic PLD | Speedup | Required |
| --- | ---: | ---: | ---: | ---: |
| retained ngram-3 | 21.7851 tok/s | 22.0880 tok/s | **+1.3902%** | +5% |
| bounded ngram-2 follow-up | 21.8526 tok/s | 22.0981 tok/s | **+1.1236%** | +5% |

Ngram-3 drafted 12 and accepted 10 tokens across the measured C=1 bursts. Ngram-2 increased that
to 14 drafted and 12 accepted, yet became slightly less beneficial. The limiting issue is not
engagement or exactness; the small saved decode count does not repay the incremental verification
and transition overhead on this workload.

At C=2/4/8, admission correctly used shared no-spec decode. Those cells prove the safety boundary,
not a solo-PLD gain. C=8 reached physical decode width six under staggered prefill, so it must not
be described as a synthetic fixed B=8 kernel claim.

## Support and product disposition

| Surface | Disposition |
| --- | --- |
| Explicit dense-Qwen `continuous-batch-no-spec` HTTP/SSE route | retained; Phase 3 model-scoped authorization remains |
| Internal incremental PLD session and transition contracts | retained for future research |
| Dynamic solo-PLD shipping CLI/default | shelved; absent |
| Shared-batch speculation | rejected |
| Lossy KV plus PLD | rejected until separately qualified |
| Other model families | normal scalar support remains; no broad PLD claim |

This is not a rejection of prompt lookup generally. PLD remains valuable in already-qualified
workloads where accepted spans are long enough. It is a rejection of this dynamic continuous
serving policy under its declared exact-speed contract.

## Reopen condition and next action

Do not rerun the unchanged policy or tune ngram length again. Reopen only after a bounded profile
identifies one actor-confined change with a credible route past +5% on the identical C=1 workload,
then use a new clean source, fresh output, and the same exactness gate.

The next roadmap item is Phase 5: clean Release smoke, hostile transport disconnect/A/B/A proof,
resident explicit-no-spec soak, redaction and provenance verification, final review, and merge.

Machine-readable criterion mapping:
[`continuous-serving-phase4-verification-2026-07-24.json`](continuous-serving-phase4-verification-2026-07-24.json),
SHA-256 `19dc0345b0362af71aea4d503baf804cf0b0ea06ab4c491f6c76b2b14f85edd3`.
