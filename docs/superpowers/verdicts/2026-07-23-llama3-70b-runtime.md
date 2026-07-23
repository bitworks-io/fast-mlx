# Llama-3.3-70B runtime verdict — no speed tier, retain capacity context

- **Date:** 2026-07-23
- **Lane:** model-family runtime qualification; non-promotable
- **Evidence sources:** loaded speed `c8a56ef00f6137b0bebfd6e494bfd9099a6a57fd`; generalized
  capacity `349331443bc37a236b1460681fe24c9c91979089`; sealed quality/task
  `90337258aa741e987111c45e7102a53a29f7415c` (all clean)
- **Model:** `mlx-community/Llama-3.3-70B-Instruct-4bit`
- **Model revision:** `de2dfaf56839b7d0e834157d2401dee02726874d`
- **Checkpoint:** `5083c6af…be1400`
- **Tokenizer:** `da67fb22…be58ab`
- **Decision:** **Reject any speed-tier or default claim for this model cycle. Retain the
  authenticated 32K affine capacity/runtime context as non-promoted evidence.**

## Operator story and acceptance

A long-context Apple-Silicon operator needs a second-family runtime answer that does not convert
fit, diagnostic speed, or partial quality evidence into a broad fast path. The useful outcome is a
clear model-scoped verdict: what ran, what failed closed, which cells remain capacity context, and
which claims are forbidden.

Acceptance required:

- every retained row binds its clean engine source, source-locked checkpoint, tokenizer, and
  workload;
- an 8K direct candidate beats both fp16 and its same-storage materialize control under the frozen
  speed gate before it can earn a speed label;
- a 32K speed lane fails closed when the control row cannot remain in the declared thermal cohort;
- capacity evidence stays separate and explicitly non-promotable;
- long-depth quality/task promotion reaches the required scored context before it can authorize a
  public long-context tier.

The representative happy path was a retained fp16 baseline plus retained affine
materialize/direct rows with exact source and model identity. The relevant failure paths were a
direct speed-gate miss, a 32K fp16 thermal cohort failure, a conservative near-128K memory no-go
before launch, and incomplete long-depth quality promotion.

## Authenticated boundary

| Artifact | SHA-256 |
| --- | --- |
| Source-lock receipt | `145127546c6c9872e80512716494eed77905d6e3ddd398c47c8f34a5ec796a4f` |
| 8K speed manifest | `f4b3ce416605b94c8fb46fe20387610a7a49631fa859cfdda33a4b5820de3daa` |
| 8K speed completion | `ee3802178bd48738f82f7a307d8b8a45cd2a51dc8ce4a9e158bb07760e69ad0c` |
| 8K speed receipt set | `041277d4e608c13117f3f2fbdaf3d921707cf386806ede52dea36a2f36a4ba9c` |
| 32K speed manifest | `48ff1fd0da562f41011fbdff24642f1ca37ece038b864315364dc86595ebf366` |
| 32K speed failure | `1473c9e5825b059d8dd5d66b56a6f9e1d27a05dd9e6adfb98623147eacd2f533` |
| 32K speed status | `03cbb6c52e522ce55e217e3f9d66dbd66d21ef6bf85024016a3b7f3080c7944f` |
| Sealed reference manifest | `0d34303497c4f5a6c8b73a8e146cd34914adeb8adc3b4755c4d8e4983df894a1` |
| fp16 quality evidence | `57ad74aca7e60bc8a97cfcbe66955c22d52e76514575aa6ec01b5005c1ee82d1` |
| Affine quality evidence | `7feb19468f307b06084cf65166b7cc38c9bc501c1e83e885601d0abd2e714c53` |
| fp16 task raw | `91bb6b290530110244fadd21d587cf4d0727a97c76a2d30fdb0ad569a95b31e4` |
| fp16 task summary | `c9ae61ba1573cd3d93c7a0652558d406c79d3589de033b3f4fec250be4e0942e` |
| Affine task raw | `a171b49da5be1c371f5628c3efbb9d042b4cb6f4f46216310e09889391765345` |
| Affine task summary | `960c44c1e6a1e233dd70f15e530f6c909437cf7862071687630d0467470cfcbd` |
| Affine task terminal status | `5b7b29ff051dc0fb718b86fc04908fc961f0e6849a6e3993d1d0420f4cb0b97b` |
| Affine task negative-auth script | `a50d9105d1fbdabb62ecf0d49b8bab75514bf895735afff23653ebaf13ec23fa` |
| Affine task negative-auth log | `59183ca4cd024dace4771cf7695f7c8117e8edf60c629727884fc5f1c150ebbc` |

Machine-readable criterion mapping:
[`llama3-70b-runtime-verification-2026-07-23.json`](llama3-70b-runtime-verification-2026-07-23.json).

## 8K loaded speed result

| Cell | Prefill tok/s | Decode tok/s | Retained interpretation |
| --- | ---: | ---: | --- |
| fp16 | 270.06 | 11.87 | same-model runtime baseline |
| Affine K4V2-g64 materialize | 268.58 | 10.51 | same-storage control |
| Affine K4V2-g64 direct | 199.41 | 11.74 | direct path; failed speed gate |

Affine direct removed the materialize-control decode penalty but remained 1.10% below fp16 decode
and regressed prefill 26.16%. The speed gate therefore closes negative. The row proves loaded
direct-path engagement; it is not a speed tier or default candidate.

## 32K speed and capacity lanes

The 32K speed matrix failed closed on the fp16 control: the retained fp16 row moved from nominal
to fair thermal state and produced zero admissible rows. No 32K speed aggregate exists for this
model.

The separate 32K affine capacity lane retained one non-promoting runtime context at
`/Users/llmbench/perf-work/results/fused-compressed-kv-llama3-70b-loaded-3493314/llama-32k-affine-capacity-v1`.
Manifest, evidence, completion, and launch-receipt SHA-256 values are
`84151bdb34b1ac9825e9ea3dcb0f8fe549ea252e23f989e595efe0a8161e0fdb`,
`8b6e147adebd1a705103222f1978f1a86916469e96847a1a105370be83871ab1`,
`e3850c665e1a2ddf59dc23e49658e48fe961774dd958b849934f6cd0b79faf6e`, and
`35be506e873412bff3f9b3a70d3fecebf7f3a65252b8f5f133eddcfdc9df6d4c`.

| Metric | Retained value |
| --- | ---: |
| Capacity window | 32,640 + 128 tokens |
| Prefill / decode | 92.61 / 9.96 tok/s |
| Direct layers | 80 |
| Materialization bytes | 0 |
| Physical footprint bytes | 130,257,818,032 |
| Process maximum RSS bytes | 83,292,012,544 |
| Promotion state | `promotable:false`; `speedAggregation:"forbidden"` |

This is evidence that the affine direct path can run the measured 32K workload with zero
materialization across all 80 layers. It is not speed evidence because the control speed lane had
already failed closed.

## Near-128K boundary

No near-128K run launched. The authenticated 32K row measured 130,257,818,032 physical-footprint
bytes and, per 32K segment, 2,028,994,560 payload bytes, 338,165,760 metadata bytes, 320 control
bytes, and 4,328,521,728 direct-workspace bytes. Conservatively adding three more segments projects
150,344,865,136 bytes:

`130,257,818,032 + 3 × (2,028,994,560 + 338,165,760 + 320 + 4,328,521,728)`.

That exceeds the host's 137,438,953,472 physical bytes by 12,905,911,664 bytes before extra safety
margin. This is a reviewed capacity no-go derived from authenticated 32K measurements, not a
near-128K runtime row or harness-refusal artifact.

## Quality and task evidence

| Evidence | Median KL | Tail KL | Pooled median | Pooled p95 | PPL delta | Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fp16 | 9.6688e-7 | 3.3231e-5 | 4.2539e-7 | 2.9269e-5 | +0.0025966% | 327/328 |
| Affine K4V2-g64 direct | 0.0069743 | 2.5286453 | 0.0064784 | 1.0595306 | -8.8987% | 285/328 |

| Cell | Math | Code | Structured | Long | Structured validity |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp16 | 10/20 | 7/20 | 20/20 | 20/20 | 20/20 |
| Affine K4V2-g64 direct | 10/20 | 6/20 | 14/20 | 20/20 | 14/20 |

Affine quality is finite and the direct split engaged with zero materialization and zero
normalization. The paired task run completed all 80 cases but exited terminal `FAILED`: structured
validity was 14/20 against the frozen 18/20 floor, so `hardFloorPassed` and
`balancedTaskDeltaPassed` were both false. A negative-only Swift validator strictly re-reduced the
raw corpus, re-derived the fp16 comparison, and authenticated that failure without altering or
promoting the failed boundary. Independently, the maximum scored teacher-forced context was
22,541 tokens against the required 24,000-token long-depth bar. Both gates reject promotion.

## Verdict

- **No speed tier.** The 8K affine direct row is dominated by fp16 on decode and prefill, and 32K
  speed has zero retained rows.
- **No broad/default claim.** This is one source-locked Llama-3.3-70B-4bit snapshot on one hardware
  boundary.
- **Capacity context only.** The 32K affine direct run is useful retained evidence, but it cannot
  promote while the speed lane is closed and long-depth quality remains incomplete.
- **Task hard floor failed.** The 80-case affine task run is retained negative evidence; its
  structured-validity failure cannot be relabeled as coherence or a user-selectable tier.
- **No near-128K runtime claim.** The no-go is a conservative projection from authenticated 32K
  measurements, not an executed context boundary.

The current next model-family candidate is Phi-4-mini's materially different Phi3 geometry. It
must independently pass source, registry/load, runtime, quality, and failure-path proof before it
can contribute to any broader support claim.
