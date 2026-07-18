# KVarN / asymmetric KV-cache frontier verdict - PROMOTE selected lossy tiers

- **Date:** 2026-07-18
- **Final runtime evidence:** `f88d26eeb3793cc1d5d8f2118043c190977ee6e0` (clean)
- **Box:** M5 Max, 128 GiB, macOS 26.5.2
- **Model:** Qwen3-32B-4bit, same weights throughout
- **Runtime setup:** Release, `Memory.cacheLimit = 8 GiB`, raised wired-memory limit, max 256
  output tokens, 64 prompt tokens, three post-warmup identical long-form workload runs
- **Evaluation lane:** [`LOSSY_FRONTIER`](README.md) - intentional KV approximation is allowed
  only as a measured speed/memory/quality trade, not as an exactness claim
- **Compact evidence:** [`kvarn-kv-frontier-evidence-2026-07-18.jsonl`](kvarn-kv-frontier-evidence-2026-07-18.jsonl)
- **Decision:** **PROMOTE fp16 KV as the Transparent baseline, affine K4V2-g64 as the Balanced
  capacity tier, frozen KVTuner as an explicit Max-fit capacity tier, and KVarN i8 as an explicit
  capacity-only Max-fit tier plus fused-kernel candidate. SHELVE the retained K8 affine cells and
  KVarN i16. REJECT the g128 4-bit affine cells that hit the hard floor.**

This verdict is Qwen3-32B/model-specific. It makes no broad-model claim until a materially
different popular family validates the same storage frontier.

## Gates and invariants

All promoted lossy cells are judged against the predeclared hard coherence floor: finite scored
positions, marginal candidate perplexity below 2x the same-weights fp16-KV reference, long-context
tail-p95 below 5 nats, teacher-forced top-1 agreement at least 50%, no task domain below both
chance/empty-baseline and 50% of its fp16 score, and at least 90% syntactically valid
structured/tool outputs.

The evidence keeps the important invariants intact:

- Temperature-zero speculative decoding remains an exactness contract. Lossy KV plus PLD remains
  rejected until separately qualified; no speed result here is a spec-decode claim.
- Quality loss is teacher-forced against a locked reference context. Free-running tasks are
  secondary coherence checks.
- Actual storage is cache-array `nbytes` plus workspace. There is no fp16 fallback hidden inside
  the lossy rows.
- MLX cache state remains actor-confined. KVarN i8 uses the explicit uncompiled correctness path;
  this preserves the cache/actor contract but is not a compiled runtime or speed path.
- Frozen KVTuner is a fixed schedule cell, not online tuning during evaluation.

## Measurement types

The fp16 24,150-token KV size is **exact-accounted**, not a serialized fp16 frontier row:
`24,150 tokens * 262,144 bytes/token = 6,330,777,600 bytes`. Runtime dense allocation behavior
corroborates that formula, but the lossy rows below are the ones with directly measured actual
total bytes including workspace.

Capacity ratios and runtime deltas are **derived** from those exact/direct measurements. Public
competitor comparison is **not measured** and is intentionally absent.

For frozen KVTuner and KVarN rows, the measured total below excludes a separate 256-byte control
record. Capacity ratios are derived from the measured totals as originally reported; the 256 bytes
are immaterial at this precision but are recorded explicitly in the compact evidence.

## Storage and quality frontier

| Cell | Actual total bytes @24,150 incl workspace | Capacity vs fp16 | Teacher KL median | Tail-p95 | Ppl delta | Top-1 | Task scores math/code/structured/long | Syntax valid | Dial outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| fp16 KV | 6,330,777,600 exact-accounted | 1.0000x | 0.0013420716 | 0.0041823454 | -0.007603% | 0.984756 | 16/10/20/20 | 20 | **PROMOTE Transparent baseline** |
| affine K4V2-g64 | 1,483,776,000 measured | 4.2667x | 0.0755989834 | 0.816420590 | +4.577062% | 0.847561 | 16/9/20/20 | 20 | **PROMOTE Balanced capacity tier** |
| affine K8V2-g64 | 2,275,123,200 measured | 2.7826x | 0.0599400843 | 0.280318425 | +4.746621% | 0.878049 | 16/7/20/20 | 20 | **SHELVE** |
| affine K8V2-g128 | 2,176,204,800 measured | 2.9091x | 0.0725969050 | 0.331127105 | +10.838807% | 0.875000 | 11/8/20/20 | 20 | **SHELVE** |
| frozen KVTuner | 1,403,404,800 measured | 4.5110x | 0.0774932328 | 0.649152545 | +10.005041% | 0.826220 | 15/9/20/20 | 20 | **PROMOTE Max-fit capacity tier** |
| KVarN i8 | 1,496,670,208 measured | 4.2299x | 0.0013420716 | 0.343278648 | +2.857380% | 0.899390 | 9/11/20/20 | 20 | **PROMOTE capacity-only Max-fit / fused candidate** |
| KVarN i16 | 1,496,670,208 measured | 4.2299x | 0.0013420716 | 0.344238191 | +4.582705% | 0.902439 | 12/10/20/20 | 20 | **SHELVE** |

`affine-k4v2-g128` (5/4/0/16, syntax 0) and `affine-k4v4-g128` (6/5/1/18, syntax 7) are
**REJECTED**. They cross the hard floor and are not user-selectable tiers.

## Runtime frontier

No lossy tier is a speed win in this evidence. Decode and prefill deltas below are derived
against the fp16 runtime row.

| Cell | Decode tok/s | Decode delta | Prefill tok/s | Prefill delta | Peak RSS bytes | Runtime evidence SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fp16 KV | 28.46 | baseline | 376.33 | baseline | 18,595,528,704 | `6df3a4792da0093cf2b952db7b19f31f77e11c7ab44c0ff5de391d13e4c8bd1e` |
| affine K4V2-g64 | 27.52 | -3.30% | 360.54 | -4.20% | 18,602,278,912 | `17b5bf80a126a9062fbaa0e4237a1adf002f8bc09d2867abda321793adae067b` |
| affine K8V2-g64 | 27.51 | -3.34% | 365.86 | -2.78% | 18,603,048,960 | `3dc4b92797a6f4457a5982ee7832f9000c1e48451bfbbf0998a1a5797ab18e7f` |
| affine K8V2-g128 | 27.55 | -3.20% | 327.43 | -12.99% | 18,602,393,600 | `d0b971f666fd6b979a9457a839b9b567109e95e20683bb714208a53c6163a496` |
| frozen KVTuner | 27.53 | -3.27% | 319.00 | -15.23% | 18,765,217,792 | `1f93e1fe49b1a9adbdc847d9cb218370acad4c7d6378edcd581bbff9cb5097b7` |
| KVarN i8 | 13.74 | -51.72% | 276.46 | -26.54% | 18,599,608,320 | `c1acf216b7281cf05a611485b1e5f5577b2b9ce140ffa80d630276ef98913a87` |
| KVarN i16 | 13.40 | -52.92% | 276.17 | -26.61% | 18,601,312,256 | `cac7e3a298ffb2c0abe0e129bdaf151654bb47437217b88fcf83732f9e33517d` |

Process RSS is reported as runtime context, not the capacity denominator. The useful capacity
claim comes from the actual KV/cache storage budget at the locked 24,150-token geometry.

## Per-cell decision

**fp16 KV - PROMOTE Transparent baseline.** This is the default quality reference and the
capacity denominator. Its 6,330,777,600-byte 24,150-token size is exact-accounted dense KV
storage, not a lossy storage artifact.

**affine K4V2-g64 - PROMOTE Balanced capacity tier.** It delivers 4.2667x KV-budget capacity
with bounded loss: +4.577062% pooled perplexity, 0.0755989834 median KL, all task domains within
one absolute point of fp16, and 20/20 syntax validity. It is not default-eligible because it is
measurably lossy and short decode is 3.30% slower, but it is a useful user-selectable capacity
point.

**frozen KVTuner - PROMOTE explicit Max-fit capacity tier.** It is the smallest actual long-cell
storage row at 1,403,404,800 bytes, or 4.5110x capacity versus fp16. The cost is substantial:
+10.005041% perplexity and +0.0774932328 median KL, plus prefill 15.23% below fp16. It clears
the hard floor and belongs only behind an explicit maximum-fit warning.

**KVarN i8 - PROMOTE explicit capacity-only Max-fit tier and fused-kernel candidate.** KVarN is
not shelved because i8 is non-dominated in the measured frontier: its aggregate median landed at
the fp16 pipeline floor at reported precision, while tail, perplexity, and task results still
differ; it also has materially better tail than K4V2-g64 and KVTuner, 4.2299x capacity, and passes
every hard-floor predicate. The current path is an explicit uncompiled correctness path with
-51.72% short decode and a severe math-domain delta, so it is not a speed tier. It is the right
KVarN cell to carry into the next fused compressed-domain attention gate, together with the shared
affine/KVTuner storage primitives.

**KVarN i16 - SHELVE.** It uses the same actual storage as i8 and is slower on the measured
runtime path. Its task and teacher metrics do not create a consistent same-byte/runtime win over
i8 or the promoted affine/KVTuner cells.

**affine K8V2-g64 and K8V2-g128 - SHELVE.** K8V2-g64 has somewhat lower teacher-forced KL than
K4V2-g64, but it gives up too much capacity at essentially the same decode rate to justify a
separate user-visible tier. K8V2-g128 has neither the best capacity nor the best quality and
crosses into Max-fit-level loss without a compelling role. Both remain useful controls, not dial
exposures.

**affine K4V2-g128 and K4V4-g128 - REJECT.** Both hit the predeclared hard floor on task and
syntax evidence. They are not shelved for lack of product polish; they are rejected as corrupted
lossy tiers for this model.

## Reopen criteria

Reopen default eligibility only if a lossy tier clears the Transparent contract and provides a
useful measured capacity or runtime advantage. Reopen K8 exposure only with a non-dominated
capacity/quality/runtime point that creates a user-visible tier distinct from K4V2-g64 and
KVTuner. Reopen KVarN i16 only with a same-byte quality or runtime advantage over i8.

The next implementation gate should be fused compressed-domain attention for KVarN i8 plus the
shared affine/KVTuner storage primitives. That gate must measure the compiled runtime directly;
this verdict supplies no fused-kernel speed claim.

## Evidence artifacts

| Artifact | Provenance |
| --- | --- |
| Task/KL source and harness | Source/harness SHA `d9071a93955be2e148dc381638d4a71a8286c59e`; this identifies the task/KL source and harness, not an evidence artifact hash |
| KVarN canonical memory artifact | SHA-256 `7b21459cf1afc1a038b87c017c1fdacd18644b08b4b63a07318d74faba2dcfd2` |
| KL manifest | SHA-256 `3be8693aa169e2e5b4c2692f7fdc115783271fb735d27e50b0d5c3cb798990df` |
| KL recovery finalization | Original runner status preserved as `ABORTED` after a post-measurement pipefail; reviewed finalizer SHA-256 `096c38dfcb648bf7cf3870c12f24b9419bff398668a0ef13402c1009f0dcdeb4`, recovery record SHA-256 `9548bd792bcd836a92b9612a70026ad54c77adc16395df1e0b6a1fed3a8d7077`, recovered completion SHA-256 `732d4e3486a8502dd09ec44628875d1e5435fbd3648ec6c8024b8f01047de4dc` |
| Runtime evidence rows | Full per-cell SHA-256 values listed in the runtime table above |

Per-cell task/KL source hashes are retained in the compact evidence and summarized here without
machine-local paths.

| Cell | KL evidence SHA-256 | KL sidecar SHA-256 | Task raw SHA-256 | Task summary SHA-256 |
| --- | --- | --- | --- | --- |
| fp16 KV | `0545e1405d57ce857de0613b8dd6c1bb10c9b1a5b63b4486d6f1e9a370d008c8` | `c3c1d1833b1ff75aa9e44f9ceaf030203a283aa00c67c41cd7ddd2fdb26b5713` | `577ac2feb177dfea95e89f6d37aff27e1fc805a3ea8d8084b911556958478e28` | `18d3922b42d0f81cb9920caca04fbc433b847d2e5da48c80362060b3334c30c1` |
| affine K4V2-g64 | `34b23e9470945f279165b4e08304bc5ede83808b896482e5ed184c0af760ff18` | `79a731145e19c6a1e4b980f36a2852bed98c8d204127753e9850330482f1719f` | `2b5949962d8aa1d5d61dfb0be22249a5e57a0fcd5567b1c217cf00639dd92a4d` | `be4750f63b4e455371a01c47689e98db90201f50ff19262750cea26236ccc54d` |
| affine K8V2-g64 | `ccd7eb67813fea47693711c9e892b27dbf0988509af680092f3302c24bc27df8` | `0d634f60682c4ed2c695efe71854ee203236a3a3ddb247acc01085f78b158086` | `54932cab249a3eeab71634ff3b29b11015aa8662531583e9dd7b97467b5682e0` | `2773f778dc1c3cc944f1134db3a557eab2c50474e689e742955337912f6cc8ae` |
| affine K8V2-g128 | `4f270fa4d8a720b8bb00e15227b08c411ccf351d058b84d2569cdad0f0cc3737` | `d7711b970ea80ea5097ebbecc5f20bb789648f0429a1e4dbe2a5b1567bd73024` | `372692a44190bd331adb154b99eb331bafc7a14ddbe011de891a43f1b4adf4a8` | `59d8f58a774ad6028874a945343cc759ad4608903bc38d62b6ef36041195401e` |
| frozen KVTuner | `46d63a5952b0b65246885eb82b3faf40ed4e20d9c11aad244a5da91d07b9c18b` | `431f6c07bff35bdc685d9a4c471805cb17ec1e47905e8762ba9230ca8b485be4` | `cfb473c8c2c117eeaf7ee8228db8f99018c0d77027ebf46ef499bb18cd161a5f` | `ee39f44378f28391aa82573828aed66e7fe28b6748469ce50a0d7c9e7454dc45` |
| KVarN i8 | `0b631fc5b09dcd507496211c4503b25c9e0322477e51348324d0dd5c6bf8b8a7` | `50600932347faeb314ad784a73716269c822073cc219e3cf188b63873ef731b3` | `557a753e19e805baef99da3a7078f88e473681f2c2030d3ed75c520bb9a98b06` | `21eb19d8db86726a64f6b1c43278c18a696200d52296a28e85a0892ab25ed975` |
| KVarN i16 | `0e506e9780e3f298408c752a8d944d1d2055f2462ad4298812d56e385e1710f1` | `e28b6df1e9dfb3f394ead5928db10a694a720d050faa2543926a052f98d80937` | `4b8722c16835184bc82583d0950c5f2e808236834f28e252fd84774d41f6228d` | `2a5e041c25bb88ea393d6b134a1ad52921b0a6c804eea7aa592a0a4094a4d8d7` |
