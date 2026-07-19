# Fused compressed-domain KV attention implementation plan

**Status:** active — Phases 0-3 verified; exact-source requalification, KVarN direct attention,
and loaded-model frontiers remain gated

**Technique class:** `LOSSY_FRONTIER`

**Task seed:** [`2026-07-12-fused-compressed-kv-attention.md`](../../task-inbox/2026-07-12-fused-compressed-kv-attention.md)

**Input verdict:** [`2026-07-18-kvarn-kv-frontier.md`](../verdicts/2026-07-18-kvarn-kv-frontier.md)

**Pinned stack:** `mlx-swift` 0.31.6 (`0bb916c`) and `mlx-swift-lm` `702e5a0`

## Outcome and operator story

A long-context Apple-Silicon operator should be able to select a measured affine, frozen KVTuner,
or KVarN KV tier that reduces unified-memory pressure **and** attention traffic, see its directly
measured speed/capacity/quality point, and retain control over more aggressive useful loss above the
non-garbage floor. fast-mlx must never describe a materialize-then-attend path as compressed-domain
attention or turn structural cache corruption into a quality tradeoff.

This is one engine gate, not the fast-mlx product boundary. The implementation should be reusable by
models supported by the pinned MLX stack, while qualification starts with popular representative
families and keeps every support claim scoped to the model/architecture matrix below. The product
remains model-generic; Qwen/Llama evidence must not become hardcoded product behavior.

## Exploration result and initial architecture decision

The pinned stack has two materially different cache/attention boundaries:

1. `KVCache.update(keys:values:)` returns ordinary K/V arrays. `attentionWithCacheUpdate` then calls
   `MLXFast.scaledDotProductAttention`. Every current fast-mlx fp16, affine, TurboQuant, and KVarN
   cache reaches this boundary with materialized arrays.
2. `QuantizedKVCacheProtocol.updateQuantized` is the only existing packed-state escape hatch.
   `attentionWithCacheUpdate` routes it through `quantizedScaledDotProductAttention`, which composes
   quantized matrix multiplies without reconstructing the complete K/V cache first.

The existing quantized protocol has one bit width, one group size, and one quantization mode for
both K and V. It can represent a symmetric affine control, but not the selected K4V2 tier, the
per-layer KVTuner K/V schedule, or KVarN's transform. Core `MLXFast.scaledDotProductAttention` has no
packed-cache argument. `MLXFast.metalKernel` can JIT a custom Metal kernel, but performance and copy
costs are shape/layout dependent. There is no repository-owned `.metal` implementation today.

Therefore the first implementation is a **stock-primitive profiling gate**, not an immediate custom
kernel:

- compare stock fp16 SDPA, the pinned Swift-LM quantized-attention helper, and the current
  materialize-then-attend affine/KVarN paths at authenticated, config-constrained synthetic tensor
  geometry;
- authenticate numeric output, latency, storage, workspace, and provenance at the context lengths
  allowed by each selected checkpoint config;
- decide from measured evidence whether the first integrated path should extend the existing
  quantized-matmul route or justify a custom Metal kernel;
- do not modify ephemeral `.build/checkouts`. Any model-router hook needed after Phase 0 must be a
  portable pinned dependency revision/fork or an accepted upstream change, with license and diff
  review. A machine-local checkout patch cannot become promotion evidence.

Phase 0 geometry probes stream checkpoint bytes for content authentication but do not instantiate
or execute those weights as MLX model tensors. They can prove packed-attention structural behavior
for a model-compatible geometry; they
cannot prove model-specific runtime, dial, quality, or end-to-end performance claims.

The first MLX-coupled causal-prefill fixture also found that the pinned Swift-LM helper's symbolic
`.causal` branch fills masked scores with `Float.leastNormalMagnitude`, a positive value, rather
than a negative-infinity sentinel. Decode-shaped `Tq=1` is unaffected, but prefill-shaped queries
failed the same-packed-bytes oracle. Phase 0 therefore canonicalizes logical causal requests to an
explicit lower-right additive mask for both the quantized helper and its oracle. Direct use of the
pinned symbolic branch is not promotion-capable; the portable router design must retain this
regression fixture or carry an authenticated upstream fix.

## Model and architecture matrix

| Family / architecture | Phase | Intended proof | Initial disposition |
| --- | --- | --- | --- |
| Qwen3 dense GQA, Qwen3-32B-4bit | first | Existing KVarN source model; Q64/KV8/D128, 8K smoke then 32K, scalar and hostile batch transition | required evidence target; model-specific; max context is 40,960, so 128K must be an authenticated refusal |
| Llama dense GQA, Llama-3.3-70B-Instruct-4bit | second | Different model family and tokenizer/config identity using the shared attention router; Q64/KV8/D128; 8K smoke, 32K, and near-128K with prompt+output <= 131,072 | required before any dense-GQA same-geometry cross-family claim; checkpoint must be source-locked before download/run |
| Gemma local/global or rotating-cache family | later boundary | Sliding/local mask and cache-lifecycle semantics | fail closed until rotating/window cache proof exists |
| Dense/MoE models whose attention uses the shared router | later matrix | Attention compatibility separate from expert/runtime behavior | no inherited claim from Qwen/Llama |
| Hybrid/recurrent/MLA/VLM/diffusion or custom attention | out of initial gate | Architecture-specific state, mask, sink, or cache contract | unsupported and rejected explicitly |

Two families do not imply every MLX model is qualified. The selected Qwen3 and Llama checkpoints
share Q64/KV8/D128 attention geometry, so a passing pair can support only a same-geometry
dense-GQA claim for the exact authenticated identities. It does not prove cross-geometry
generality. Broad product/default claims need another popular model with materially different
attention geometry. KVTuner schedules remain Qwen-only unless independently calibrated and
authenticated for Llama or any later family.

## Acceptance criteria and proof

1. **Authenticated probe plan.** A pure value contract names the operation (`fp16-sdpa`,
   `swiftlm-quantized-attention`, `split-affine-quantized-mm`, or
   `materialize-then-sdpa`), KV context length, query-token
   length, prefill chunk shape, requested output-token count, explicit stop-token set, B/Q/KV
   heads, head dimension, K/V bits and groups, mask, dtype, warmups/runs, source SHA, workload
   nonce, and output path. Decode-shaped (`Tq=1`) and prefill-shaped cells have distinct workload
   identities and can never enter one comparison aggregate.
   Only predeclared qualification cells allowed by the authenticated checkpoint config are strict
   evidence-capable. The qualification flag is intentionally not named or treated as promotion:
   every artifact declares `checkpoint-authenticated-synthetic-geometry`, and no probe run loads
   model weights. The synthetic probe uses Qwen 8K/32K plus an authenticated 128K refusal and
   Llama at the frozen near-128K pair of 130,944 cached tokens plus 128 requested output tokens
   (with an optional low-cost identity canary); repeating the same Q64/KV8/D128
   synthetic matrix under both model IDs is not independent family evidence. Loaded-model proof
   separately uses Qwen 8K/32K and Llama 8K/32K/near-128K with prompt+output at or below 131,072.
   Proof: local failing-first `HarnessCoreTests`
   for missing, unknown, overflowed, aliased, symlinked, or inconsistent inputs.
2. **Numeric control.** The fp16 probe control agrees with stock SDPA at the pinned MLX
   float16 qualification envelope, `rtol=3e-4`, `atol=3e-4`. Evidence reports raw maximum
   absolute and relative errors plus the maximum mixed-tolerance ratio
   `abs(error) / (atol + rtol * abs(reference))`; the ratio must be at most `1`.
   Every packed affine, KVTuner, and KVarN fixture is compared to dequantize-then-attend from the
   exact same authenticated packed bytes and must pass `rtol=2e-3`, `atol=2e-3` plus identical
   top-1 output; max/mean error is also reported. These are structural equivalence thresholds,
   frozen before performance work, not model-quality allowances. A row outside either threshold,
   with independently requantized control data, or with non-finite input/output is a hard failure
   and cannot be relabeled as acceptable lossy quality. Proof: Xcode MLX tests and immutable probe
   JSONL.
3. **Honest engagement and memory.** Every row records actual persistent packed arrays, scales,
   biases, control bytes, alignment, materialization bytes, and peak temporary workspace. It also
   records MLX active/cache/peak memory and process footprint. Workspace totals must be derived
   from and validate against the raw MLX peak receipt. Because resetting MLX peak memory does not
   inject the already-resident active baseline into the counter, the effective peak is
   `max(raw post-reset peak, pre-run active bytes)`. The authenticated cache policy names the
   actual run-wide preservation boundary. A nominal tier name is insufficient.
4. **Profile-before-kernel gate.** Phase 0 records warmed per-step latency distributions for stock
   fp16 SDPA, pinned quantized attention, and current materialize-then-attend paths at the
   checkpoint-config-allowed synthetic geometries. A runtime integration design is selected only
   after the data identifies the dominant cost. Kernel-only or synthetic-geometry evidence may
   authorize engineering work but cannot promote a dial tier.
5. **Portable model-router seam.** If integration proceeds, models using the shared attention
   helper can invoke one actor-confined packed-cache attention contract without importing
   fast-mlx or hardcoding model names. The dependency source and exact patch/revision are pinned.
   Unknown or unsupported model attention paths return a typed refusal before model execution.
6. **Selected formats consume packed state.** Affine K4V2-g64 and frozen KVTuner use their actual
   independent K/V widths; KVarN i8 uses its actual selected transform/layout. No path expands
   KV heads or reconstructs the full cache. KVTuner authenticates the frozen Qwen bundle and layer
   schedule before arrays are allocated, and remains unavailable for Llama unless a Llama-specific
   calibration is independently produced and authenticated.
7. **Cache lifecycle correctness.** Growth, rollback, reset, masks, GQA, head dimensions, and
   compilation preserve the existing contracts. The hostile batch case merges unequal rows,
   removes the longest row at the zero-padding boundary, appends again, and proves explicit
   physical end, mask width, survivor packed bytes/logits, and dense-control parity. Structural
   divergence is a correctness failure, never measured loss.
8. **Fail-closed combinations.** Unknown geometry, unsupported architecture, schedule/model hash
   mismatch, non-finite values, arithmetic overflow, partial evidence, or a missing workspace
   receipt aborts the run. Lossy KV plus PLD remains rejected. No silent fp16/materialized fallback
   occurs after a request starts.
9. **End-to-end Apple frontier.** Promotion requires identical model workloads within each
   checkpoint's context limit: Qwen 32K plus authenticated 128K refusal, and Llama 32K plus
   near-128K with prompt+output <= 131,072. Evidence includes direct prefill/decode,
   TTFT/TPOT, actual cache bytes/capacity, MLX active/cache/peak, process RSS, task floor, and
   teacher-forced KL/perplexity/tail-p95. At least three post-warmup runs are retained, not only
   aggregates. The evidence binds query/prefill/output/stop shape as well as KV context length.
10. **Speed gate.** A speed-tier candidate must beat both its same-storage materialize path and
    fp16 base decode at every promotion-capable model workload for that checkpoint by at least 5%
    in every retained post-warmup run, without a greater than 5% prefill regression. Cells run in a
    deterministic counterbalanced/interleaved
    order with identical explicit `Memory.cacheLimit`, `Memory.memoryLimit`, wired-memory setting,
    model residency,
    prompt/output/stop identity, and cache-reset policy. Every row records run position, monotonic
    timestamps, power source, Low Power Mode, thermal state, MLX cache before/after, and
    allocator/process-resident receipts; a thermal transition, power-source change, or Low Power
    Mode change invalidates the complete paired block rather than one inconvenient row. Smaller
    or lossier cells that do not meet this gate retain only their prior capacity disposition;
    this cycle does not manufacture a speed label.
11. **User-controlled loss.** Transparent/Balanced/Max-fit classification uses the already-locked
    teacher-forced and task hard floor. Useful aggressive points above the floor remain available
    with measured warnings; incoherent points remain impossible to select.
12. **Cross-family scope and closure.** Qwen3 is adjudicated first at 8K smoke/32K with an
    authenticated 128K refusal. A source-locked Llama-family checkpoint repeats the applicable
    8K/32K/near-128K gate before any same-geometry cross-family claim. Another popular model with
    materially different attention geometry is required before any broad/default product claim. The
    cycle ends with a dated PROMOTE/SHELVE verdict, compact evidence, a `docs/content/` piece,
    verification packet, focused review, secret scan, coherent commits, and `--no-ff` merge only
    after fresh proof.

## Happy, failure, and recovery paths

- **Happy path:** a deterministic Qwen3 geometry probe shows packed attention avoids full-cache
  materialization with acceptable numeric behavior at 8K smoke/32K and refuses 128K because the
  authenticated config max context is 40,960. The portable router seam is integrated behind an
  experimental flag, scalar decode passes, the hostile batch transition passes, and valid Qwen plus
  Llama end-to-end model evidence advances one or more measured dial cells.
- **Stock primitive is sufficient:** extend the existing quantized-attention route with the
  smallest portable contract that supports independent K/V geometry; do not write custom Metal
  merely for novelty.
- **Custom kernel justified:** only when the profile attributes the loss to the stock primitive or
  materialization and the selected packed layouts cannot be represented faithfully. First prove one
  Qwen geometry before generalizing templates.
- **No bottleneck or end-to-end loss:** preserve the artifacts and SHELVE the speed path with a
  dated verdict. Existing capacity tiers remain valid; there is no obligation to promote a kernel.
- **Unsupported/malformed request:** fail before cache mutation with a typed error. The operator may
  explicitly rerun the already-qualified materialize or fp16 tier as a new request.
- **Mid-run fault:** abort and discard the request/cache. Never switch representation or attention
  implementation inside an active sequence.
- **Long-run interruption:** preserve immutable completed rows, status, progress, and watchdog
  artifacts. Resume only through exact identity/idempotence checks into a fresh output boundary.

## Proof methods and build boundary

- Pure plan/schema/reducer/evidence tests: local `swift test --package-path spike --filter
  HarnessCoreTests`.
- Anything importing MLX, the model, caches, or a custom kernel: sync with
  `spike/scripts/sync_llmbench.sh`, then Xcode tests/build on the bench Mac with
  `-skipPackagePluginValidation`. Never use `swift test` for those targets.
- Phase 0 probe outputs: fresh-output directory, held runner lock, atomic progress heartbeat,
  absent watchdog on success, clean source/package/checkpoint-config identity, exact row count and
  hashes. Probe rows are synthetic geometry evidence; end-to-end model evidence requires separate
  model-weight runs.
- Long contexts: if `iogpu.wired_limit_mb` is raised, record and set the explicit
  `Memory.cacheLimit` recommended by the capacity model.
- Quality: teacher-forced/context-locked primary metrics; free-running task/coherence checks are
  secondary and must retain the non-garbage hard floor.

## Work order

### Phase 0 — profile and evidence contract

- [x] TDD a pure `CompressedAttentionProbePlan` and immutable evidence schema, including
  operation/context/query/prefill/output/stop/geometry/layout/provenance identities, paired run
  order and environment receipts, and exact raw-run retention.
- [x] Complete verification of the MLX-coupled probe CLI comparing stock fp16 SDPA, pinned
  Swift-LM quantized attention, split K/V affine quantized matmuls, and
  materialize-then-attend from the same packed bytes.
- [x] Run small deterministic correctness fixtures, then interleaved Qwen 8K smoke/32K plus
  authenticated 128K refusal and one Llama near-128K synthetic geometry (optionally an 8K identity
  canary) on the bench Mac. Do not repeat the full same-geometry matrix under both checkpoint IDs.
  Record materialization, workspace high-water bytes, fixed cache/wired-memory settings, MLX cache
  state, and thermal state.
- [x] Write a short architecture decision from the measured bottleneck. Stop here if a direct
  packed path lacks a credible speed envelope.

#### Phase 0 measured architecture decision — 2026-07-18

Clean `07219679280abd2f7cefbeef86b71bbec018a1c2` produced 12 probe-evidence-schema-v2
synthetic artifacts and 72 retained measured rows plus the authenticated Qwen refusal. The compact
index uses its own schema v1 and records the five/five/two source artifact counts explicitly. The
evidence is [`fused-compressed-kv-phase0-evidence-2026-07-18.jsonl`](../verdicts/fused-compressed-kv-phase0-evidence-2026-07-18.jsonl);
its SHA-256 is `bb8387e2f4b6d1ac4862f09816a5517c2a34b4f7e7a9383f300d3002c13a3cf5` and
the 77-file bench manifest has SHA-256
`6f999443363267a8a6429140766027fa0f1f14df4b0f400a257171d2230d2d89`.

- Qwen 8K decode was a correctness/memory smoke, not a timing conclusion: sub-millisecond paired
  ratios crossed both sides of `1.0`, including split K4V2 at `0.9224...1.1439x` fp16.
- At Qwen 32K decode, split K4V2 used 29,360,128 persistent bytes plus 4,751,366 peak temporary
  bytes and measured `0.5360...0.5758x` its paired fp16 attention time in all three runs. The
  same-storage materialize control used 138,559,488 peak temporary bytes and measured
  `0.8999...1.3628x` fp16. Symmetric stock K4V4 also showed a credible packed route at
  `0.5276...0.6221x` fp16.
- At the Llama-compatible near-128K geometry, split K4V2 used 117,325,824 persistent bytes plus
  17,317,894 peak temporary bytes and measured `0.3866...0.4557x` fp16. Its same-storage
  materialize control required 553,664,512 peak temporary bytes and measured
  `0.6869...0.8310x` fp16. Qwen refused the same 130,944+128 window before MLX allocation because
  its authenticated maximum is 40,960.

**Decision:** proceed to the portable router seam by extending the existing quantized-matmul
route to independent K/V geometry. Do not start with a custom Metal kernel: the pinned primitives
already show a credible long-context decode envelope and remove full-cache materialization. The
split operation is not fused SDPA and retains score/weight workspace; long-context prefill,
loaded-model TTFT/TPOT, cache lifecycle, hostile batch compaction, KVTuner schedule routing, KVarN,
quality, and end-to-end speed remain unproven gates. Because Qwen and Llama share Q64/KV8/D128,
the Llama geometry row is not independent cross-geometry or model-runtime evidence.

### Phase 1 — portable attention-router seam

- [x] Define the narrow packed-cache attention contract required by the selected formats and
  review it against every pinned model call shape that uses the shared helper.
- [x] Choose a portable dependency strategy. Do not patch `.build/checkouts`; pin every source
  revision and keep the dependency diff upstream-reviewable.
- [x] TDD unsupported path, independent K/V geometry, mask, GQA, sink/window, and non-finite
  failures before connecting any model.

Clean implementation commit: `faa385f` (`Add portable packed-affine attention router`). The
portable source is vendored and revision-pinned; the shared helper routes by cache capability, not
model name. Unsupported/windowed/sink/non-finite paths fail closed.

### Phase 2 — actor-confined scalar runtime

- [x] Integrate affine K4V2-g64 and frozen KVTuner first, because their packed affine data maps
  most directly to quantized matrix multiplies.
- [ ] Add KVarN i8 only after its query/key transform and value/output reconstruction are proven
  from the same packed bytes without materialization.
- [x] Preserve exact engagement, persistent bytes, workspace, reset, growth, and rollback receipts
  for the implemented affine/KVTuner formats. KVarN receipts remain part of its open item above.

Clean implementation commit: `e2d719e` (`Add authenticated compressed KV attention runtime`).
This is an implementation/correctness milestone, not a loaded-model speed result.

### Phase 3 — continuous-batch poison case

- [x] Implement merge/extract/filter only for a proven dense architecture class.
- [x] Track explicit physical written end independently of surviving logical/padded offsets.
- [x] Pass unequal-row merge -> longest-row removal -> append -> survivor byte/logit/mask parity.

Clean implementation commit: `5e6abb6ebf13ea8641b26638278680e99884adea` (`Add
provenance-bound compressed continuous batching`). Clean verification passes 457 XCTest + 17 Swift
Testing HarnessCore tests, 65 FastMLXHarness tests, 127 SpikeCore tests, and the Release build.
Focused review reports no remaining High/Medium findings; the commit scan reports no secrets or
banned concurrency escape hatches. The same change migrates KVTuner from manifest-only checkpoint
identity to exact checkpoint-content identity through calibration, sensitivity, candidates,
search, schedule, runtime, task, and KL evidence.

### Phase 3.5 — exact-source KVTuner requalification

- [x] Produce a fresh schema-v2 calibration manifest from the clean Phase 3 SHA. Artifact SHA-256:
  `e8b069cafb697a332325def638effdaf8f56b9bc62d2139b2c7dc2aba1719a5f`; checkpoint-content
  SHA-256: `636f358d4f51c9394400fa46ef684b918e45c14d369d95df0399c80abc8a09d9`.
- [x] Capture the clean-SHA g128 sensitivity artifact. Artifact SHA-256:
  `9426976a9215ce5276ac80ea165de3b084b239652ce138e670163bcbdf41d7fc`; 64 layers and 3,840
  authenticated samples under artifact ID `fused-compressed-kv-qwen3-32b-v1-5e6abb6`.
- [ ] Evaluate the exact canonical 64-candidate set at target pair-bits 390 through the resumable,
  exact-byte/idempotent runner path. This is the current long-running stage.
- [ ] Authenticate search and schedule, then freeze a new qualification bundle before any KVTuner
  runtime, task, KL, or end-to-end row.

The historical KVarN-cycle bundle remains immutable evidence for its dated verdict. It is not
rewritten or silently upgraded and cannot authorize the new runtime path.

### Phase 4 — end-to-end Qwen frontier

- [ ] Run 8K as the bounded smoke, then 32K only from the clean verified SHA; record the
  authenticated 128K refusal because Qwen3-32B max context is 40,960.
- [ ] Compare fp16, same-storage materialize, packed affine K4V2-g64, frozen KVTuner, and KVarN i8.
- [ ] Preserve negative/dominated and hard-floor-failed rows rather than filtering the matrix.

### Phase 5 — second family and adjudication

- [ ] Source-lock the selected Llama-3.3-70B-family checkpoint and repeat the applicable 8K
  smoke/32K/near-128K gate with prompt+output <= 131,072.
- [ ] Keep Qwen-specific KVTuner unavailable unless separately calibrated and authenticated for
  Llama.
- [ ] Add a third popular, materially different attention geometry before broad/default product
  support claims.
- [ ] Quantify Transparent/Balanced/Max-fit speed/capacity/loss, write verdict/content, verify,
  review, scan, commit, and merge only if every claimed gate has fresh proof.

## Security, maintenance, and rollback

- Custom Metal source is executable GPU code. Bound every shape, byte calculation, grid, thread
  group, and output allocation before dispatch; malformed evidence or model metadata cannot select
  arbitrary source/templates.
- No MLX state crosses its owning actor and no `@unchecked Sendable` or
  `nonisolated(unsafe)` escape hatch is permitted.
- The feature remains behind an explicit experimental runtime selection until the full gate passes.
  Rollback is removal of that selection at request admission; persisted packed-cache formats are not
  silently opened by a runtime that does not recognize their version.
- A dependency fork or upstream patch increases maintenance surface. Its source, license, pinned
  revision, diff, and upstream status are part of release evidence.
