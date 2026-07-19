# Fused compressed-domain KV attention implementation plan

**Status:** active — Phase 0 profiling/evidence work authorized; runtime integration remains gated

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
families and keeps every support claim scoped to the model/architecture matrix below.

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
  materialize-then-attend affine/KVarN paths at identical tensor geometry;
- authenticate numeric output, latency, storage, workspace, and provenance at 8K/32K/128K;
- decide from measured evidence whether the first integrated path should extend the existing
  quantized-matmul route or justify a custom Metal kernel;
- do not modify ephemeral `.build/checkouts`. Any model-router hook needed after Phase 0 must be a
  portable pinned dependency revision/fork or an accepted upstream change, with license and diff
  review. A machine-local checkout patch cannot become promotion evidence.

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
| Qwen3 dense GQA, Qwen3-32B-4bit | first | Existing KVarN source model; head-dim/GQA, 8K then 32K/128K, scalar and hostile batch transition | required evidence target; model-specific |
| Llama dense GQA, 128K-capable popular checkpoint | second | Different model family and tokenizer/config identity using the shared attention router | required before any dense-GQA cross-family/default claim; checkpoint must be source-locked before download/run |
| Gemma local/global or rotating-cache family | later boundary | Sliding/local mask and cache-lifecycle semantics | fail closed until rotating/window cache proof exists |
| Dense/MoE models whose attention uses the shared router | later matrix | Attention compatibility separate from expert/runtime behavior | no inherited claim from Qwen/Llama |
| Hybrid/recurrent/MLA/VLM/diffusion or custom attention | out of initial gate | Architecture-specific state, mask, sink, or cache contract | unsupported and rejected explicitly |

Two families do not imply every MLX model is qualified. If Qwen3 and Llama pass, the claim remains
limited to the tested dense full-attention GQA geometry and exact model identities. KVTuner schedules
remain model-specific and must fail authentication on the second family.

## Acceptance criteria and proof

1. **Authenticated probe plan.** A pure value contract names the operation (`fp16-sdpa`,
   `swiftlm-quantized-attention`, or `materialize-then-sdpa`), KV context length, query-token
   length, prefill chunk shape, requested output-token count, explicit stop-token set, B/Q/KV
   heads, head dimension, K/V bits and groups, mask, dtype, warmups/runs, source SHA, workload
   nonce, and output path. Decode-shaped (`Tq=1`) and prefill-shaped cells have distinct workload
   identities and can never enter one comparison aggregate.
   Only the predeclared 8K/32K/128K contexts are promotion-capable. Proof: local failing-first
   `HarnessCoreTests` for missing, unknown, overflowed, aliased, symlinked, or inconsistent inputs.
2. **Numeric control.** The fp16 probe control agrees with stock SDPA at `rtol=1e-4`, `atol=1e-5`.
   Every packed affine, KVTuner, and KVarN fixture is compared to dequantize-then-attend from the
   exact same authenticated packed bytes and must pass `rtol=2e-3`, `atol=2e-3` plus identical
   top-1 output; max/mean error is also reported. These are structural equivalence thresholds,
   frozen before performance work, not model-quality allowances. A row outside either threshold,
   with independently requantized control data, or with non-finite input/output is a hard failure
   and cannot be relabeled as acceptable lossy quality. Proof: Xcode MLX tests and immutable probe
   JSONL.
3. **Honest engagement and memory.** Every row records actual persistent packed arrays, scales,
   biases, control bytes, alignment, materialization bytes, and peak temporary workspace. It also
   records MLX active/cache/peak memory and process footprint. A nominal tier name is insufficient.
4. **Profile-before-kernel gate.** Phase 0 records warmed per-step latency distributions for stock
   fp16 SDPA, pinned quantized attention, and current materialize-then-attend paths at 8K/32K/128K.
   A runtime integration design is selected only after the data identifies the dominant cost.
   Kernel-only evidence may authorize engineering work but cannot promote a dial tier.
5. **Portable model-router seam.** If integration proceeds, models using the shared attention
   helper can invoke one actor-confined packed-cache attention contract without importing
   fast-mlx or hardcoding model names. The dependency source and exact patch/revision are pinned.
   Unknown or unsupported model attention paths return a typed refusal before model execution.
6. **Selected formats consume packed state.** Affine K4V2-g64 and frozen KVTuner use their actual
   independent K/V widths; KVarN i8 uses its actual selected transform/layout. No path expands
   KV heads or reconstructs the full cache. KVTuner authenticates the frozen bundle and layer
   schedule before arrays are allocated.
7. **Cache lifecycle correctness.** Growth, rollback, reset, masks, GQA, head dimensions, and
   compilation preserve the existing contracts. The hostile batch case merges unequal rows,
   removes the longest row at the zero-padding boundary, appends again, and proves explicit
   physical end, mask width, survivor packed bytes/logits, and dense-control parity. Structural
   divergence is a correctness failure, never measured loss.
8. **Fail-closed combinations.** Unknown geometry, unsupported architecture, schedule/model hash
   mismatch, non-finite values, arithmetic overflow, partial evidence, or a missing workspace
   receipt aborts the run. Lossy KV plus PLD remains rejected. No silent fp16/materialized fallback
   occurs after a request starts.
9. **End-to-end Apple frontier.** Promotion requires identical 32K and 128K model workloads for
   fp16, materialize, and packed-attention cells. Evidence includes direct prefill/decode,
   TTFT/TPOT, actual cache bytes/capacity, MLX active/cache/peak, process RSS, task floor, and
   teacher-forced KL/perplexity/tail-p95. At least three post-warmup runs are retained, not only
   aggregates. The evidence binds query/prefill/output/stop shape as well as KV context length.
10. **Speed gate.** A speed-tier candidate must beat both its same-storage materialize path and
    fp16 base decode at 32K and 128K by at least 5% in every retained post-warmup run, without a
    greater than 5% prefill regression. Cells run in a deterministic counterbalanced/interleaved
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
12. **Cross-family scope and closure.** Qwen3 is adjudicated first. A source-locked Llama-family
    checkpoint repeats the applicable gate before any cross-family/default claim. The cycle ends
    with a dated PROMOTE/SHELVE verdict, compact evidence, a `docs/content/` piece, verification
    packet, focused review, secret scan, coherent commits, and `--no-ff` merge only after fresh proof.

## Happy, failure, and recovery paths

- **Happy path:** a deterministic Qwen3 geometry probe shows packed attention avoids full-cache
  materialization with acceptable numeric behavior and a meaningful long-context latency win. The
  portable router seam is integrated behind an experimental flag, scalar decode passes, the hostile
  batch transition passes, and full 32K/128K model evidence advances one or more measured dial cells.
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
  absent watchdog on success, clean source/package/model identity, exact row count and hashes.
- Long contexts: if `iogpu.wired_limit_mb` is raised, record and set the explicit
  `Memory.cacheLimit` recommended by the capacity model.
- Quality: teacher-forced/context-locked primary metrics; free-running task/coherence checks are
  secondary and must retain the non-garbage hard floor.

## Work order

### Phase 0 — profile and evidence contract

- [x] TDD a pure `CompressedAttentionProbePlan` and immutable evidence schema, including
  operation/context/query/prefill/output/stop/geometry/layout/provenance identities, paired run
  order and environment receipts, and exact raw-run retention.
- [ ] Add an MLX-coupled probe CLI that compares stock fp16 SDPA, pinned Swift-LM quantized
  attention, and materialize-then-attend from the same packed bytes.
- [ ] Run small deterministic correctness fixtures, then interleaved 8K/32K/128K Qwen and
  Llama-compatible decode and prefill geometries on the bench Mac. Record materialization,
  workspace high-water bytes, fixed cache/wired-memory settings, MLX cache state, and thermal state.
- [ ] Write a short architecture decision from the measured bottleneck. Stop here if a direct
  packed path lacks a credible speed envelope.

### Phase 1 — portable attention-router seam

- [ ] Define the narrow packed-cache attention contract required by the selected formats and
  review it against every pinned model call shape that uses the shared helper.
- [ ] Choose a portable dependency strategy. Do not patch `.build/checkouts`; pin every source
  revision and keep the dependency diff upstream-reviewable.
- [ ] TDD unsupported path, independent K/V geometry, mask, GQA, sink/window, and non-finite
  failures before connecting any model.

### Phase 2 — actor-confined scalar runtime

- [ ] Integrate affine K4V2-g64 and frozen KVTuner first, because their packed affine data maps
  most directly to quantized matrix multiplies.
- [ ] Add KVarN i8 only after its query/key transform and value/output reconstruction are proven
  from the same packed bytes without materialization.
- [ ] Preserve exact engagement, persistent bytes, workspace, reset, growth, and rollback receipts.

### Phase 3 — continuous-batch poison case

- [ ] Implement merge/extract/filter only for a proven dense architecture class.
- [ ] Track explicit physical written end independently of surviving logical/padded offsets.
- [ ] Pass unequal-row merge -> longest-row removal -> append -> survivor byte/logit/mask parity.

### Phase 4 — end-to-end Qwen frontier

- [ ] Run 8K as the bounded smoke, then 32K and 128K only from the clean verified SHA.
- [ ] Compare fp16, same-storage materialize, packed affine K4V2-g64, frozen KVTuner, and KVarN i8.
- [ ] Preserve negative/dominated and hard-floor-failed rows rather than filtering the matrix.

### Phase 5 — second family and adjudication

- [ ] Source-lock the selected Llama-family checkpoint and repeat the applicable 32K/128K gate.
- [ ] Keep model-specific KVTuner unavailable unless separately calibrated and authenticated.
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
