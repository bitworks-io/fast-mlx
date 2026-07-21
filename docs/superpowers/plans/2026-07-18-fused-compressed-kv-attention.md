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
- [x] Add KVarN i8 only after its query/key transform and value/output reconstruction are proven
  from the same packed bytes without materialization.
- [x] Preserve exact engagement, persistent bytes, workspace, reset, growth, and rollback receipts
  for the implemented affine/KVTuner/KVarN formats.

Clean implementation commit: `e2d719e` (`Add authenticated compressed KV attention runtime`).
This is an implementation/correctness milestone, not a loaded-model speed result.

Clean pure-contract commit `3bb0a2a43fba9690dfaab86cb302a746c9556ef9` added the distinct
`split-kvarn-quantized-mm` request/receipt and fail-closed frontier/task-promotion rules for the
exact K4V2/G128/i8 cell. The subsequent direct runtime increment implements packed K/V qMM,
normalized-Hadamard V reconstruction, compiled tile lifecycle, bounded prefill, and authenticated
CLI/task evidence without materializing the full cache.

The compile-lifecycle review further freezes these direct-KVarN implementation constraints before
production work:

- the packed V product must apply KVarN's final normalized Hadamard reconstruction after the
  quantized matrix multiply and channel scale; a finite rotated-channel result is incorrect;
- completed packed slots and the live fp16 tail need distinct source-liveness masks, driven by an
  in-graph packed-tile count included in `innerState()`, so unused slots cannot enter softmax or
  duplicate the live tail;
- checked geometry, mask, dtype, and finiteness preflight must finish before any cache mutation;
- tile finalization at compiled replay boundaries must encode the boundary token before that
  step's attention reads the completed packed slot, clear the live tail for the next replay, and
  update stable MLX array identities in place; host `offset` is not authoritative compiled state;
- direct KVarN prefill uses the same bounded chunking discipline as the affine split route to cap
  the explicit score tensor.

The bench red/green suite covers negative-score inactive-slot leakage, direct packed-byte algebra,
pre-mutation failure, compiled tile-boundary identity/order, reset/reuse, KVarN prefill bounds,
full packed-capacity boundaries, hostile masks, scoring-cache capacity, and reset-time rejection.
Dirty-tree implementation proof on 2026-07-20 passed 72 FastMLXHarness tests, 146 SpikeCore tests,
the complete HarnessCore suite, and the Release build; focused review found no remaining issue.
This is implementation proof only. Promotion remains blocked on repetition from the resulting clean
SHA and the loaded-model frontier in Phases 4 and 5.

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
- [x] Evaluate the exact canonical 64-candidate set at target pair-bits 390 through the resumable,
  exact-byte/idempotent runner path.
- [x] Authenticate search and schedule, then freeze a new qualification bundle before any KVTuner
  runtime, task, KL, or end-to-end row.

The completed clean-`5e6abb6` cohort contains all 64 exact candidates and 12,800 authenticated rows.
The frozen schedule SHA-256 is
`d76d2534939677711cf27123eea0cf15ad3512bd5fd04965abe85e36232f9260`; the qualification bundle
SHA-256 is `8217bc37cd3d100c493fb6f76b15d7abe5c168baa2bc2fb6561ac9f12c9dc125`.

#### Candidate-run recovery — 2026-07-19

The original clean-`5e6abb6` candidate process published and independently authenticated ordinals
0...4, then macOS killed the harness while ordinal 5 was in flight. The kernel recorded
`no paging space` and selected `fastmlx-harness` as the largest compressed process at 153,803 MB.
This is an infrastructure/lifetime failure, not a candidate-quality or model verdict. The hash-bound
failure record has SHA-256
`48ce0f5f76748e08711dc571084f2eec1aaf2f6884a038ab42f7bcbb4f1fe655`; the captured kernel excerpt
has SHA-256 `0d12fa549850e3ab2d30df8426a4d088ef8e7e80cfcc009e9af848f83e888abe`.

The lifetime audit found that the exhaustive runner loads one MLX model/driver for the complete
pending set. It creates a new compiled decoder per candidate and resets that decoder per prompt,
but neither reset clears MLX's process-level allocations. The same process therefore accumulated
compressed memory across candidates despite prompt/cache resets.

Recovery uses the existing exact-byte/idempotent single-ordinal path with one fresh harness process
per remaining ordinal. It preserves the same clean source SHA, Release binary SHA
`20aff66121fa7d7d82e3fa664f0ae3e9be514e337201114ca5775c8c79f862cd`, manifest SHA above, and
sensitivity SHA above; it does not rebuild, resync, or mix execution environments mid-cohort. The
held-lock wrapper has SHA-256
`427b7c3a0378ec6e878348be5fde15956f7891037b9ffeaaf2981c3bfc134765`. A first launcher attempt
failed closed before starting a harness because the crashed wrapper had removed the empty lock
artifact; that attempt is preserved at SHA-256
`81deb8fe369e4d575cf4b9663cc072a946c3a3bf02c7505e6ba42ddf51703efc`. Attempt 2 recreated the lock
artifact, acquired it, and completed the remaining ordinals through one fresh harness process per
candidate. The final exact-set/search/bundle validation passed for all 64 candidates and froze the
schedule and qualification-bundle hashes above. Subsequent loaded qualification keeps the
process-lifetime boundary explicit with one fresh harness process per matrix position; decoder or
cache reset is never treated as proof that process-level MLX allocations were released.

The historical KVarN-cycle bundle remains immutable evidence for its dated verdict. It is not
rewritten or silently upgraded and cannot authorize the new runtime path.

### Loaded-Qwen thermal admission recovery — 2026-07-21

Loaded Qwen exposed a benchmark-cohort problem rather than a kernel correctness failure. The first
7-cell matrix, v7, stopped after three authenticated rows when its block crossed from nominal to
fair at direct KVarN. The v8 materialize-only and v9 direct-only workload preconditions each
completed three authenticated rows but remained stable nominal, so their fair-only launchers
correctly refused to admit a matrix. These terminal boundaries are preserved unchanged and are not
promotion evidence. Workload-only preheating is closed; per-run or cross-row thermal equality was
not weakened.

Clean `b1289b783c1e156355a24c5db5f0e9b150a1cb3b` implemented the first narrower causal contract:

- qualification manifests require post-warmup target `nominal`, a bounded timeout, and poll
  interval; v10 froze 600 seconds and 1,000 ms;
- the dropped warmup now has an authenticated before/after receipt and may move only between safe
  nominal/fair states on AC power with Low Power Mode off;
- after warmup, the harness waits for an exact nominal admission snapshot before retained work;
  serious, critical, unknown, power drift, malformed timestamps, timeout, or retained drift fail
  closed;
- evidence schema v3 binds policy, warmup, admission, retained timestamps, and the manifest
  timeout, while historical schema-v2 rows remain readable and cannot claim the new contract;
- the loaded runner passes the frozen policy to every child and reauthenticates it in row, receipt,
  progress, block, and completion artifacts.

Fresh proof passes 506 HarnessCore XCTest plus 17 Swift Testing tests off-box, 93
FastMLXHarnessTests and 155 SpikeCoreTests through Xcode on the bench, the Xcode Release build,
focused re-review, shellcheck, diff/banned-pattern checks, and gitleaks. The clean Xcode binary
SHA-256 is `19f66d67d689fc3e6dc8e8a158b97224a4f76fe890cefb63cedc08c7b1450ec4`; runner SHA-256 is
`ff429bdc4fefd380c1dbcbbef936701879b91e43abb90af7966dc6c1b005d4b3`.

V10 is the fourth immutable failed thermal boundary. It used the unchanged 7-cell and 7-block
cyclic Latin square under manifest SHA-256
`353820692294f408ddae91e7f3b7d1522604ea77e40646c6a45d812af44434e1` and stopped after 5/49
authenticated rows at block 0, position 5, `affine-k4v2-g64-materialize`. The admitted snapshot
was nominal, but the thermal state changed during retained measurement. A point-in-time nominal
snapshot was therefore insufficient to establish a stable retained cohort. Preserve v10 unchanged; its terminal
status SHA-256 is `03cbb6c52e522ce55e217e3f9d66dbd66d21ef6bf85024016a3b7f3080c7944f` and its rejected partial
receipt-set SHA-256 is `2c8c9b1e2b7dc24f70bdb9935f9bcbdb9ad73efd06a249809dbb5906fa6f2889`.

Clean `d4102e6a3029b161d99ee27aceabbad8d5696fb5` implements the recovery contract:

- fresh runner manifests use schema v3 and bind a positive continuous stability duration in
  addition to target, timeout, and poll; the next smoke freezes 60 seconds;
- evidence schema v4 authenticates at least two strictly monotonic nominal/AC/non-low-power
  observations, resets the dwell on safe non-target state, requires the last observation to equal
  admission, and proves the sampled interval meets the frozen duration;
- serious, critical, unknown, power drift, malformed timing, and timeout still fail closed, and
  retained before/after equality remains exact;
- hash-bound runner-failure schema v1 records the source/binary/runner/manifest/model/row/policy,
  log/evidence, and retained before/after diagnostic with `promotable:false`; it never enters the
  promotion receipt set;
- parent artifacts are published atomically through authenticated boundaries so hostile child
  unlink, symlink, future-directory, aggregate, PID, or output-root replacement cannot create
  promotable or out-of-boundary evidence.

Fresh recovery proof passes 519 HarnessCore XCTest plus 17 Swift Testing tests off-box, 95/95
FastMLXHarnessTests and 155/155 SpikeCoreTests through Xcode with
`-skipPackagePluginValidation` on the bench, the Release build, two final no-issue reviews,
ShellCheck, diff checks, and staged gitleaks. The clean binary SHA-256 is
`cfe029ad2138013a5904e6afd2475a881081a37bcf89db25bfbb91abf8484397`; runner SHA-256 is
`e2f6e55bb0aeae6b1ec585f6d0d3c85b13c00879e13ed2a0fa3826ee074f8c0f`.

V11 is the fifth immutable failed loaded-Qwen boundary. Its clean-`d4102e6` 3-cell x 3-block
preflight stopped after 2/9 authenticated rows and before any block completion. FP16 and KVarN
materialize each satisfied the 60-second continuous admission and exact nominal/nominal retained
equality. Block 0, position 2, direct KVarN reacquired nominal for 60 seconds after a fair warmup,
then changed nominal -> fair during its roughly 144-second retained measurement. Failure-receipt
SHA-256 is `02893f30229f79861f95e8d037536ae6d0bca7855539bbc88ab2f82dd788293b`;
partial receipt-list SHA-256 is
`ae9dcb7e0ef082aff6cd6509c13a9e41f85fe8937d0a203867d762559c259a56`; partial block-receipt
SHA-256 is `2780c3a9b4880236082feb3dcca26613f69220c854d68dc0db39f67ea0e7ac90`; and runner-log SHA-256 is
`02ade4793feafc228a22c5a99afc8fe8ce4ab14252abaece25805dc8eb0bc291`. The failure receipt is
`promotable:false`, carries no evidence hash, and is absent from both partial promotable receipt
artifacts. Preserve v11 unchanged.

The recovery gate now moves from thermal admission to loaded direct-path performance diagnosis.
V11 proves that another unchanged dwell or workload-only preheat cannot establish retained nominal
equality for this heat-generating candidate. The diagnostic also reported direct KVarN at roughly
63.26 prefill tok/s and 7.18 decode tok/s versus fp16 at
533.73 and 23.48. Those partial numbers cannot support a public or promotion claim, but they close
unchanged thermal retries as an engineering strategy. Profile the actual loaded direct KVarN path;
implement one measured recovery only if it retains a credible route to the unchanged 5% decode and
prefill gates. Otherwise SHELVE KVarN's speed role and retain its capacity-only disposition. Do not
launch another preflight or a reduced/full promotion matrix until that decision and any matrix
scope amendment receive focused review.

### Loaded direct-KVarN disposition and matrix amendment — 2026-07-21

The bounded diagnostic ran on clean `d4102e6a3029b161d99ee27aceabbad8d5696fb5` with Release
binary SHA-256 `cfe029ad2138013a5904e6afd2475a881081a37bcf89db25bfbb91abf8484397`.
The operator had already selected High Power Mode; before and after the run, `pmset` reported
`powermode 2`, Foundation reported Low Power Mode false and thermal nominal, and the host remained
on a 140 W AC adapter. `system_profiler` reported contradictory High/Low Power labels on this host;
the harness does not use those labels as its readiness authority.

The diagnostic boundary is preserved unchanged at
`/Users/llmbench/perf-work/results/fused-compressed-kv-profile-d4102e6/qwen-8k-kvarn-direct-metal-v1`.
Its launch receipt binds the clean source, binary, model config, v11 qualification reference,
Xcode/xctrace version, and non-promotable status. `xctrace` reached the 240-second Metal System
Trace limit, produced a 16,441,155,536-byte Apple Trace File, then remained live beyond the
launcher's bounded save interval. The watchdog terminated the recorder and target at 307 seconds.
Peak recorder RSS
was 107,315,600 KiB. The raw trace SHA-256 is
`12073b786fb06d5569269500129bee3f9b1926319f9a34a9de97c5cdf24853ea`; terminal status SHA-256 is
`982b2e8659cd53abb2c403c15e949cd6268fffa94f48c06cb85837141c2958b2`.
Both the trace bundle and its raw `.atrc` payload fail `xctrace export --toc` with
`Document Missing Template Error`. The artifact is diagnostic failure evidence only; it cannot
attribute a kernel, promote a cell, or be repaired in place.

The retained v11 diagnostic and source map are nevertheless sufficient for the engineering
scope decision:

| Qwen3-32B 8K diagnostic cell | Prefill tok/s | Decode tok/s | Status |
| --- | ---: | ---: | --- |
| fp16 | 533.73 | 23.48 | authenticated v11 row |
| KVarN materialize | 236.92 | 0.46 | authenticated v11 row |
| KVarN direct | 63.26 | 7.18 | hash-bound failed-row diagnostic; non-promotable |

Direct KVarN is about 15.61x faster than its same-storage materialize control in decode, so the
compressed-domain route removes a real materialization penalty. It is still only 30.6% of fp16
decode. The unchanged gate requires at least 24.65 decode tok/s, about 3.43x the observed direct
rate. The prefill gate requires at least 507.04 tok/s, about 8.02x the observed direct rate. The
loaded source path also contains independent costs rather than one isolated toggle: 512-token
prefill graph/eval boundaries, host tile packing plus eight-iteration KVarN normalization, and
capacity-wide packed key/value attention work at every layer. A single test-first,
actor-confined recovery has no credible route to both required multipliers.

**Decision:** SHELVE KVarN's speed role for this Qwen3-32B cycle. Retain the previously qualified
KVarN i8 capacity-only Max-fit role, exact lifecycle/correctness tests, and direct implementation as
research evidence. Do not expose, imply, or benchmark-promote a KVarN speed tier. A future revival
requires a materially new kernel/algorithm design and a new task, not another parameter, thermal,
or trace retry inside this gate.

**Reviewed remaining-cell amendment:** replace the abandoned seven-cell speed matrix with a fresh
five-cell 5x5 cyclic matrix containing exactly:

1. fp16;
2. affine K4V2-g64 materialize;
3. affine K4V2-g64 `split-affine-quantized-mm`;
4. frozen Qwen KVTuner materialize; and
5. frozen Qwen KVTuner `split-affine-quantized-mm`.

The matrix keeps the frozen workload/model/checkpoint/tokenizer/KVTuner identities, one fresh
process per position, explicit memory/cache/wired limits, schema-v3 manifest, schema-v4 evidence,
60-second continuous nominal/AC/non-low-power dwell, exact retained equality, and the original 5%
decode/prefill gates. It must use a fresh output and new clean source/build identity after this
amendment is reviewed. KVarN is measured separately as capacity-only at Qwen 32K and is excluded
from speed aggregation. Any incomplete row or environment drift still invalidates its paired
block; no old row is imported into the new matrix.

### Phase 4 — end-to-end Qwen frontier

- [ ] Run 8K as the bounded smoke, then 32K only from the clean verified SHA; record the
  authenticated 128K refusal because Qwen3-32B max context is 40,960.
- [ ] At 8K, run only the amended five-cell speed scope: fp16 plus affine K4V2-g64 and frozen
  KVTuner materialize/direct pairs. At 32K, retain that speed scope and collect KVarN i8 only as a
  separate authenticated capacity/runtime-context row with no speed aggregation or label.
- [ ] Preserve negative/dominated and hard-floor-failed rows rather than filtering the matrix.

### Phase 5 — second family and adjudication

- [ ] Source-lock the selected Llama-3.3-70B-family checkpoint and repeat the applicable 8K
  smoke/32K/near-128K gate with prompt+output <= 131,072.
- [ ] Keep Qwen-specific KVTuner unavailable unless separately calibrated and authenticated for
  Llama.
- [ ] Add a third popular, materially different attention geometry before broad/default product
  support claims. The current first candidate is
  [`mlx-community/Phi-4-mini-instruct-4bit`](https://huggingface.co/mlx-community/Phi-4-mini-instruct-4bit/tree/ac1c269cb4222a4e136a3d09edad301056c1f36a)
  at revision `ac1c269cb4222a4e136a3d09edad301056c1f36a`, using its `phi3` registry identity
  (Q24/KV8/D128, partial RoPE); prove registry/load/runtime support before accepting it. Gemma 3
  remains the next rotating local/global-cache semantic boundary, not a substitute for that load
  proof.
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
