# KVarN / asymmetric KV-cache frontier implementation plan

**Status:** complete — closed 2026-07-18 with selected lossy tiers and no lossy speed win

**Technique class:** `LOSSY_FRONTIER`

**Source lock:** [`docs/reference/kvarn-kv-algorithm.md`](../../reference/kvarn-kv-algorithm.md)

**Task seed:** [`docs/task-inbox/2026-07-12-kvarn-kv-frontier.md`](../../task-inbox/2026-07-12-kvarn-kv-frontier.md)

## Outcome and user story

A user with limited unified memory should be able to exchange KV-cache precision for more context
or concurrency, see the measured loss/capacity/speed point, and choose an aggressive setting when
their workload tolerates it. fast-mlx must prevent incoherent tiers, but it must not collapse every
lossy result into a binary “near-lossless or rejected” judgment.

The cycle selects a storage winner before custom Metal work. It does not claim a production
throughput win from a reference materialize-then-attend implementation.

## Acceptance criteria and proof

1. **Reference fidelity.** A pure Swift KVarN tile transform reproduces the pinned official
   fixture: packed payload and fp16 metadata byte-identically; dequantization remains finite and
   has the expected shape. Proof: local `HarnessCoreTests`, red before implementation and green
   after.
2. **Honest size.** A format-aware integer accountant includes payload, scale/bias metadata,
   alignment, fp16 sink/recent window, incomplete tail, and workspace. Its prediction equals
   the sum of real MLX cache-array `nbytes` for representative geometries. Proof: pure boundary
   tests plus on-box `SpikeCoreTests`.
3. **Fail-closed format contract.** Unsupported bit widths, invalid groups/head dimensions,
   partial reference tiles, missing/duplicate KVTuner layers, model-hash mismatch, arithmetic
   overflow, and non-finite source values produce explicit errors. There is no fp16 fallback.
4. **Runnable controls.** Same-weights fp16 KV, affine K4V2/K8V2 (groups 64/128), affine K4V4,
   frozen KVTuner schedule cells, and KVarN K4V2-g128 are independently selectable and carry
   structured engagement plus actual-storage evidence. Proof: on-box cache/factory/engagement
   tests and a one-row real-model smoke for every tier.
5. **Context-locked quality.** Every lossy cell records teacher-forced short KL, pooled
   diagnostics, perplexity delta, top-1 agreement, and 24K tail-p95 against the same checkpoint
   with fp16 KV. The record pins candidate and reference model hashes, matrix/cell identity,
   corpus hash, and clean git SHA. Proof: harness schema tests and clean-SHA JSONL.
6. **Coherence and task evidence.** Math, code, structured/tool, and long-retrieval checks exercise
   error accumulation. They use frozen prompts distinct from KVTuner calibration and never replace
   teacher-forced metrics. Proof: task artifact with per-domain denominators and reference deltas.
7. **Apple frontier.** Batch-1 prefill/decode, peak unified memory, exact allocated KV bytes,
   usable context/concurrency, and quality are reported on the bench Mac for every retained cell.
   Proof: fresh bench artifacts stamped by `sync_llmbench.sh`.
8. **Flywheel closure.** A dated verdict classifies each non-dominated cell, explains user-visible
   speed/capacity versus loss, and either selects the next fused-attention format or shelves KVarN.
   A `docs/content/` piece explains the finding without turning repository claims into local data.

Happy path: Qwen3-32B 4-bit weights run the same corpus positions under fp16 KV, affine K4V2/K8V2,
a frozen per-layer schedule, and KVarN K4V2-g128; a non-dominated format passes the coherence floor
and advances to the fused-attention queue.

Recovery/edge paths: a partial tile remains explicit fp16 tail storage; an unsupported model or
schedule is rejected before loading; a nominal bit label whose real arrays exceed the prediction
fails the evidence gate; a tier that is fast or small but crosses the coherence floor is not exposed.
Continuous-batching integration and a fused Metal kernel are intentionally out of this storage gate.

## Predeclared dial interpretation

The labels describe measured loss; they are not all-or-nothing promotion gates:

| Dial label | Marginal same-weights KV loss contract | Exposure |
|---|---|---|
| **Transparent** | Ppl delta <=1%, KL median <=0.05 nats, and task deltas inside the declared confidence/noise interval | Eligible for automatic/default selection when it also offers a useful capacity or speed win |
| **Balanced** | Ppl delta <=5%, KL median <=0.2 nats, and no domain loses more than 5 absolute points | User-selectable with the measured warning; not silently selected for strict code/tool workloads |
| **Max-fit** | Above Balanced but passes every hard coherence-floor predicate below | Explicit opt-in for constrained hardware, maximum context, or loss-tolerant workloads |
| **Rejected** | Any hard-floor failure, non-finite behavior, provenance failure, or false storage claim | Not exposed as a dial tier |

The **hard coherence floor** is deliberately much looser than the default-quality bar so users retain
real freedom: finite logits at every scored position; marginal candidate perplexity less than 2x the
same-weights fp16-KV reference; long-context tail-p95 below 5 nats; teacher-forced top-1 agreement at
least 50%; no task domain below both chance/empty-baseline and 50% of its fp16-KV score; and at least
90% syntactically valid structured/tool outputs. These predicates will be printed individually.
They may be revised only before viewing candidate measurements, with a dated rationale.

A format advances when at least one cell is Pareto-nondominated in actual bytes, quality, and observed
runtime and passes the hard floor. “Transparent” is required only for default-on, not for manual dial
exposure. KVarN is shelved if affine/KVTuner dominates every KVarN cell at equal-or-lower actual bytes,
or if all KVarN cells fail the floor.

### 2026-07-15 pre-evaluation KVTuner search-budget lock

The first clean Qwen3-32B g128 sensitivity artifact was captured at harness SHA `e044ae9` before
any candidate output was generated or inspected. Its 3,840 canonical samples produce 64 singleton
DBSCAN groups: 63 layers retain K8V4/K8V2/K4V2 and layer 47 retains K8V4/K4V2. Exact dynamic
programming over those groups gives 718,704 schedules at the nominal equal-byte target of 400
pair-bits, so the authenticated runner correctly exceeds its 10,000-candidate safety cap. The
nearby counts are 3,969 at 394, 1,953 at 392, 64 at 390, 63 at 388, and one at 384.

Candidate scoring is the pinned 200-prompt, up-to-256-token GSM8K evaluation for every schedule;
3,969 candidates would permit more than 203 million generated tokens. Before observing quality,
the exhaustive qualification cell is therefore locked to **390 pair-bits**, the highest nearby
budget with at most 100 candidates. It evaluates all 64 possible placements of one K8V4 layer over
an otherwise K4V2 policy. At g128 this is 105.5 persistent bytes/token/head averaged across layers,
below KVarN K4V2-g128's 108 bytes/token/head, so it is a valid stricter size control rather than an
inflated quality comparator. The 400-bit topology remains recorded as a search-complexity finding
and may be revisited only with a separately qualified search reduction or batched evaluator.

## Architecture and work order

### Phase 0 — source, fixture, and evidence contract

- [x] Pin KVarN and KVTuner primary sources and separate paper evidence from post-paper repository
  claims in `docs/reference/kvarn-kv-algorithm.md`.
- [x] Generate a deterministic reference fixture with the pinned KVarN pure PyTorch functions.
  Record source commit, script hash, explicit iteration count, input, packed bytes, fp16 metadata,
  and reconstruction values. Do not import the upstream vLLM fork as a runtime dependency.
- [x] Move the KL evidence payload into testable pure code and add `sameWeights`, reference model
  config/checkpoint hash, matrix ID, cell ID, format geometry, actual storage bytes, and comparison
  baseline. Switch the promotion path to fail-closed evidence writing.

Phase 0 evidence proof (2026-07-14): commit `e1a3cff6cc95ffcd980cc8e05709fdb0ab7edc39`
makes candidate/reference identity, same-weights baseline, matrix/cell identity, canonical format
geometry, predicted-versus-actual storage, short/long cohort counts, deepest actually scored context,
teacher-forced top-1, quality floors, clean SHA, and runtime/corpus provenance one typed validation
boundary. It also serializes appenders while validating and synchronizing JSONL, so concurrent
promotion jobs cannot silently lose rows. The final pure run passes 211 XCTest + 17 Swift Testing
tests; the focused evidence slice passes 28/28. Final line coverage is 93.63% for
`KVFrontierEvidence`, 85.12% for `Provenance`, and 87.50% for `QualityMetric`. Independent focused
review found no remaining issue.

The clean-SHA bench pass builds Release and passes 48/48 `SpikeCoreTests` through Xcode. A
same-checkpoint Qwen3-4B fp16-KV smoke records all five corpus entries, including distributions
actually scored at a 24,150-token context; its exploratory JSONL validates. The identical
`--promotion-evidence true` run exits 1 with `missingFormat` and creates no JSONL because Phase 2
has not yet connected real cache-array geometry/bytes. This is the intended boundary: Phase 0
proves that incomplete evidence cannot be promoted; it does not manufacture an affine result.

### Phase 1 — pure format/reference layer, test first

Files planned:

- `spike/Sources/HarnessCore/KVQuant/KVStorageFormat.swift`
- `spike/Sources/HarnessCore/KVQuant/KVarNReference.swift`
- `spike/Sources/HarnessCore/KVQuant/KVTunerSchedule.swift`
- matching `spike/Tests/HarnessCoreTests/` files and a compact checked-in fixture

- [x] Write failing layout tests for K4V2-g128 (13,824 bytes/tile/head; 108 bytes/token/head;
  3.375 effective bits/element), g64 metadata, affine cells, fp16 sink/tail, alignment, concurrency,
  workspace, overflow, and partial-tile accounting.
- [x] Implement the smallest format-aware integer accountant; keep the old approximate
  `KVQuantTier.bytesPerElement` from being used as evidence for new formats.
- [x] Write the failing pinned-fixture test, then implement normalized Hadamard rotation, log-domain
  variance balancing, asymmetric RTN, low-bit pack/unpack, absorbed scales, and dequantization.
- [x] Write schedule-validation failures first, then implement a versioned JSON artifact with model
  hash, calibration provenance, objective/budget, and complete per-layer K/V widths.
- [x] Run focused tests, the full pure suite, and coverage. Commit this phase independently.

Phase 1 proof (2026-07-14): the intended red tests failed on aggregate overflow, invalid
Hadamard allocation, axis packing, unrunnable preset geometry, and missing fixture provenance.
The green run passes 21/21 focused tests and the full 187 XCTest + 17 Swift Testing pure suite.
Final line coverage is 98.95% for `KVStorageFormat`, 97.22% for `KVTunerSchedule`, and 99.14% for
`KVarNReference`, with 100% function coverage for all three. Focused review found no remaining
correctness issue.
The accountant is explicitly the planned tight fast-mlx layout, not the pinned vLLM block
allocator; Phase 2 must reconcile it to real MLX array bytes before any capacity claim.

### Phase 2 — MLX affine controls and actual-byte telemetry

- [x] Add an actor-confined `AffineKVCache` conforming to `CompiledCache`, with independent K/V
  bits and group sizes, native packed MLX arrays, materialize-then-attend reads, chunked growth,
  reset, and truncation. No `Sendable` escape hatches.
- [x] Extend `KVCacheKind`, CLI parsing, scoring-cache construction, and engagement telemetry with
  fail-closed affine tier names. Keep spec-decode combinations rejected until independently measured.
- [x] Add actual array dtype/shape/`nbytes` telemetry and reconcile it exactly with Phase 1.
- [x] TDD on-box with `xcodebuild ... -skipPackagePluginValidation`; never use `swift test` for
  this MLX-importing target.

Phase 2 proof (2026-07-14): commits `4464ac9`, `d4bda35`, and `4c1bfe1` add the native affine
cache, independently selectable controls, exact array/workspace accounting, and per-corpus-entry
teacher-forced sample evidence. The MLX-coupled suite passes 56/56 through Xcode at the final
engine SHA; the current clean evidence SHA builds Release. The full pure suite passes 217 XCTest +
17 Swift Testing tests. Focused review found and closed an evidence-writer path that had accepted
detail-less exploratory rows; re-review found no remaining issue.

All five declared affine controls engaged in Qwen3-4B plumbing smokes. The preliminary
K4V2-g128, K4V2-g64, K8V2-g64, K8V2-g128, and K4V4-g128 triads passed 4/8, 7/8, 7/8, 7/8, and
3/8 predicates respectively. These deliberately undersampled runs prove selection and evidence
flow only; they are not monotonicity findings or dial verdicts. On clean `4c1bfe1`, K8V2-g128's
predicted and actual allocation agree byte-for-byte: 1,112,832,000 payload + 55,641,600 metadata +
98,918,400 workspace = 1,267,392,000 bytes, with 144 control bytes. Every one of the three short
and two long entries carries its real ID and scored-position count. Promotion mode rejects the
2-position-per-entry smoke, exits nonzero, and creates no JSONL; the full Qwen3-32B matrix still
owes at least 24 positions per short entry and 128 per long entry.

### Phase 3 — KVarN correctness-first MLX cache

- [x] Port the reference with MLX operations and native packed affine storage after variance
  normalization. Keep the fp16 sink and incomplete tail explicit.
- [x] Verify the MLX transform against the same fixture and the pure Swift oracle, including 8
  versus 16 iterations and 2-bit V packing.
- [x] Integrate KVarN as an unambiguously named cache kind with engagement/storage telemetry.
  If tile-boundary mutation cannot be captured safely, use the explicit uncompiled correctness path
  and record that fact; do not weaken actor confinement or pretend it is a speed path.
- [x] Add frozen KVTuner per-layer cache selection only after uniform affine cells pass.

Phase 3 proof (2026-07-18): the final clean runtime evidence is stamped
`f88d26eeb3793cc1d5d8f2118043c190977ee6e0`. The KVarN canonical memory artifact is
SHA-256 `7b21459cf1afc1a038b87c017c1fdacd18644b08b4b63a07318d74faba2dcfd2`; the task/KL
source packet is `d9071a93955be2e148dc381638d4a71a8286c59e`; and the KL manifest is
SHA-256 `3be8693aa169e2e5b4c2692f7fdc115783271fb735d27e50b0d5c3cb798990df`. Final local/on-box
verification recorded 389 XCTest + 17 Swift Testing pure tests, 30 `FastMLXHarnessTests`,
92 `SpikeCoreTests`, and a Release build. KVarN remained on the explicit uncompiled correctness
path, so this phase proves correctness/selection/storage telemetry, not a compiled speed path.

The present storage frontier remains intentionally batch-1. Before any selected quantized format
is admitted to continuous batching, add the adversarial compaction case: merge unequal rows,
remove the longest/zero-padding boundary row, append again, then prove physical end, mask width,
survivor bytes/logits, and dense-control parity. Upstream precedent shows that deriving the write
end from surviving padded offsets can corrupt the cache; that is a correctness bug, never dial loss.

### Phase 4 — clean-SHA matrix on the bench Mac

- [x] Sync with `spike/scripts/sync_llmbench.sh`, build Release through Xcode, and record model,
  package, harness, hardware, OS, cache-limit, corpus, and checkpoint provenance.
- [x] Run same-weights fp16-KV pipeline floor, then the declared affine/KVTuner/KVarN cells at
  identical positions. Run the lossy triad and coherence predicates before expensive 24K scoring.
- [x] Measure actual cache bytes, peak active/cache/RSS memory, batch-1 prefill/decode, and capacity.
  Any raised wired-memory limit is paired with an explicit `Memory.cacheLimit`.
- [x] Preserve raw JSONL/log artifacts and derive a compact matrix without dropping failed cells.

Phase 4 proof (2026-07-18): the compact evidence is
[`docs/superpowers/verdicts/kvarn-kv-frontier-evidence-2026-07-18.jsonl`](../verdicts/kvarn-kv-frontier-evidence-2026-07-18.jsonl)
and the final adjudication is
[`docs/superpowers/verdicts/2026-07-18-kvarn-kv-frontier.md`](../verdicts/2026-07-18-kvarn-kv-frontier.md).
The matrix is model-specific to Qwen3-32B-4bit on the bench Apple system. It reports actual
cache-array storage plus workspace, teacher-forced KL/perplexity/top-1/tail metrics, task checks,
batch-1 prefill/decode, and peak RSS. Every lossy retained row improved KV-budget capacity, but no
lossy row was a measured speed win on the materialize-then-attend path.

### Phase 5 — adjudication and check-in

- [x] Write `docs/superpowers/verdicts/2026-07-18-kvarn-kv-frontier.md` with one finding per cell,
  user-facing dial labels, marginal loss, actual capacity, speed, caveats, and PROMOTE/SHELVE outcome.
- [x] Write a `docs/content/` explanation centered on what the Apple evidence changed.
- [x] Produce the verification packet, run a focused review, secret-scan, and inspect the diff.
- [x] Commit coherent changes with the required co-author trailer and merge `--no-ff` only after
  fresh proof.
- [x] If a format advances, update the next queue item to fused compressed-domain attention using
  the selected layout; otherwise record why the more exotic kernel investment was avoided.

Phase 5 proof (2026-07-18): the selected verdict is fp16 KV as the **Transparent** baseline,
affine K4V2-g64 as **Balanced** capacity, frozen KVTuner as explicit **Max-fit** capacity, and
KVarN i8 as **Max-fit capacity-only** plus the fused compressed-domain attention candidate.
The K8 affine controls and KVarN i16 were shelved when they failed to create a separate practical
frontier; the two g128 4-bit affine cells were rejected for hard-floor failure. The reader-facing
writeup is
[`docs/content/2026-07-18-when-smaller-kv-is-not-faster.md`](../../content/2026-07-18-when-smaller-kv-is-not-faster.md).
This is not a broad-model claim: a second materially different popular model family must pass the
same gate before generalizing beyond Qwen3-32B-4bit. The final packet is recorded in
[`docs/verification-evidence.md`](../../verification-evidence.md): pure and MLX-coupled suites,
Release build, JSON/link/diff checks, focused re-review, and staged secret scan all pass. The
coherent closure commit `687b09f` was merged `--no-ff` to `main` as `f435312` after that proof.

## Build and safety invariants

- Pure `HarnessCore` tests run locally with SwiftPM. Anything importing MLX runs on
  `llmbench@192.168.1.252` via Xcode with `-skipPackagePluginValidation`.
- MLX state remains actor-confined. `@unchecked Sendable` and `nonisolated(unsafe)` are forbidden.
- Loss is teacher-forced. Free-running tasks are secondary coherence checks.
- Speculative decoding at temperature 0 remains byte-identical; lossy KV plus PLD stays rejected
  until separately qualified.
- Unknown tiers, schedule mismatches, non-finite values, and evidence-write failures stop the run.
- Machine-local paths and raw bench logs remain out of git; durable compact evidence carries hashes.
