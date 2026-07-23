# Verification Evidence

This file preserves concise proof for real user and operator outcomes. Large raw logs remain
in the project evidence store; committed records name the artifact, its hash, and the source
SHA without copying machine-local paths or transient output.

## Acceptance Matrix

| Story | Acceptance criterion | Automated proof | Bench / integration proof | Last verified | Gaps |
| --- | --- | --- | --- | --- | --- |
| As an operator, I can enable PLD without changing greedy output. | Temperature-0 output is byte-identical with real drafting, for both verify strategies. | `SpecDecodeTests` pins full/zero/partial prefetched accept walks and exact emitted prefixes. | Two `verify --spec pld --n 120` records: 120/120 identical, engagement true, triad PASS, compiled verify off/on. | 2026-07-11 | Temperature >0 is not part of the exact-greedy contract. |
| As an operator, I retain PLD's target-workload multiplier. | Echo gain is at least +80%. | Exactness tests protect the accept/bonus and rollback contract used by the fast path. | 28.28 → 56.70 tok/s, **+100.50%**, 684/696 accepted; compiled verify also +86.10%. | 2026-07-11 | Qwen3-32B-4bit on one reference hardware tier. |
| As an operator, leaving PLD available does not tax cold or low-yield requests. | No non-target regression beyond −1%; empty-draft fallback uses the base pipeline; low yield disables quickly and recovers. | `testGate_defaultLowYield_disablesAfterFourSamples`, partial-window and cooldown tests; 26/26 `SpecDecodeTests`. | Code 28.39 → 29.31 (**+3.24%**); zero-draft prose 28.62 → 28.66 (**+0.14%**). | 2026-07-11 | PLD remains disabled under continuous batching. |
| As a maintainer, the change preserves MLX build and concurrency invariants. | Pure suite and MLX-coupled suite pass; no banned concurrency escape in changed files. | 113 XCTest + 17 Swift Testing tests off-box; 20/20 `SpikeCoreTests` through Xcode on-box. | Clean feature SHA stamped into all seven bench and two exactness records; focused reviews found no issues. | 2026-07-11 | Full serving/API default wiring is not implemented by this kernel change. |
| As the owner, I can reject a trained speculator before paying for a Swift port. | Checkpoint/head fidelity is proven first; every implemented temperature-0 verify shape is byte-identical and engaged, otherwise the arm fails closed without a speed claim. | 119 HarnessCore XCTest + 17 Swift Testing tests; 21 MLX-free EAGLE preflight tests; authenticated file-manifest and cache-drift classifier regressions. | Clean `1a70c4d`: checkpoint + parity PASS; 4-bit exactness FAIL at generated index 17; 8-bit FAIL at index 7; both output hashes differ and return exit 1. | 2026-07-12 | EAGLE is shelved. `k=3`, long-output throughput, BF16, and the Swift port intentionally did not run after the earlier hard failure. |
| As a maintainer, I can enter continuous-batching actor integration only after dense cache shapes are exact and compile-stable. | Scalar-aligned merge/append/filter/extract preserves logits, greedy tokens, row identity, and one trace per stable shape; unsupported state layouts fail closed. | 9 focused `BatchedCompiledKVCacheTests`; 29/29 total `SpikeCoreTests` through Xcode; 134 HarnessCore XCTest + 17 Swift Testing tests off-box. | Clean `7b9d709`: Qwen3-32B-4bit B=1/2/4/7/8 fixed+shapeless PASS with zero initial max-logit delta; 64-step B=4/B=8 fixed PASS; `qwen3_moe` rejected before load. | 2026-07-12 | Historical Phase-1 gate; Phase 2 and Phase 3 are closed below. |
| As an operator, concurrent dense streams can join, leave, and prefill in chunks without changing greedy output or exhausting an unbounded queue. | B1→drain→B2→B1 and B3→B2 remain token/byte exact; decode precedes bounded prefill; queue, per-request context, and aggregate logical context are capped; speculation is rejected locally for this runtime. | 16 scheduler + 10 coordinator pure tests within 145 HarnessCore XCTest + 17 Swift Testing; 12 cache-history-sensitive runtime tests within 41/41 Xcode `SpikeCoreTests`. | Clean `2a5a5f4`: Qwen3-32B-4bit staggered join and middle-cancel probes PASS; Qwen3-4B-4bit chunk-size-1 interleave PASS; zero batched speculation. | 2026-07-12 | Historical Phase-2 gate; byte admission, service frontier, cancellation latency, and soak are closed by Phase 3 below. |
| As a dense-Qwen service operator, I can select the measured exact policy for an isolated request or a simultaneous burst without hidden queueing, state-poison, cancellation, or resident-memory failure. | Same-workload C=1/2/4/8 frontier selects solo PLD at C=1 and batch-no-spec at C≥2; exact transition/cancellation, conservative byte admission, A/B/A, responsiveness, and the full 24-hour RSS gate pass. | 166 HarnessCore XCTest + 17 Swift Testing tests off-box; 48/48 `SpikeCoreTests` through Xcode; final-SHA real-model probes and fail-closed `qwen3_moe` smoke. | Clean `7a775f6` frontier: batch +45.8%/+58.6%/+74.7% at C=2/4/8; clean `0aed280` soak: 86,412.85 measured seconds, all 33 predicates 3,519/3,519, peak RSS drift 2.2444%. | 2026-07-14 | Explicit probe path only; production routing/API, network disconnect propagation, sampling, non-dense state, other models and hardware remain open. |
| As a constrained-hardware operator, I can choose a measured KV-capacity tier with quantified loss, while incoherent cells remain unavailable. | Same-weights Qwen3-32B cells bind actual storage, teacher-forced quality, task coherence, runtime, and clean provenance; hard-floor failures reject; every promoted label states capacity versus speed. | 389 HarnessCore XCTest + 17 Swift Testing tests off-box; 30/30 `FastMLXHarnessTests`; 92/92 `SpikeCoreTests`; Release build through Xcode. | Clean `d9071a9` task/KL packet and `f88d26e` runtime packet: Transparent fp16, Balanced K4V2-g64, Max-fit KVTuner, capacity-only KVarN i8; two aggressive cells reject. | 2026-07-18 | Qwen3-32B-4bit on one M5 Max; no lossy speed win, compiled KVarN/fused attention, or broad-model claim. |
| As an operator, I can qualify a second model family against an exact source-locked checkpoint and tokenizer rather than a mutable cache alias. | Exact public repo/revision, API/tree/per-file identities, full shard contents, tokenizer, index, and Llama geometry authenticate; loaded speed/capacity/quality/task claims remain model-scoped and fail closed independently. | Clean source-lock `dcfbbe3`; loaded speed `c8a56ef`; generalized capacity `3493314`; sealed quality/task `9033725`; strict source, runtime, capacity, sealed-replay, and task evidence contracts. | Source lock PASS; 8K affine speed NEGATIVE; 32K speed HARDWARE-UNAVAILABLE; 32K affine capacity NON-PROMOTABLE; near-128K NOT RUN; long-depth quality NON-PROMOTABLE. | 2026-07-23 | Both first families share Q64/KV8/D128. The independent Phi3 Q24/KV8/D128 follow-on is closed in the next row; Qwen KVTuner is not reusable. |
| As an operator, I can tell whether compressed attention is portable to a materially different head geometry without mistaking fit or a decode-only win for a safe default. | The exact source-locked Phi3 Q24/KV8/D128 inert-window geometry must pass registry/load/runtime/failure-path proof; generic sliding windows must still reject; speed, capacity, teacher-forced loss, and task-reference admission remain independent gates. | Clean admission `850997e`; loaded evidence `9c542ef`; deep corpus `19cbb8e`; corrected LongRoPE teacher `e29ee4c`; 541 HarnessCore XCTest plus 17 Swift Testing tests off-box; 113/113 `FastMLXHarnessTests`, 157/157 `SpikeCoreTests`, and Release through Xcode. | Source/registry/load PASS; 8K direct speed NEGATIVE; 32K and near-128K capacity NON-PROMOTABLE; affine teacher-forced floor PASS with material loss; fp16 task reference INVALID, so paired affine task refused and no quality tier exists. | 2026-07-23 | Exact checkpoint only. No Phi speed/quality/default tier, no generic sliding-window admission, and no cross-family KVTuner reuse. |

## Current Verification Commands

```sh
# Pure/Foundation checks
cd spike
swift test --filter HarnessCoreTests
swift test --enable-code-coverage --filter SpecDecodeTests
```

```sh
# From the synced tree on the configured Apple-Silicon bench host
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild test -scheme fast-mlx-spike-Package -destination "platform=macOS" \
  -skipPackagePluginValidation -only-testing:SpikeCoreTests

# Run in a separate xcodebuild invocation to isolate pointer-keyed MLX compiled state.
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild test -scheme fast-mlx-spike-Package -destination "platform=macOS" \
  -skipPackagePluginValidation -only-testing:ExactPrefixMLXTests

<fastmlx-harness> verify --model <model> --spec pld --n 120
<fastmlx-harness> verify --model <model> --spec pld --n 120 --compiled-verify true
bash scripts/bench_pld_shapes.sh <model> 3 256

<spike-cli> batch-stream-probe --model <dense-qwen3> --max-tokens 16 \
  --join-after 2 --prefill-chunk 16
<spike-cli> batch-membership-probe --model <dense-qwen3> --max-tokens 8 \
  --cancel-after 2 --prefill-chunk 16

<fastmlx-harness> service-bench --model <dense-qwen3> --policy <batch-no-spec|solo-pld> \
  --scenario burst --concurrency <1|2|4|8> --runs 3 --workload-nonce <shared-id>
bash scripts/bench_continuous_service.sh <dense-qwen3> 3 128 16
<fastmlx-harness> service-cancel-bench --model <dense-qwen3> --runs 5
<fastmlx-harness> service-state-poison-bench --model <dense-qwen3> --runs 3
bash scripts/soak_continuous_service.sh <dense-qwen3> 86400 4 64 16
```

## PLD Acceptance Pass — 2026-07-11

- **Date:** 2026-07-11
- **Change:** PLD gate tuning, feature `bb5b06f22dc62e258c5ee1bdaadd6e53e1f8019d`,
  merged to `main` as `29cc453`.
- **Overall verdict:** ALL PASS — exactness, target speed, non-target safety, MLX build, pure
  tests, and SHA provenance.
- **Tests:** 113 XCTest + 17 Swift Testing tests off-box, 0 failures; 20/20 MLX-coupled
  `SpikeCoreTests`, `TEST SUCCEEDED`; exactness 120/120 in both verify modes.
- **Coverage:** `PLDGate.swift` 48/53 lines (90.6%); `SpecAccept.swift` 19/21 lines (90.5%).
- **Focused review:** cache-state / rollback exactness and gate policy both returned no
  correctness, security, or logic findings.
- **Evidence:** `final-bb5b06f.csv` SHA-256
  `5bd9014167121301aea1f7022a141dc7b2d3b2396247934e8041aec4d810e9a1`;
  `final-bb5b06f.jsonl` SHA-256
  `dbff6efa31b7ca3c3d1706e75f5a1dc485d46c5682511dd8416c9e4a1e0af07e`;
  `final-bb5b06f-verify.jsonl` SHA-256
  `d1513972350cd744898292373aa02e71fe886b85c1a3bec74ba5a8c8bedc4690`.
  All nine JSONL records carry the full feature SHA.
- **Skipped / residual risk:** temperature >0, continuous batching interaction, other model
  families, and serving/API default wiring were intentionally out of scope. The harness still
  requires explicit `--spec pld` selection for A/B measurement.

## Evidence Hygiene

- Preserve acceptance proof, command names, dates, environments, skipped checks, and residual risk.
- Do not paste full logs, secrets, tokens, customer data, private host details, transient cache paths, or noisy command output.
- Store large artifacts in a project-appropriate artifact location and record a hash plus a short interpretation here.
- Prefer one concise acceptance pass per meaningful change over repeating raw output.

## Portfolio Research Acceptance Pass — 2026-07-12

**Story:** the owner can see one complete, source-qualified optimization backlog and select
the next flywheel cycle without confusing fast-mlx features, retired-Zig evidence, or
upstream-only capabilities.

| Acceptance criterion | Verdict | Fresh evidence |
|---|---|---|
| Reconcile all durable local portfolio inputs | **PASS** | Enumerated 5 plans, 5 Markdown verdicts, 11 task items (excluding the inbox README), the intake, carry-forward backlog, competitive landscape, operability specs, and current Swift cache/spec/scheduler surfaces. The dated audit has explicit completed, shelved, active, and do-not-reopen sections. |
| Give every new recommendation a disposition and bounded owner | **PASS** | At audit time the handoff ranked 11 performance cycles; after the EAGLE result, the current handoff advances the 10 remaining cycles. Seven new inbox seeds own trained speculation, batching, KV storage quality, fused attention, prefix/session caching, sampling, and learned weight quantization. Research-later/rejected candidates remain in the intake/audit rather than becoming duplicate tasks. |
| Rank by impact, evidence, cost, and Apple fit without promoting research | **PASS** | The audit's scored matrix labels every external number unverified until a clean-SHA fast-mlx run. KVarN storage quality precedes its fused Metal phase; TurboQuant owns its own closure ablations. |
| Primary sources support material claims | **PASS** | Independent source review caught and corrected the EAGLE `acceptance_length` denominator and the post-paper KVarN attribution. Final source re-review: **No issues found**. All 35 public external URLs introduced/touched by this change returned HTTP 200; the pre-existing authenticated Claude Artifact returned 403 as expected. |
| Durable docs are internally navigable and review-clean | **PASS** | Repository-wide relative-link check passed across 56 Markdown files; `git diff --check` passed. Final focused review found no High/Medium correctness or priority issues. |

**Not run:** Swift/Xcode tests and the Apple-Silicon bench. This is a documentation/research
cycle with no engine change; running a technique benchmark would falsely imply that an
external candidate had entered the flywheel. Every performance result in the audit remains a
local historical result or a source-labeled external claim.

**Overall verdict:** ALL PASS for the Sol portfolio-audit acceptance criteria. Engine
promotion gates remain intentionally open in their individual task seeds.

## Qwen3-32B EAGLE-3 Phase 0 closure — 2026-07-12

**Story:** the owner can decide whether the public Qwen3-32B EAGLE-3 checkpoint merits a
production Swift port using authenticated, same-target Apple-Silicon evidence.

**Verdict lane:** [`EXACT`](superpowers/verdicts/README.md). Its byte-identity failure is a
speculative-decoding correctness failure, not a rejection of intentional `LOSSY_FRONTIER`
tiers. Lossy techniques remain eligible when teacher-forced measurements quantify a useful
speed/memory/capacity trade above the coherence/garbage floor.

| Acceptance criterion | Verdict | Fresh evidence |
| --- | --- | --- |
| Accounting is unambiguous | **PASS** | 119 HarnessCore XCTest + 17 Swift Testing tests pass. `SpeculativeAcceptanceSummary` separates proposal acceptance, accepted drafts/round, and inclusive length; zero measured round time now returns no economics projection. `SpeculativeEconomics.swift`: 129 lines, 89.92% line coverage. |
| Checkpoint and MLX head are faithful | **PASS** | Full checkpoint/config hashes and 16 tensor payload ranges match. Pinned PyTorch→MLX fixture: cosine `0.9999815822`, argmax match `1.0`, max absolute error `0.0625`. |
| 4-bit output is exact and engaged | **FAIL (candidate RED)** | Clean SHA `1a70c4d`: first mismatch index 17, bytes/token IDs differ, 23/39 accepted across 39 rounds. Replay classifies `rejected-future-cache-drift` at equal offset 59. |
| 8-bit output is exact and engaged | **FAIL (candidate RED)** | Same clean SHA: first mismatch index 7, bytes/token IDs differ, 25/38 accepted across 38 rounds. Same sequential prefix: one-token argmax `279`, batched-probe argmax `264`, equal offset 49. |
| Losing arm fails closed | **PASS** | Both verify commands exited 1 and wrote failed evidence. No authoritative throughput table, `k=3`, BF16 download, or production Swift implementation followed. |

**Overall verdict:** the **preflight workflow passes** its fail-closed acceptance story; the
**EAGLE candidate is RED / SHELVED**. Apparent failed-run rates are explicitly non-authoritative
because outputs differ.

**Evidence:** [`2026-07-12-qwen3-32b-eagle3-preflight.md`](superpowers/verdicts/2026-07-12-qwen3-32b-eagle3-preflight.md)
and [`eagle3-evidence-2026-07-12.jsonl`](superpowers/verdicts/eagle3-evidence-2026-07-12.jsonl).
Raw bench artifact SHA-256 values: checkpoint `a87be2…e34c`, parity `68c6c9…f1b3`, 4-bit
verify `e9edd3…23d4`, 8-bit verify `e4191c…4686`; every record carries full clean SHA
`1a70c4d6b3dce20f3254226123dd588a3c80f052`.

**Commands:** off-box `swift test --package-path spike --filter HarnessCoreTests`; on-box
`python -m unittest discover`, `inspect_checkpoint.py`, `dump_reference_fixture.py`,
`check_head_parity.py`, and `run_preflight.py verify` for each authenticated target. The
runner sets an explicit 8 GiB MLX cache limit under the raised wired-memory ceiling.

**Docs source review:** CONFIRMED that pinned Speculators uses one target pass plus
longest-prefix acceptance, that mlx-lm's standard trim decrements the logical offset, and that
MLX states equality only up to numerical precision while shapes can trigger compilation work.
UNVERIFIED—and deliberately not claimed—as an upstream fact: that any one MLX 0.32 kernel is
the cause of these observations. The verdict labels that mechanism as inference and relies on
the clean four-history replay for the local result.

## Continuous batching Phase 1 acceptance pass — 2026-07-12

**Story:** a maintainer can proceed from pure scheduling into actor integration only after the
pinned Swift/MLX stack proves exact dense cache membership and stable compiled shapes.

| Acceptance criterion | Verdict | Fresh evidence |
| --- | --- | --- |
| Scalar cache geometry survives batching | **PASS** | 9 focused Xcode tests cover per-row offsets, scalar-aligned right padding, prefix masks, one- and multi-token scatter, merge, ordered filter, extraction, growth, capacity, and invalid state. |
| Real dense model matches scalar decode | **PASS** | Clean `7b9d709`: Qwen3-32B-4bit B=1/2/4/7/8 fixed and shapeless runs all report exact greedy tokens and `0.000000` initial max-logit delta. |
| Stable shapes do not retrace per token | **PASS** | Every 8-step shape records `main_trace_count=1`; B>1 middle-row removal records exactly one new membership trace. Fixed B=4/B=8 also pass 64-step replay with one main trace. |
| Membership preserves request identity | **PASS** | Each run extracts all rows, compares scalar continuation, removes the middle row, and requires rebuilt and in-place-filtered batches to match the independent scalar references in stable order. B=7 explicitly covers B=8 middle removal. |
| Unsupported layouts fail closed | **PASS** | `model_type=qwen3_moe` exits before model load; only dense `qwen3` enters this probe. Authoritative capacity and explicit-length assertions fail before cache mutation. |

**Design finding:** copying upstream left-padding into an already fixed-capacity cache caused an
avoidable exactness failure at close argmax boundaries. Keeping every row at scalar physical
positions removed the delta. This was treated as a correctness bug, not a lossy performance
tier. Intentional future lossy batching variants remain eligible only through the normal
teacher-forced speed↔quality frontier and coherence floor.

**Commands:** off-box `swift test --package-path spike --filter HarnessCoreTests`; on-box Xcode
`SpikeCoreTests`; stamped `spike-cli batch-probe` sweeps for B=1/2/4/7/8 and 64-step B=4/B=8.
The probe sets an explicit 8 GiB MLX cache limit and prints the full clean harness SHA.

**Historical Phase-1 boundary:** this row closed only cache/shape feasibility. Actor integration
and chunked-prefill exactness are covered by the Phase 2 pass below; the service frontier and
soak are subsequently closed by Phase 3.

## Continuous batching Phase 2 acceptance pass — 2026-07-12

**Story:** a dense Qwen3 serving actor can execute decode-first chunked prefill and dynamic
batch membership without changing any request's temperature-zero stream or allowing one
malformed/unsupported admission to poison the service.

| Acceptance criterion | Verdict | Fresh evidence |
| --- | --- | --- |
| Staggered join preserves exact output | **PASS** | Clean `2a5a5f4`: Qwen3-32B-4bit first and second streams are token- and byte-identical to independent compiled scalar baselines across B1→drain→B2→B1. Required transitions are present; batched speculation is false. |
| Chunk boundaries do not change output | **PASS** | Same clean SHA on Qwen3-4B-4bit with `prefillChunkSize=1`: 17 first-prompt chunks and 11 interleaved joiner chunks remain exact through drain, four shared steps, and return to solo. |
| Middle cancellation preserves survivors | **PASS** | Qwen3-32B-4bit B3→B2: both survivors are token/byte exact; the cancelled middle stream is the exact two-token scalar prefix; cancellation reports decoding at emitted count 2. |
| Actor/runtime state transitions are regression-protected | **PASS** | 12 cache-history-sensitive `DenseContinuousBatchRuntimeTests` cover chunking, solo/drain, reorder/removal, B2→B1, compile reuse, admission limits, invalid state, and coordinator integration. Full on-box suite: 41/41. |
| Queue and resource admission are bounded | **PASS (Phase-2 proxy)** | Pure scheduler caps queued requests (default 256) with atomic burst rejection. Dense runtime validates config/vocabulary/context, reserves a full burst's logical context atomically, and releases on every removal path. |
| Pure policy remains green | **PASS** | 145 HarnessCore XCTest tests plus 17 Swift Testing tests, 0 failures. Final focused review: no High/Medium issue; no banned concurrency escape. |

**Provenance:** the three clean replay rows are committed in the
[Phase 2 evidence artifact](superpowers/verdicts/continuous-batching-phase2-evidence-2026-07-12.jsonl)
(artifact SHA-256 `96f403e04abf9cf6c4a80963892a9247f0a65df8e930a7a7c7667599fc6d0c93`;
harness SHA `3cc5f6321afaea80d62b6f894c35d50ece8ad0a5`). Every row records full harness SHA, model config hash, checkpoint
manifest hash (config/index plus shard names/sizes), declared quantization, MLX Swift `0.31.6`,
pinned MLX Swift LM revision, fixed-membership compile policy, and the operation trace. The
Qwen3-32B-4bit config hash is `b3f033c21f563996`; manifest hash `33827ddf1b497615`;
quantization `int4:group=64`.

**Historical Phase-2 boundary:** this was an exactness and lifecycle pass, not the promotion
verdict. Phase 3 below subsequently closes the 1/2/4/8 frontier, cancellation, conservative
byte admission, A/B/A, and required soak. The checkpoint manifest remains a config/index plus
shard-name/size identity rather than a full weight-content digest.

## Latest Acceptance Pass — Continuous batching Phase 3 — 2026-07-14

**Story:** a dense-Qwen service operator can choose the measured exact policy for isolated and
simultaneous work, knowing the comparison uses identical prompts and that transition,
cancellation, state recovery, memory admission, responsiveness, and resident stability have
all passed their gates.

| Acceptance criterion | Verdict | Fresh evidence |
| --- | --- | --- |
| Exact and engaged | **PASS** | Clean `7a775f6f1db9495d60eecdf030bf63d752f936e0`: Qwen3-32B B1→drain→B2→B1 and B3→B2, plus Qwen3-4B chunk-size-1, are token/byte exact; expected shared batches engage; speculation is absent; `qwen3_moe` exits 1 before load. Exact artifact SHA-256 `5db4bb5fecc853f04fd5869326c730b702b19511b3d28a271063a9c336b9d0de`. |
| Same-workload service frontier | **PASS** | All eight rows carry the same clean SHA and `workloadNonce=frontier-20260714-final`; every paired run has identical prompt/output counts; one warmup is dropped and three runs measured. Solo PLD wins C=1 (28.30 vs 26.72 tok/s). Batch wins C=2/4/8 by 45.8%/58.6%/74.7%, with Jain mean 1.0000. JSONL SHA-256 `9404c0aa699ecc37a5575ceb5e13dd2ec75908cc47e06d7b1ee0cf7251c1f70c`; CSV `122d5638ee687873941c1fbe98ce8136b707f86423293ee9715fad3321861d68`. |
| Byte-bounded admission | **PASS** | Dense geometry tests charge rounded capacity, per-row metadata, and the conservative five-copy transition envelope; over-budget bursts reject atomically and every cancellation/completion path releases reservations. This is not claimed as exact whole-process RSS. |
| Cancellation and recovery | **PASS** | Five Qwen3-4B measured runs: p50 19.542 µs, p95/max 21.042 µs <1 s, removal/reuse/survivor/cleanup all pass; raw SHA-256 `963feb1a1ebebf13f35c19a3246fbfa4f09b23adc5d5648a587dca5cdd9367df`. Three A/B/A runs: per-request output byte-identical before/after hostile cancellation, p95/max 12.583 µs, reservations zero; raw `b9f21fd01a315a1a2b0430392493e77b0ae5f1c61df1c1a72842aa0e7ef420dc`. |
| Required resident soak | **PASS** | Clean `0aed28087a3f495e1582977e263bc30e0986703d`: 86,412.8508 post-warmup seconds ≥86,400; 3,518 measured +1 warmup; all 33 aggregate predicates pass 3,519/3,519 and all 3,519 cycle-log lines report A/B/A PASS; max cancellation 28.833 µs <1 s; max responsiveness 344.469 ms <30 s; peak RSS drift 2.2444% <5%; watchdog absent and wrapper cleaned. The JSONL retains full fields for warmup +288 measured checkpoint cycles, all independently rederived. Raw SHA-256 `dcc7e446234cf6defc8746964bdefe3eb80fabe543e93078a096eeb3b9452b22`. |
| Provenance and claim boundary | **PASS** | Qwen3-32B-4bit `int4:group=64`, fp16 KV, M5 Max 128 GB, MLX Swift 0.31.6, pinned mlx-swift-lm revision, explicit 8 GiB cache limit, config/manifest identities. PROMOTE is limited to an exact dense-Qwen probe-path service building block; it does not claim a wired API/default or generalize to sampling/non-dense state. |

**Tests:** `swift test --package-path spike --enable-code-coverage --filter HarnessCoreTests`
passes 166 XCTest + 17 Swift Testing tests. Continuous policy line coverage: coordinator
95.09%, scheduler 91.31%, service metrics 94.76%; the new full shared CLI parser reports
38.75% because this change regression-tests its workload-identity value path rather than every
pre-existing accessor. On the bench Mac, Xcode `SpikeCoreTests` passes 48/48 and the Release
build succeeds. No MLX-coupled suite was run with `swift test`.

**Review:** independent evidence audit caught one warmup-inclusive A/B/A p50 in the compact
extract; it was recomputed from the three post-warmup runs, labeled
`measured-post-warmup`, and re-audited with no remaining discrepancy. Focused review's scope
split, stale-queue, and missing-affine-comparator findings were corrected; re-review found no
High or Medium issues. Repository-wide Markdown path/fragment validation and diff-aware secret
scanning pass.

**Committed compact evidence:**
[`continuous-batching-phase3-evidence-2026-07-14.jsonl`](superpowers/verdicts/continuous-batching-phase3-evidence-2026-07-14.jsonl),
15 valid JSONL records, SHA-256
`6c188275f2cad49010e0473356dacf3629d066b75b9670f3270a4771fcb97167`.

**Residual risk:** the checkpoint manifest identifies config/index plus shard names/sizes, not
full weight contents. The canonical frontier is one simultaneous-burst prompt shape on one
Apple box. Production routing, real network disconnect propagation, sampled generation,
additional dense families, and MoE/hybrid/recurrent/vision layouts remain separate work.

## Public evidence/community platform research acceptance pass — 2026-07-15

**Owner-policy update (2026-07-18):** this remains proof that the research and planning artifact
was completed. Its competitor-performance portion is private internal litmus only. The public
platform may publish fast-mlx evidence, not head-to-head or reproduced-external rows; the policy
override in the linked task supersedes the earlier publication language below.

**Story:** a prospective user, maintainer, or research agent can use one source-qualified review
and one agent-ready backlog artifact to understand fast-mlx's relative standing, build a public
proof/community loop, and avoid unsafe benchmark-publication or update mechanisms.

| Acceptance criterion | Verdict | Fresh evidence |
| --- | --- | --- |
| Review oMLX's website, benchmark, release, updater, and community loop from primary sources | **PASS** | The [dated review](reference/2026-07-15-omlx-public-ecosystem-review.md) labels material claims CONFIRMED, CONTRADICTED, INFERRED, or UNVERIFIED. Source probes confirmed automatic post-benchmark upload, the hardware-derived owner hash, updater's absent EdDSA archive check, current M5 Max/Qwen3-32B rows, and fast-mlx verdict values. All 21 external URLs returned HTTP 200. |
| State fast-mlx's relative performance without manufacturing a winner | **PASS** | The review places the local ~28 tok/s single-request rows in the same broad band as public oMLX 4K rows at 23.9–26.3 tok/s, explicitly rejects a winner claim, identifies mismatched prompt/runtime/cache/statistical boundaries, and defines a same-machine/common-boundary comparison lane. |
| Produce an agent-ready roadmap for site, benchmark automation, update, and engagement | **PASS** | The [task artifact](task-inbox/2026-07-15-public-evidence-community-platform.md) records the raw request, user/operator stories, 12 acceptance signals, failure/recovery paths, public evidence schema, four trust levels, P0/P1/P2 spawn packets, dependencies, ownership decisions, and next safe action. Sol review moved fleet/full-matrix automation behind a bounded first proof and added the minimal feedback loop, compatibility manifest, and reproducible agent-workload packs. |
| Preserve privacy, supply-chain, and performance-evidence boundaries | **PASS** | Community submission is explicit opt-in with payload preview/deletion and no stable hardware ID; updater requires signed/notarized/Team-ID-checked failure-safe delivery; lossy rows retain teacher-forced/task loss plus coherence floor; dirty/incomplete/incomparable evidence fails closed; lab, reproduced competitor, community, and external rows stay visibly separate. |
| Keep durable docs navigable, review-clean, and secret-free | **PASS** | All relative links in the four touched Markdown files resolve; `git diff --check` passes; all 21 external URLs return 200; working-tree `gitleaks dir --redact .` scans 2.26 MB with no leaks. Focused review's RAM-filter, three-engine counterbalancing, and compatibility-manifest dependency findings were corrected; Sol's five prioritization/engagement findings were incorporated; final re-review found no issues. |

**Not run:** Swift/Xcode tests or a new Apple-Silicon benchmark. This is a documentation/research
cycle and the current oMLX rows are intentionally EXTERNAL REFERENCE evidence; running unlike
workloads would not satisfy the same-box comparison gate. The domain, host, public/commercial
policy, privacy owner, signing authority, application target, and second-node availability remain
explicit implementation decisions.

**Overall verdict:** ALL PASS for the review and durable-roadmap acceptance criteria. Website,
ingest, competitor runner, community upload, and updater implementation remain open work.

## KVarN Phase 0 evidence-contract acceptance pass — 2026-07-14

**Story:** a bench operator can record exploratory KV-quality rows, but cannot promote a dial cell
unless the row proves same weights, context-locked 24K quality, canonical format geometry, honest
actual storage, clean provenance, and durable append-only persistence.

| Acceptance criterion | Verdict | Fresh evidence |
| --- | --- | --- |
| Typed identity and comparison contract | **PASS** | `KLPayload` and `KVFrontierEvidence` bind matrix/cell identity, candidate/reference config and checkpoint-manifest hashes, same-weights fp16-KV baseline, canonical tier geometry, corpus identity, and the outer result provenance. Historical optional fields still decode. |
| Teacher-forced quality cannot be mislabeled | **PASS** | New rows require finite context-locked top-1 evidence plus short and long cohorts. Promotion requires a distribution actually scored at context depth >=24,000; a shallow custom `long-context` tag fails closed. The available perplexity, long-tail, and top-1 garbage-floor predicates are enforced before persistence. |
| Storage claims fail closed | **PASS** | The validator reconstructs the predicted allocation from `KVStorageFormat`, rejects fabricated or mismatched breakdowns, and requires predicted bytes to equal actual bytes for promotion. Until Phase 2 supplies real arrays, promotion exits 1 with `missingFormat` and writes no artifact. |
| Evidence persistence is durable under contention | **PASS** | The writer rejects corrupt/partial prior rows, opens without following symlinks, holds an exclusive file lock across validation and `O_APPEND`, and synchronizes before success. A 128-writer regression preserves 128 unique, decodable rows. |
| Clean-SHA Apple proof | **PASS** | Commit `e1a3cff6cc95ffcd980cc8e05709fdb0ab7edc39` was stamped by the sync script. Xcode passes 48/48 `SpikeCoreTests`; the Release harness builds. Qwen3-4B same-weights fp16-KV scores the five-entry v2 corpus through a 24,150-token context and writes one schema-valid exploratory row; the corresponding promotion run refuses to write. |

**Pure verification:** focused evidence/provenance/quality tests 28/28; full HarnessCore run 211
XCTest + 17 Swift Testing, zero failures. Line coverage: `KVFrontierEvidence.swift` 93.63%,
`Provenance.swift` 85.12%, `QualityMetric.swift` 87.50%. Final focused re-review: no issues.
Diff check, banned-concurrency scan, and staged gitleaks scan pass.

**Bench artifact hashes:** Xcode test log
`e1fb2f34004490f76d3e973b443e6ef1a79db08250b970ccfdbcb1f3c01a96e9`; Release build log
`34a9ec8a7f0a51edb548d43c993a99e3edb4d7cbe7d4b100ab26945d5471634c`; provenance
`7f726e19b5262e105033ff765886095b0c8fc78f08832edc5eda6bdfa90e0396`; exploratory JSONL
`edbce7dfc17a3b7d14f0049a29daa6dd41d80067c83e4b988ca63dbabd481716`; exploratory log
`11744e02b897fd92d66c36b8a9035c6d775ef0ddb8e9f5982a8dea60bcd31435`; promotion-refusal log
`bcf8c4290d2df7f9ea9b3d749c4421e0b56c7956f0f21a853cc320e9516672d4`.

**Boundary:** this closes only the Phase 0 evidence gate. It is not an affine/KVarN performance
or capacity result. Phase 2 must reconcile native MLX array bytes—including compile-control state
and materialization workspace—before any dial cell can become promotion-eligible.

## KVarN Phase 2 affine-control acceptance pass — 2026-07-14

**Story:** a bench operator can select every declared affine KV control and receive native-array,
teacher-forced evidence whose storage and per-entry sample claims fail closed, without treating a
small plumbing smoke as a quality verdict.

| Acceptance criterion | Verdict | Fresh evidence |
| --- | --- | --- |
| Runnable controls fail closed | **PASS** | `fp16`, K4V2/K8V2 at groups 64/128, and K4V4-g128 parse independently and report the selected geometry. Unknown or incompatible tiers fail before scoring; lossy KV plus PLD remains rejected. Qwen3-4B smokes engaged all five affine cells. |
| Native storage is honest | **PASS** | `AffineKVCache` stores packed K/V plus native scale/bias arrays and reports payload, metadata, workspace, and control bytes from real MLX arrays. Clean `4c1bfe1` K8V2-g128 predicts and observes exactly 1,112,832,000 payload + 55,641,600 metadata + 98,918,400 workspace = 1,267,392,000 bytes, plus 144 control bytes. |
| Cache lifecycle preserves the MLX contract | **PASS** | Xcode tests cover materialize/read, growth, reset, truncation, mask behavior, two-bit packing, compile integration, and accounting. MLX state remains actor-owned; no banned concurrency escape was introduced. Engine SHA `d4bda35dbfa78c079fa970bbb8aaf1526a0022f7` passes 56/56 `SpikeCoreTests` through Xcode; the extracted result summary records 56 passed, 0 failed, 0 skipped. |
| Teacher-forced evidence is entry-complete | **PASS** | Each short/long aggregate is now backed by unique real corpus entry IDs and scored-position counts whose overflow-safe sums must match. Promotion requires every short entry >=24 and every long entry >=128. A clean undersampled promotion run exits 1 at the first 2/24 short entry and creates no JSONL. |
| Current clean-SHA integration is reproducible | **PASS** | Synced clean `4c1bfe1f6947b184a6fc8834e0787bbfd4e3354a` builds Release and produces a schema-valid exploratory K8V2-g128 row across all three short and two long entries. Full pure verification passes 217 XCTest + 17 Swift Testing tests; focused re-review reports no issue. |

**Artifact hashes:** current clean Release log
`a9c56edf34592939da2f01fd7d5c7670064419f7d1e7b68b6eea9c73b49f1237`; final-engine Xcode
result summary `ccc7795197c52ce98a035c30326388ebaba6eb363ff2beed695b904b72bc455a`;
launch/build log `bd36cbcd25d7f67560178b128f4ead7124fea622bbb4f4dbf9444e00353931b2`; exploratory JSONL
`ecd8760845b9e727df64731f400b04544f4943425efeeae1fd1dae3f5de38a3e`; promotion-refusal log
`9188293d41b06d39b0a971db7c828d7d726e0d7f14de47ee1af1380bb6fa667b`; refusal status artifact
`4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`.

**Boundary:** this closes affine runtime/control plumbing and its evidence gate—not the KVarN
implementation, the Qwen3-32B 24/128-position matrix, monotonicity, capacity benefit, or a dial
verdict. The exploratory affine predicate counts (4/8, 7/8, 7/8, 7/8, and 3/8 across the five
cells) are deliberately non-authoritative until the frozen full matrix runs.

## KVarN / asymmetric KV frontier final acceptance pass — 2026-07-18

**Story:** an Apple-Silicon operator constrained by context or concurrency can select a measured
KV-capacity tier with its real quality and runtime costs visible, while cells below the coherence
floor remain unavailable and exactness contracts elsewhere in the engine remain unchanged.

| Acceptance criterion | Verdict | Fresh evidence |
| --- | --- | --- |
| Same-weights reference fidelity | **PASS** | The fp16 reference and all retained candidates bind the Qwen3-32B-4bit checkpoint, model/config identities, task corpus v2, tokenizer/prompt layout, and per-cell task/KL artifacts. The compact packet records every KL evidence/sidecar and task raw/summary SHA-256. |
| Honest size and capacity accounting | **PASS** | fp16 is explicitly exact-accounted as 6,330,777,600 bytes at 24,150 tokens. Lossy rows use directly measured cache-array bytes plus workspace. KVTuner/KVarN's separate 256 control bytes are preserved distinctly; derived capacity ranges from 2.7826x to 4.5110x. |
| Lossy frontier fails closed | **PASS** | `affine-k4v2-g128` and `affine-k4v4-g128` are retained as hard-floor failures and rejected from the dial. Unknown tiers, malformed schedules, missing evidence, non-finite metrics, and lossy KV plus PLD remain rejected. |
| Runnable selection and frozen-policy controls | **PASS** | Uniform affine cells, frozen KVTuner, and KVarN i8/i16 engaged under their declared identities. KVTuner authenticates its frozen bundle/schedule; KVarN reports the explicit uncompiled correctness path instead of claiming compiled execution. |
| Context-locked quality | **PASS** | Eligible cells completed teacher-forced KL, pooled perplexity delta, top-1 agreement, and 24K tail-p95 against the locked reference context. Free-running output was used only for task/coherence adjudication. |
| Task/coherence floor and user freedom | **PASS** | The admitted fp16 reference scored 16/10/20/20 with 20/20 structured syntax. Retained lossy cells stayed above the predeclared per-domain/syntax floor; aggressive cells that collapsed structured behavior were rejected rather than exposed as an expert setting. |
| Same-workload Apple runtime evidence | **PASS** | Clean runtime SHA `f88d26eeb3793cc1d5d8f2118043c190977ee6e0` completed all seven cells with one authenticated JSONL row per cell and absent watchdog. fp16 measured 28.46 decode / 376.33 prefill tok/s; no lossy cell was faster, so promotion is capacity-only where applicable. |
| Flywheel closure and claim boundary | **PASS** | The dated verdict promotes fp16 Transparent, affine K4V2-g64 Balanced capacity, frozen KVTuner Max-fit capacity, and KVarN i8 capacity-only Max-fit/fused candidate; it shelves or rejects the rest and links a reader-facing content piece. The result is Qwen3-32B-4bit/model-specific; a materially different popular family is required before broad support claims. |

**Fresh verification:** `swift test --package-path spike --filter HarnessCoreTests` passes 389
XCTest + 17 Swift Testing tests. On the clean synced bench tree, Xcode with
`-skipPackagePluginValidation` passes 30/30 `FastMLXHarnessTests` and 92/92 `SpikeCoreTests`; the
Release `fastmlx-harness` build succeeds. MLX-coupled targets were not run through SwiftPM.

**Evidence authentication:** compact JSONL
[`kvarn-kv-frontier-evidence-2026-07-18.jsonl`](superpowers/verdicts/kvarn-kv-frontier-evidence-2026-07-18.jsonl)
parses as 12 records. The task/KL source/harness is clean `d9071a93955be2e148dc381638d4a71a8286c59e`;
the KL manifest is `3be8693aa169e2e5b4c2692f7fdc115783271fb735d27e50b0d5c3cb798990df`;
the same-SHA KVarN memory gate is
`7b21459cf1afc1a038b87c017c1fdacd18644b08b4b63a07318d74faba2dcfd2`.
The original KL wrapper status remains `ABORTED` after a post-measurement pipefail. Reviewed
recovery finalization authenticated all seven immutable rows/sidecars twice and records finalizer
SHA-256 `096c38dfcb648bf7cf3870c12f24b9419bff398668a0ef13402c1009f0dcdeb4`, recovery
SHA-256 `9548bd792bcd836a92b9612a70026ad54c77adc16395df1e0b6a1fed3a8d7077`, and recovered
completion SHA-256 `732d4e3486a8502dd09ec44628875d1e5435fbd3648ec6c8024b8f01047de4dc`.

**Review and residual risk:** focused review found and corrected one 256-byte accounting wording
error and required the recovery chain above to be explicit. Final review, link/JSON validation,
diff checks, banned-concurrency scan, and staged secret scan pass. Coherent closure `687b09f` was
merged `--no-ff` to `main` as `f435312`. The
largest remaining technical risk is that storage reduction has not become a compiled speed path;
fused compressed-domain attention and a second model family are the next gates.

## Fused compressed-attention Phase 0 implementation acceptance pass — 2026-07-18

**Story:** a bench operator can compare independent K/V affine packed attention with fp16, the
pinned symmetric Swift-LM helper, and materialize-then-attend using one authenticated synthetic
geometry contract, without allowing that kernel probe to masquerade as loaded-model or dial
promotion evidence.

| Acceptance criterion | Verdict | Fresh evidence |
| --- | --- | --- |
| Synthetic qualification identity fails closed | **PASS** | The CLI and schema use `qualification-evidence`, reject the legacy promotion flag, freeze 8K/32K and the exact 130,944+128 near-128K window, authenticate context+output against checkpoint config, and stamp schema v2 as `checkpoint-authenticated-synthetic-geometry`. |
| Independent K/V packed path is real and actor-confined | **PASS** | `split-affine-quantized-mm` executes separate MLX affine `quantizedMM` operations for K and V with independent bits/groups, explicit precise softmax, GQA reshaping, and mask handling. It does not reconstruct dense K/V and is explicitly not labeled fused SDPA. Only scalar/value evidence leaves the owning actor. |
| Timing and memory receipts isolate the attention path | **PASS** | Source generation, hashing, quantization, and host-detached prepared cache state occur before the peak reset. Timing/evaluation and the immediate post-attention snapshot exclude oracle work. Workspace totals are rederived from raw MLX receipts; impossible active growth and nonzero peaks below the prepared active baseline fail closed. |
| Changed contracts and MLX behavior are freshly verified | **PASS** | Off-box `HarnessCoreTests` passes 429 XCTest plus 17 Swift Testing tests. After sync to the bench Mac, Xcode with `-skipPackagePluginValidation` passes 47/47 `FastMLXHarnessTests` and 92/92 `SpikeCoreTests`; the Release package build succeeds. MLX-coupled targets were not run through SwiftPM. |
| Focused review and claim boundary | **PASS** | Review findings for arbitrary near-128K aliases, contradictory raw-memory receipts, cache-policy naming, and help wording were fixed test-first. Final focused re-review reports no remaining High/Medium findings. The plan and handoff state that the probes stream checkpoint bytes for content authentication but do not instantiate or execute those weights as MLX model tensors; they cannot support loaded-model, family, or dial claims. |
| Clean-SHA synthetic matrix | **PASS** | Clean `07219679280abd2f7cefbeef86b71bbec018a1c2` produced five Qwen 8K, five Qwen 32K, and two Llama near-128K probe-evidence-schema-v2 artifacts: 72 measured rows whose canonical artifact IDs were independently rederived. The compact index uses schema v1 and carries those source artifact counts explicitly. Qwen's 130,944+128 request failed before MLX allocation at the authenticated 40,960-token limit. Compact evidence SHA-256 `bb8387e2f4b6d1ac4862f09816a5517c2a34b4f7e7a9383f300d3002c13a3cf5` links the 77-file bench manifest SHA-256 `6f999443363267a8a6429140766027fa0f1f14df4b0f400a257171d2230d2d89`. |
| Runtime integration and promotion gate | **PENDING** | The portable router, scalar runtime, hostile batch-compaction case, loaded-model Qwen/Llama frontiers, teacher-forced loss, task floor, verdict, and content remain later phases. No PROMOTE or SHELVE conclusion is made by this packet. |

**Overall verdict:** Phase 0 passes and selects the portable independent-K/V quantized-matmul
route. All loaded-model, prefill, lifecycle, batching, quality, product, and promotion claims remain
open.

## Fused compressed-attention Phases 1-3 implementation acceptance pass — 2026-07-19

**Story:** an Apple-Silicon long-context operator can enter an experimental packed-affine or
authenticated KVTuner runtime without full-cache materialization, and a continuous-batch survivor
cannot be corrupted when the longest zero-padding-boundary row leaves. This implementation proof
must not be mistaken for a loaded-model speed or dial-promotion result.

| Acceptance criterion | Verdict | Fresh evidence |
| --- | --- | --- |
| Portable model-generic router | **PASS** | Clean `faa385f` vendors the pinned Swift-LM source and routes the shared attention helper by cache capability rather than model name. Independent K/V geometry, explicit causal masks, GQA, unsupported window/sink paths, and non-finite refusal are regression-covered. |
| Actor-confined scalar runtime and honest telemetry | **PASS** | Clean `e2d719e` integrates affine K4V2 and frozen KVTuner through the packed route while preserving reset/growth/rollback, exact engagement, persistent packed bytes, materialization bytes, and attention workspace. MLX state remains actor-confined and lossy KV plus PLD remains rejected. |
| Hostile continuous-batch transition | **PASS** | Clean `5e6abb6ebf13ea8641b26638278680e99884adea` tracks a physical written end independently from logical row offsets. Tests merge unequal rows, remove the longest/zero-padding-boundary row, append again, and prove physical end, fixed mask width, survivor packed bytes/logits, growth, extraction, and dense-control parity. |
| Exact KVTuner source and evidence binding | **PASS** | The same clean commit carries exact config and checkpoint-content SHA-256 through manifest, sensitivity, candidate evaluation, search, schedule, runtime selection, task, and KL evidence. Contentless or substituted fp16 references and same-manifest/different-content checkpoints fail closed. Historical schema-v1 qualification artifacts remain readable only where explicitly supported and cannot authorize the new content-bound path. |
| Clean-SHA build and test proof | **PASS** | Off-box `HarnessCoreTests` passes 457 XCTest + 17 Swift Testing tests with zero failures. On the synced clean SHA, Xcode with `-skipPackagePluginValidation` passes 65/65 `FastMLXHarnessTests` and 127/127 `SpikeCoreTests`; the Release `fastmlx-harness` build succeeds. MLX-coupled targets were not run through SwiftPM. Pure log SHA-256 `15faac23858b973a95f0b51ab0ab55feaaee63bdb71c4f2f98f94efdfceedc08`; Xcode log `e7d1641a42e1ff85b1b4ba0a3787b271fb42ca7b317456e61ca18d0c1c952b9d`; Release log `86c3cd4dc482cd0e19739e6518fa366747431e8d1b6605d55a4d31d4177d4153`. |
| Review and repository safety | **PASS** | Two focused reviews report no remaining High/Medium findings after exact-reference fixes. `git diff --check`, banned isolation/machine-path/key-marker scans, and commit-scoped gitleaks pass; the commit retains the required co-author trailer. |
| Loaded-model speed, quality, and cross-family promotion | **PENDING** | KVarN i8 still materializes and needs its own direct packed-attention proof. Exact-source KVTuner requalification is in progress. Qwen 8K/32K plus 128K refusal, Llama 8K/32K/near-128K, TTFT/TPOT, task floor, teacher-forced loss, capacity, and same-workload speed remain unmeasured for this implementation. |

**2026-07-21 supersession:** exact-source KVTuner requalification subsequently completed, and
the loaded-Qwen direct-KVarN smoke completed 3/3 with all 64 layers engaged and zero
materialization. Later loaded diagnostics established that direct KVarN has no credible route to
the frozen speed gate in this cycle, so its speed role is now SHELVED while its capacity-only role
remains. The table above remains the immutable dated 2026-07-19 acceptance packet rather than being
rewritten after the fact.

**Fresh requalification boundary:** schema-v2 KVTuner manifest SHA-256
`e8b069cafb697a332325def638effdaf8f56b9bc62d2139b2c7dc2aba1719a5f` binds checkpoint-content
SHA-256 `636f358d4f51c9394400fa46ef684b918e45c14d369d95df0399c80abc8a09d9`.
Schema-v3 g128 sensitivity SHA-256
`9426976a9215ce5276ac80ea165de3b084b239652ce138e670163bcbdf41d7fc` authenticates 64 layers and
3,840 samples at the same clean SHA. At this dated packet boundary, the canonical 64 candidates,
search, and a new qualification bundle remained pending. No old bundle was rewritten, and this
implementation packet itself issued no PROMOTE/SHELVE verdict.

**Coverage boundary:** the full acceptance suites above were rerun without coverage instrumentation.
The prior Phase 0 coverage packet remains historical evidence; this packet does not claim a new
line-coverage percentage.

## Loaded qualification post-warmup thermal admission — 2026-07-21

**Story:** a bench operator can compare loaded-model cells only when every retained row begins
from the same manifest-bound unthrottled host cohort, without discarding the warmup that caused the
thermal transition or weakening per-run and cross-row equality.

| Acceptance criterion | Verdict | Fresh evidence |
| --- | --- | --- |
| Failed thermal boundaries remain immutable | **PASS** | Qwen v7 is terminal `INVALID_BLOCK_ENVIRONMENT`; v8 materialize and v9 direct are terminal `PRECONDITION_NOT_FAIR`. V10 is terminal `FAILED` after 5/49 authenticated rows at block 0, position 5, `affine-k4v2-g64-materialize`, when point-in-time nominal admission did not survive retained measurement. V10 manifest SHA-256 is `353820692294f408ddae91e7f3b7d1522604ea77e40646c6a45d812af44434e1`, status SHA-256 is `03cbb6c52e522ce55e217e3f9d66dbd66d21ef6bf85024016a3b7f3080c7944f`, and rejected partial receipt-set SHA-256 is `2c8c9b1e2b7dc24f70bdb9935f9bcbdb9ad73efd06a249809dbb5906fa6f2889`. None may be resumed, overwritten, or promoted. |
| Warmup and continuous admission are explicit | **PASS** | Clean `d4102e6a3029b161d99ee27aceabbad8d5696fb5` retains the authenticated warmup and requires a positive manifest-bound continuous nominal/AC/non-low-power dwell before retained work. Safe non-target state resets the dwell; serious, critical, unknown, power drift, malformed timestamps, and timeout fail closed. Retained before/after equality is unchanged. |
| Policy and timing are authenticated end to end | **PASS** | Fresh runner manifest schema v3 requires target, timeout, poll, and positive stability duration. Evidence schema v4 requires at least two strictly monotonic target-state observations, binds the first observation after warmup, proves the interval meets the frozen dwell, and requires the last observation to equal the admission snapshot. Evidence v3 remains readable as instantaneous-admission compatibility and v2 remains legacy-readable only. CLI, row validation, progress, receipts, block completion, and matrix completion bind the same policy. |
| Failure evidence is explicit and non-promotable | **PASS** | Runner-failure schema v1 binds exact source, binary, runner, manifest, model, row, policy, log and optional-evidence hashes, and embeds the validated retained before/after diagnostic with `promotable:false`. It is never appended to the promotion receipt set. Parent-owned artifacts use randomized atomic publication and authenticated directory/file identities; adversarial child symlink, unlink, duplicate-diagnostic, future-run, aggregate, PID, and output-root replacement paths fail closed. |
| Clean-SHA automated proof | **PASS** | Off-box `swift test --package-path spike --filter HarnessCoreTests` passes 519 XCTest plus 17 Swift Testing tests; pure log SHA-256 is `d9712cf2183166f392a7bfda6c669f4d09b3db8a5103166b0acfe3bc96ae7139`. On synced clean `d4102e6a3029b161d99ee27aceabbad8d5696fb5`, Xcode with `-skipPackagePluginValidation` passes 95/95 `FastMLXHarnessTests` and 155/155 `SpikeCoreTests`; the Release build succeeds. Binary SHA-256 is `cfe029ad2138013a5904e6afd2475a881081a37bcf89db25bfbb91abf8484397`, runner SHA-256 is `e2f6e55bb0aeae6b1ec585f6d0d3c85b13c00879e13ed2a0fa3826ee074f8c0f`, Xcode log SHA-256 is `090534e0ae5f3c9f36a6b4917079defb757f0d6cc8bfba270a659d88578b0376`, and Release log SHA-256 is `fd1d87c09187f2c3187a5bc02d91e82997d1b7f9e9da573ad9d027f8e28d1876`. |
| Review and repository safety | **PASS** | Two independent final reviews report no issues after output-root and parent-artifact hardening. Shell syntax, ShellCheck, diff checks, the 36-test loaded-runner suite, and staged gitleaks pass. The code commit carries the required co-author trailer. |
| Continuous-dwell loaded preflight | **FAIL (preserved)** | V11 is terminal after 2/9 authenticated rows and zero completed blocks. The fp16 row carried 59 nominal observations spanning 60.0088 seconds and retained nominal/nominal; KVarN materialize carried 59 observations spanning 60.0282 seconds and retained nominal/nominal. The direct KVarN row reacquired nominal after its fair warmup but changed nominal -> fair during retained work, emitted no evidence, and produced non-promotable failure SHA-256 `02893f30229f79861f95e8d037536ae6d0bca7855539bbc88ab2f82dd788293b`. Partial receipt-list SHA-256 `ae9dcb7e0ef082aff6cd6509c13a9e41f85fe8937d0a203867d762559c259a56` contains only the two promotion receipts and excludes the failure; partial block-receipt SHA-256 is `2780c3a9b4880236082feb3dcca26613f69220c854d68dc0db39f67ea0e7ac90`; complete runner-log SHA-256 is `02ade4793feafc228a22c5a99afc8fe8ce4ab14252abaece25805dc8eb0bc291`. |
| Loaded direct-KVarN speed disposition | **SHELVE SPEED / RETAIN CAPACITY** | The bounded clean-`d4102e6` Metal diagnostic is terminal non-promotable failure at `/Users/llmbench/perf-work/results/fused-compressed-kv-profile-d4102e6/qwen-8k-kvarn-direct-metal-v1`. High Power/AC/Foundation nominal readiness passed. `xctrace` reached 240 seconds but exceeded the save watchdog after producing a 16,441,155,536-byte raw Apple Trace File; peak recorder RSS was 107,315,600 KiB. Raw trace SHA-256 is `12073b786fb06d5569269500129bee3f9b1926319f9a34a9de97c5cdf24853ea`; status SHA-256 is `982b2e8659cd53abb2c403c15e949cd6268fffa94f48c06cb85837141c2958b2`; `xctrace export --toc` fails `Document Missing Template Error`. V11's hash-bound direct diagnostic (63.26 prefill / 7.18 decode tok/s) needs about 8.02x prefill and 3.43x decode to clear the unchanged fp16 envelope. That rules out one credible recovery; no promotion result is inferred from the partial row. KVarN remains its already-qualified capacity-only Max-fit cell. |
| Remaining Qwen 8K speed scope | **COMPLETE / NEGATIVE** | Clean `a2af840d6f02c3a9097e4df0372e969d18bd7bc8` completed 25/25 authenticated rows in five cyclic blocks at `/Users/llmbench/perf-work/results/fused-compressed-kv-qwen3-32b-loaded-a2af840/qwen-8k-v12-five-cell`. Manifest SHA-256 is `551504e541b8e9a21786be536abc24b31ea68dea1dd71b5b0d0819a04ef92591`; completion, receipt-set, and runner-log SHA-256 values are `762c276456f33bab6927af6aa38297a9949b381de868b9b70f43b218b18e8b1d`, `bd716829efa223ffb46c25f304df81c40cbac07e29890b3f1590afd50d85bfa1`, and `71344817efd30bcf099f62cba814c7937599ee8c9b126aafdc293764761640c6`. Exact-source bundle `8217bc37cd3d100c493fb6f76b15d7abe5c168baa2bc2fb6561ac9f12c9dc125` and schedule `d76d2534939677711cf27123eea0cf15ad3512bd5fd04965abe85e36232f9260` authenticated; historical bundle `dac242bca836b5819ea9e4e2a7de14d06707124afc4c2319a8fc88ca96202ada` did not authorize this run. The independently reviewed authenticator and fail-closed reducer have SHA-256 `25e11db330b90e6196dd520816bed0fb597df101396c6e4117730a5663012e22` and `65877508be97ff20acbc19a25a25cee97c6306ccffeedc9d3d09bd462b0dd777`. Median decode/prefill tok/s were fp16 23.32/531.19, affine direct 24.19/333.29, and KVTuner direct 23.19/333.34. Direct affine and KVTuner beat their same-storage materialize controls in every block, but neither beat fp16 decode by 5% in every block and both exceeded the 5% prefill-regression limit in every block. Both all-block gates are `false`; preserve as negative/dominated evidence and promote no 8K speed tier. |
| Loaded Qwen 32K speed scope | **FAILED CLOSED / HARDWARE-UNAVAILABLE** | The first clean-source 32K matrix is terminal at `/Users/llmbench/perf-work/results/fused-compressed-kv-qwen3-32b-loaded-a2af840/qwen-32k-v13-five-cell`, manifest SHA-256 `07c6bbb16a6b1e2a636e39692f84fbf4ba3621638f7d25f09a1582210e09930c`. The exact 32,628-token fp16 control warmed at 255.22 prefill tok/s, completed 60.051 seconds of continuous nominal admission, then changed nominal -> fair during its 127.999-second retained prefill. Runner-failure, bench-log, and runner-log SHA-256 values are `5c6798d5f2299c145672d0ac8b053d9cc7db78dff267ad4a86738e2f33bce948`, `fb0aac111797ea05122ac26e6b2464f2e63ee0f091bd2ef4fef55c1979f2d9b4`, and `2db7afb333e3a1d54081928454c6ea8e6979901c2acb87886e80c6f3f79ba534`. Status/progress SHA-256 values are `03cbb6c52e522ce55e217e3f9d66dbd66d21ef6bf85024016a3b7f3080c7944f` and `254346128f679477511dc341f37679e809216b0ed16f8e321589f03f2fbed2fe`. The audited boundary has zero evidence, zero receipts, no live process/lock/watchdog, and no overwrite. Do not infer speed from the diagnostic timings or retry the full matrix: this bench cannot currently produce promotable 32K speed evidence under the frozen unthrottled contract. |
| Loaded Qwen near-128K context refusal | **PASS / NON-PROMOTABLE** | Preserve v15 as a fail-closed launcher-contract boundary: its mismatched affine `cell-id` stopped before tokenization, emitted no evidence, and bound failure/log/time SHA-256 values `9d8d1df8cf2749fca445367fa3950fd6b80ed8d77079d3c0e78b76bf8da45451`, `5aac09ba9d95e38088a5cb6ef14d82616e0bc241ed882ddc97f78700557af934`, and `4a7662abf9776ff93d33370c64b372b9936abbb94bba2595d67c4fd4c21d7276`. Fresh v16 corrected only the exact tier identity. Manifest/launcher SHA-256 values `ecbfc323b1cd13537968f854fa18c9cff02f90f24af53623500ffe53a500587d` and `a7a844a77b29493cae452d82af53adf7e59752a9d62b929966a5a21b945e2d27` produced `EXPECTED_REFUSAL` status SHA-256 `847f316fd0a906cfb8bd637cf86355670d68fa63e82d34c864f7566d48ca442f`. The compressed preflight authenticated 130,911 prompt tokens and a 128-token output budget, then rejected the 131,039-token request against the checkpoint-bound 40,960-token limit before driver/model/KV allocation. Exit was 1; maximum RSS was 18,584,862,720 bytes; there was no warmup/retained forward, evidence, live process, or lock. Refusal/time log SHA-256 values are `71167f7a1d97a0f08cfb3b210c306ca0e127c382753901c231b6534e751d84cb` and `16e9da974d49cbf07035ae4aa0405083628cf9993c801ba619ea8e70f49afd90`. |

**Verification artifact note:** the Xcode tests and Release build succeeded. Their first evidence
wrapper assumed a SwiftPM product path after the Xcode build; a second receipt command then left a
zero-byte placeholder when its shell envelope failed. Both failures are preserved. The final
`COMPLETE_WITH_ARTIFACT_FINALIZATION_RECOVERY` record only hashes the already-built Xcode product
and reruns no test or build; recovery-record SHA-256 is
`696c9793ad7848fb70ef7bc70755c8ff2533121d83a236232b5f5ba2e090f37d`.
The earlier code commit's intended co-author line was written with literal escaped separators, so
Git does not parse it as a trailer. That historical evidence-bound SHA remains unchanged; later
recovery commits carry the correctly parsed trailer and record the exception rather than rewriting
old evidence.

**Overall verdict:** continuous post-warmup nominal admission passes implementation, review, pure
verification, MLX-coupled Xcode verification, and Release build. V11 proved the contract fails
closed on an actual retained transition. The follow-up trace failed safely and produced no usable
kernel attribution. KVarN's speed role is now SHELVED while its qualified capacity-only role is
retained. Loaded Qwen 8K speed promotion is closed negative: the reviewed five-cell
affine/KVTuner matrix completed cleanly, but both direct candidates failed the unchanged all-block
speed gate. The first Qwen 32K fp16 control then failed closed on retained nominal -> fair drift,
so 32K speed evidence is hardware-unavailable under the frozen unthrottled contract. The
checkpoint-bound near-128K refusal is authenticated; only separate KVarN capacity-only context
remained before the Qwen model-scoped frontier could close at this 2026-07-21 boundary.

## Qwen3-32B 32K KVarN capacity-only closure — 2026-07-22

**Story:** a long-context Apple-Silicon operator can inspect the actual memory and runtime behavior
of the already-qualified KVarN i8 capacity tier at 32K without that evidence entering a speed
aggregate, changing a default, or weakening the preserved speed and model-limit failures.

| Acceptance criterion | Verdict | Fresh evidence |
| --- | --- | --- |
| Frozen non-promotable contract | **PASS** | Clean `8b454754ed9e631b05ce1164b9c30853b4e416f8` hard-codes `bench-capacity` as KVarN-direct only, exactly one dropped warmup plus one retained run, `promotable:false`, and `speedAggregation:"forbidden"`. Separate clean-SHA verification passed 521 HarnessCore XCTest plus 17 Swift Testing tests off-box (log SHA-256 `8b27f1d43ec68e6a5177ab4ca4f70b2c916c2bcb7fc17630bd54b5d18462989a`), 100/100 `FastMLXHarnessTests` and 155/155 `SpikeCoreTests` (Xcode log SHA-256 `356525c32f12562a2ef873dcfb7c5e9d06c5b45893bc901ffd615641fcb0e0f2`), and Release build (log SHA-256 `7908687d20f05d69040f263af2e6e72e6ac03aace03657604d404a1f767ee174`). Clean binary SHA-256 is `5d72ea1b6ab98749421c059984fe1ff69294d2971d35ba6557bb44c7033675a6`; source-stamp SHA-256 is `179963aada3ebfbe145893c04afb16a2862db96f16558a7d794c0d88e8a620fc`. |
| Fresh terminal boundary | **PASS** | The one-shot output `/Users/llmbench/perf-work/results/fused-compressed-kv-qwen3-32b-loaded-8b45475/qwen-32k-kvarn-capacity-v20` completed 1/1 after 6,359 seconds with no watchdog or failure artifact. Manifest/launcher SHA-256 values are `f81a5d95dc2554d1ba749965bf49703fbe546ee14700147d4064d7cc9a980da2` and `851a99f525be4c2d108d880dc636e3efad199cd81cf89f0d795b9c3b2aeda59c`. Final progress SHA-256 `830874b2b57d757a06e6a96d0e851a2d47f4301d793f28f8c2a9b1227fbaf850` records `COMPLETE`, 1/1, zero current RSS, and watchdog false; a separate read-only final monitor observed every exact PID exited and the identity-bound lock absent while Foundation remained low-power false/thermal nominal. V17, v18, and uninvoked v19 remain separate preserved boundaries. |
| Exact provenance, workload, and limits | **PASS** | Evidence binds Qwen3-32B-4bit config/checkpoint/tokenizer identities, nonce `qwen3-32b-32k-kvarn-i8-capacity-v20-run`, exact per-run prompt hashes, 32,628 prompt plus 128 generated tokens, 96 GiB MLX memory, explicit 8 GiB `Memory.cacheLimit`, 115 GiB wired memory, and `split-kvarn-quantized-mm`. Both High Power profiles, 140 W AC, `iogpu.wired_limit_mb=117760`, and Foundation nominal/non-low-power readiness were recorded before launch. |
| KVarN direct engagement and storage | **PASS** | All 64 layers engaged compiled split KVarN attention with mixed bf16/fp32 ingress normalization, bf16 native storage, zero materialization, 32,756 cached tokens, 33,024-token capacity, 32,512 compressed tokens, and 254 completed tiles. Payload/metadata/control bytes are 1,616,904,192 / 202,113,024 / 256; attention/total workspace bytes are 8,659,140,608 / 8,661,237,760. |
| Memory and environment evidence | **PASS** | MLX active/cache/peak maxima are 20,343,562,498 / 6,870,047,163 / 59,560,463,681 bytes. Sampled physical footprint and process maximum RSS are 64,618,487,912 and 55,875,010,560 bytes. The single warmup and retained snapshots were AC, non-low-power, nominal -> nominal with strictly increasing monotonic timestamps and finite metrics. |
| Runtime is context, not promotion | **PASS / NON-PROMOTABLE** | Retained prefill/decode were 10.38 / 4.934 tok/s and TTFT was 3,144,859 ms. The evidence row preserves `promotable:false`; the manifest, launch receipt, and completion additionally preserve `speedAggregation:"forbidden"`. These values do not revise the shelved KVarN speed role or enter Transparent/Balanced/Max-fit speed aggregation. |
| Independent artifact authentication | **PASS** | Direct `validate-bench-capacity` returned `bench capacity evidence: VALID`; focused-review-clean read-only authenticator SHA-256 `08221e1629da4e2b5abe010a8a422abf5844b61eabafb743ebe5489d1ee4de1f` returned PASS twice. Evidence/completion/launch-receipt/validator-log SHA-256 values are `99932d17ec7014b80eb8f50a0f1393e565444cad4dc6e7629c4888c467cad59c`, `d81308c2e233679e2693564e59e23d44757a7be5ca35b6edd26d23d117964b88`, `b47aeb543d20b42d12e1737e9d24108fa69e757895bda64487f2693c2b9331bb`, and `4c1ac0ecc101469994d690fe5c9740773c906d7ade562c704d2fe30a3e6a24c9`. Local rehashing of the copied immutable packet matched every published sidecar. The two PASS objects and normalized final PID/lock/watchdog observation are durable in [`qwen3-32b-kvarn-capacity-verification-2026-07-22.json`](superpowers/verdicts/qwen3-32b-kvarn-capacity-verification-2026-07-22.json), SHA-256 `168da54a58ef6b1b0aede8beda2564f4da24c232c9c8dc563933e11a8f6b328f`. |

**Overall verdict: ALL PASS for capacity-only context.** Qwen's model-scoped 8K speed result remains
negative, 32K speed evidence remains hardware-unavailable under the frozen unthrottled contract,
near-128K remains a checkpoint-limit refusal, and KVarN speed remains shelved.

## Llama-3.3-70B model-scoped runtime closure — 2026-07-23

**Story:** an Apple-Silicon operator can inspect a materially different model family's loaded
speed, capacity, quality, and task evidence without converting a negative speed result, a thermal
failure, a memory projection, or finite loss into a broad/default claim.

| Acceptance criterion | Verdict | Fresh evidence |
| --- | --- | --- |
| Exact source and model boundary | **PASS** | Loaded speed `c8a56ef00f6137b0bebfd6e494bfd9099a6a57fd`, generalized capacity `349331443bc37a236b1460681fe24c9c91979089`, and sealed quality/task `90337258aa741e987111c45e7102a53a29f7415c` each bind source-lock receipt `145127546c6c9872e80512716494eed77905d6e3ddd398c47c8f34a5ec796a4f`, model revision `de2dfaf56839b7d0e834157d2401dee02726874d`, checkpoint `5083c6af8977bfe2afae26835a7163f4dd16f25499d5f34eb4184c2b97be1400`, and tokenizer `da67fb228e066467d7fe3759a1277252c83ca61c81d6e453e553a8edbfbe58ab`. Qwen KVTuner was not reused. |
| Loaded 8K speed | **NEGATIVE / DOMINATED** | fp16, affine materialize, and affine direct medians were 270.06/11.87, 268.58/10.51, and 199.41/11.74 prefill/decode tok/s. Direct improved decode 11.70% over its same-storage control but remained 1.10% below fp16 and regressed prefill 26.16%; no speed tier or default follows. |
| Loaded 32K speed | **FAILED CLOSED / HARDWARE-UNAVAILABLE** | The first fp16 control changed nominal -> fair during retained work and emitted zero evidence/receipts. No retry or partial aggregate is used. |
| Loaded 32K capacity | **PASS / NON-PROMOTABLE** | Manifest `84151bdb34b1ac9825e9ea3dcb0f8fe549ea252e23f989e595efe0a8161e0fdb`, evidence `8b6e147adebd1a705103222f1978f1a86916469e96847a1a105370be83871ab1`, and completion `e3850c665e1a2ddf59dc23e49658e48fe961774dd958b849934f6cd0b79faf6e` bind 32,640+128 tokens, 80 affine-direct layers, zero materialization, 92.61/9.96 tok/s, 130,257,818,032 physical-footprint bytes, 83,292,012,544 maximum RSS bytes, `promotable:false`, and `speedAggregation:"forbidden"`. |
| Near-128K boundary | **NO-GO / NOT RUN** | Scaling the authenticated 32K physical footprint, packed storage, and direct workspace projects 150,344,865,136 bytes, 12,905,911,664 above the host's 137,438,953,472 physical bytes before safety margin. This is a reviewed projection, not a runtime/refusal row. |
| Sealed teacher-forced replay | **PASS FOR MEASUREMENT / NON-PROMOTABLE** | Sealed manifest `0d34303497c4f5a6c8b73a8e146cd34914adeb8adc3b4755c4d8e4983df894a1`; fp16 evidence `57ad74aca7e60bc8a97cfcbe66955c22d52e76514575aa6ec01b5005c1ee82d1`; affine evidence `7feb19468f307b06084cf65166b7cc38c9bc501c1e83e885601d0abd2e714c53`. Affine median KL was 0.0069743, pooled p95 1.0595306, perplexity delta -8.8987%, and top-1 285/328 with zero materialization/normalization. Deepest scored context 22,541 is below the required 24,000-token promotion bar. |
| Paired task coherence | **REJECTED / HARD-FLOOR FAIL** | fp16 raw `91bb6b290530110244fadd21d587cf4d0727a97c76a2d30fdb0ad569a95b31e4` authenticated 10/20 math, 7/20 code, 20/20 structured, 20/20 long retrieval, and 20/20 structured validity. Affine raw `a171b49da5be1c371f5628c3efbb9d042b4cb6f4f46216310e09889391765345`, summary `960c44c1e6a1e233dd70f15e530f6c909437cf7862071687630d0467470cfcbd`, and terminal status `5b7b29ff051dc0fb718b86fc04908fc961f0e6849a6e3993d1d0420f4cb0b97b` bind 10/20 math, 6/20 code, 14/20 structured, 20/20 long retrieval, and 14/20 structured validity. Negative-auth script/log `a50d9105d1fbdabb62ecf0d49b8bab75514bf895735afff23653ebaf13ec23fa` / `59183ca4cd024dace4771cf7695f7c8117e8edf60c629727884fc5f1c150ebbc` strictly re-derived `hardFloorPassed:false` and `balancedTaskDeltaPassed:false`; the failed boundary and sibling lock remain unchanged. |

Machine-readable criterion mapping:
[`llama3-70b-runtime-verification-2026-07-23.json`](superpowers/verdicts/llama3-70b-runtime-verification-2026-07-23.json),
SHA-256 `4dc2870d2f6872fd9a46d85993cf91c24ba0aaab506d0e88da460308b3e912a4`.

**Overall verdict: model cycle closes without a Llama speed tier or broad/default claim.** The 32K
affine row is capacity context only; near-128K has no executed row; long-depth quality remains
below the frozen depth gate; and the affine task candidate failed its structured-validity hard
floor. The separate Phi-4-mini third-geometry closure below records the completed follow-on proof.

## Phi-4-mini third-geometry runtime closure — 2026-07-23

**Story:** an Apple-Silicon operator can see whether compressed attention works on materially
different Phi3 Q24/KV8/D128 geometry without converting exact-source admission, a decode-only win,
long-context fit, or an unpaired quality measurement into a broad/default model tier.

| Acceptance criterion | Verdict | Fresh evidence |
| --- | --- | --- |
| Exact source and model boundary | **PASS** | Exact revision `ac1c269cb4222a4e136a3d09edad301056c1f36a`, checkpoint `c5ccdee8b3d37fd42c7e42e3e22c47c7549cdaa0b82904bdbfb1a52b31af5ec8`, tokenizer `7d8143e8d1f217f392488bddfb4febf2a53b783a2a00bf698b29218865a3620a`, source-lock receipt `f5abbbfd3d6989488160c9e5cc775c52aa928881786a6bc8893c48e087d61c2f`, completion `24c9afffd5b5b741e6845dded295eb36974dc6ee4575b73aae93a69796ab9ece`, and authenticator output `bff714f37f70e75c8a39df3e34fccd24495b1abe6ee635d090f8481ce978ff05` bind the 32-layer Phi3 Q24/KV8/D128 partial-LongRoPE snapshot. |
| Source, registry/load, and failure paths | **PASS** | Clean `850997e93e37575f4922bf213d558d63c80f1c40` admits only the exact Phi3 geometry and exact inert `sliding_window == 262144` above `max_position_embeddings == 131072`, with `use_sliding_window` absent. Changed geometry, active or changed windows, unsupported architectures, malformed source identity, and generic sliding-window configurations fail closed. |
| Corrected LongRoPE reference path | **PASS** | Clean `e29ee4cac7894a10cc6dc532a41f8b5d27f7c034` pins every discovered Python `SuScaledRoPE` to the source-locked long factors before chunk zero. Five focused Python regressions and compile check pass; 541 HarnessCore XCTest plus 17 Swift Testing tests pass off-box. On llmbench, 113/113 `FastMLXHarnessTests`, 157/157 `SpikeCoreTests`, and Release build pass; binary SHA-256 is `1c56ebb9a14fb72c2a4642dceacf0552f463923c455271b6557ceaecf1e6ed01`. |
| Loaded 8K speed | **NEGATIVE / DOMINATED** | Clean `9c542ef1d681ee1ea93d22958885ff6a41072a50` completed 9/9 cyclic rows. fp16, affine materialize, and affine direct medians were 3540.95/124.80, 3538.09/90.57, and 2011.61/148.54 prefill/decode tok/s. Direct decode improved 19.022% versus fp16 and 64.006% versus materialize, but prefill regressed 43.190% and 43.144%; the unchanged joint speed gate fails. Auth receipt/completion/receipt-set SHA-256 values are `ced79c0b3d8fc959ef1c83997817b5f02f58da8007584b67fe5b111b79533337`, `976051ffaf75b5cfdf19896c892b0bc0b2ff25fcd62b1dade52ff3488a5f2a38`, and `f1f518288526ead47d86c37bd13e380e8275255748edf8532c8543aa34ab425d`. |
| Loaded 32K and near-128K capacity | **PASS / NON-PROMOTABLE** | Separate affine-direct rows completed 32,628+128 tokens at 665.969/106.341 tok/s and 130,944+128 tokens at 178.917/45.028 tok/s. Both prove 32 direct layers, zero materialization, exact storage/workspace/memory telemetry, `promotable:false`, and `speedAggregation:"forbidden"`. Independent auth receipt SHA-256 values are `7251d261075fd14a32b26643d717c7f1a6ab3b246cea28c7b67beb47fe556140` and `eb9ce27f21af30c00bce8cb1aefe8a8a876ecbfde645012ad9592420e5e4a72a`. |
| Sealed teacher-forced loss | **PASS FOR FLOOR / NON-PROMOTABLE** | The corrected sealed teacher scored 328 positions through token 27,145. fp16 median KL / pooled p95 / PPL delta / top-1 were `0.0000431723` / `0.0001306214` / `+0.012199%` / `327/328`. Affine values were `0.2148781` / `1.4232503` / `+33.1269%` / `254/328`; its long-context tail p95 was `1.617082` and candidate/reference perplexity ratio `1.331269`. The affine row clears the frozen non-garbage perplexity, tail, top-1, and depth floor while preserving material loss as a warning; it cannot become a quality tier without the paired task gate. Teacher/fp16/affine evidence SHA-256 values are `7d3263012f891d2eb84f4063efabd07189c44dcf916dae7b62246422191d773d`, `a55af6f14b65f4b41c74aee8c680b6dc3a2a20893a6b60ef7ebc99196e78b3e8`, and `99b0b19d0ad80030ce81793b357fbf998cce0b7d5049d47d24aa984672158050`. |
| Paired task coherence | **AUTHENTICATED INVALID REFERENCE / REFUSED** | The fp16 baseline scored 8/20 math, 1/20 code, 15/20 structured, 20/20 long retrieval, and 15/20 structured validity. Independent receipt `a2e3f58cde07b760841866bfd1e362efe0a3665e3c8a1a4d1573d49d0d6f5ee7` authenticates all 80 rows, the summary, terminal boundary, owners, and preserved locks while recording `referenceAdmissible:false` and `affineReplayRunnable:false`. The candidate correctly never loaded; the floor was not weakened. |
| Broad/default claim refusal | **PASS** | No Phi speed, quality, default, generic sliding-window, broad-family, cross-family, or reusable-KVTuner claim follows. The exact implementation remains explicit-only; the capacity rows and measured teacher-forced loss remain visible under their non-promotable dispositions. |

Machine-readable criterion mapping:
[`phi4-mini-runtime-verification-2026-07-23.json`](superpowers/verdicts/phi4-mini-runtime-verification-2026-07-23.json),
SHA-256 `52d1865b27c78a69d7c2b4c371f1d8d8d3acf346aaafad4b4d329d2bfe238ef8`.

**Overall verdict: PARTIAL / NO MODEL TIER.** Exact-source Phi3 runtime portability and the
teacher-forced non-garbage floor pass. The 8K speed gate fails; long-context rows are capacity
context only; the invalid fp16 task reference makes paired task qualification unavailable.
Generic-window and broad/default claims remain rejected.
