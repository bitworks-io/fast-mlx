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
