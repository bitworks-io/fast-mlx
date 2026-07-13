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
| As a maintainer, I can enter continuous-batching actor integration only after dense cache shapes are exact and compile-stable. | Scalar-aligned merge/append/filter/extract preserves logits, greedy tokens, row identity, and one trace per stable shape; unsupported state layouts fail closed. | 9 focused `BatchedCompiledKVCacheTests`; 29/29 total `SpikeCoreTests` through Xcode; 134 HarnessCore XCTest + 17 Swift Testing tests off-box. | Clean `7b9d709`: Qwen3-32B-4bit B=1/2/4/7/8 fixed+shapeless PASS with zero initial max-logit delta; 64-step B=4/B=8 fixed PASS; `qwen3_moe` rejected before load. | 2026-07-12 | Phase 2 actor/stream integration, concurrent service metrics, cancellation latency, and throughput frontier remain open. |

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
```

## Latest Acceptance Pass

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

**Gap:** this closes only cache/shape feasibility. It is not the continuous-batching promotion
verdict: actor integration, streaming cancellation, chunked prefill, aggregate throughput,
per-request TTFT/TPOT/fairness, and soak evidence remain Phase 2–3 work.
