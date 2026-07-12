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
| Give every new recommendation a disposition and bounded owner | **PASS** | Handoff ranks 11 performance cycles; seven new inbox seeds own trained speculation, batching, KV storage quality, fused attention, prefix/session caching, sampling, and learned weight quantization. Research-later/rejected candidates remain in the intake/audit rather than becoming duplicate tasks. |
| Rank by impact, evidence, cost, and Apple fit without promoting research | **PASS** | The audit's scored matrix labels every external number unverified until a clean-SHA fast-mlx run. KVarN storage quality precedes its fused Metal phase; TurboQuant owns its own closure ablations. |
| Primary sources support material claims | **PASS** | Independent source review caught and corrected the EAGLE `acceptance_length` denominator and the post-paper KVarN attribution. Final source re-review: **No issues found**. All 35 public external URLs introduced/touched by this change returned HTTP 200; the pre-existing authenticated Claude Artifact returned 403 as expected. |
| Durable docs are internally navigable and review-clean | **PASS** | Repository-wide relative-link check passed across 56 Markdown files; `git diff --check` passed. Final focused review found no High/Medium correctness or priority issues. |

**Not run:** Swift/Xcode tests and the Apple-Silicon bench. This is a documentation/research
cycle with no engine change; running a technique benchmark would falsely imply that an
external candidate had entered the flywheel. Every performance result in the audit remains a
local historical result or a source-labeled external claim.

**Overall verdict:** ALL PASS for the Sol portfolio-audit acceptance criteria. Engine
promotion gates remain intentionally open in their individual task seeds.
