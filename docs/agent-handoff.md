# Agent Handoff

Last reviewed: 2026-07-21

For the next Codex/Claude Code/human agent. Decision-focused; link durable artifacts instead of
rediscovering status from chat.

## Project Purpose

**fast-mlx** is an Apple-Silicon MLX inference platform: Swift engine, planned native macOS
product, Python/research plane, engine-agnostic conformance and precision-loss harness, and an
optimization dial with quantified quality loss. The product goal is to serve every model supported
by the pinned MLX / MLX-Swift stack, with optimization investment prioritized for popular families
and high-leverage operator workflows. Qwen3-32B is representative/model-specific evidence only.
Broad support claims require model-support and architecture matrices plus at least one additional
popular model with materially different attention geometry. Do not hardcode model names in shared
paths.

Design source: [`2026-07-08-fast-mlx-platform-design.md`](superpowers/specs/2026-07-08-fast-mlx-platform-design.md).

## Current State

**Shipped engine/operator surface:** measured compiled decode seed, strict actor confinement,
`HarnessCore` equivalence/engagement/acceptance triad, teacher-forced KL/perplexity/tail-p95,
capacity model by architecture, `SystemProfiler`, `fastmlx-capacity`, PLD framework/gate tuning,
and exact continuous-batching/chunked-prefill engine proof. The currently shipped operator surface
is the engine, harness, and capacity CLI.

**Not shipped yet:** adaptive native macOS UI/dial, production OpenAI-compatible service route,
model management, menu-bar controls/metrics, auto-update, website/community surfaces, and
benchmark publication automation. These are high-priority roadmap work, not current product
surface. Captures include [`2026-07-18-adaptive-macos-dial-ui.md`](task-inbox/2026-07-18-adaptive-macos-dial-ui.md)
and [`2026-07-18-website-benchmark-community.md`](task-inbox/2026-07-18-website-benchmark-community.md).

**Major dated outcomes:** TurboQuant v1 was shelved as a dated negative result; PLD was promoted
as exact for repetition-heavy/echo workloads; Qwen3-32B EAGLE-3 was shelved on byte-identity
failure; continuous batching was promoted as an exact engine building block but has no production
API route yet.

**KVarN/asymmetric KV frontier — COMPLETE (2026-07-18):** selected lossy capacity tiers were
promoted, but no speed tier was proven. Verdict:
[`2026-07-18-kvarn-kv-frontier.md`](superpowers/verdicts/2026-07-18-kvarn-kv-frontier.md);
content: [`2026-07-18-when-smaller-kv-is-not-faster.md`](content/2026-07-18-when-smaller-kv-is-not-faster.md).
Promoted cells: fp16 KV as Transparent baseline, affine K4V2-g64 as Balanced capacity, frozen
KVTuner as explicit Max-fit, and KVarN i8 as capacity-only Max-fit plus fused-kernel candidate.
Shelved/rejected cells remain model-specific controls. This is Qwen3-32B-4bit evidence only; the
frozen KVTuner schedule is Qwen-only unless independently calibrated and authenticated for another
family.

**Compressed-attention loaded qualification — ACTIVE, NOT PROMOTED (2026-07-21):** the portable
affine/KVTuner router, direct KVarN K4V2/G128/i8 runtime, and hostile-compaction-safe continuous
batching are implemented and verified. Clean `cf3248b594da4bfced428347c89b833d928b243b` repaired
mixed bf16/fp32 KVarN ingress telemetry; its loaded-Qwen smoke completed 3/3 with 64 engaged layers,
zero materialization, and authenticated receipts. Qwen 8K v7 is terminal
`INVALID_BLOCK_ENVIRONMENT`; v8 materialize and v9 direct preconditions are terminal
`PRECONDITION_NOT_FAIR`. Preserve all three boundaries: they proved workload-only preheating could
not create a stable fair cohort and must not be resumed or promoted.

Clean `b1289b783c1e156355a24c5db5f0e9b150a1cb3b` now records the dropped warmup and requires a
bounded post-warmup nominal/AC/non-low-power admission before every retained qualification row.
The resulting v10 matrix is terminal `FAILED` and preserved unchanged at
`/Users/llmbench/perf-work/results/fused-compressed-kv-qwen3-32b-loaded-cf3248b/qwen-8k-v10-nominal`.
It stopped after 5/49 authenticated rows at block 0, position 5,
`affine-k4v2-g64-materialize`, because point-in-time nominal admission did not survive the
retained measurement. Its immutable manifest SHA-256 is
`353820692294f408ddae91e7f3b7d1522604ea77e40646c6a45d812af44434e1`, terminal status SHA-256 is
`03cbb6c52e522ce55e217e3f9d66dbd66d21ef6bf85024016a3b7f3080c7944f`, and rejected partial
receipt-set SHA-256 is `2c8c9b1e2b7dc24f70bdb9935f9bcbdb9ad73efd06a249809dbb5906fa6f2889`.
Those five rows are diagnostic only and cannot promote a cell or speed tier.

Clean `d4102e6a3029b161d99ee27aceabbad8d5696fb5` adds a manifest-bound continuous stability dwell
and hash-bound, non-promotable failure receipts. Fresh runner manifests use schema v3 and require a
positive dwell no longer than the timeout. Evidence schema v4 authenticates the ordered
nominal/AC/non-low-power observation window, while evidence v3 remains readable as
instantaneous-admission compatibility and v2 remains legacy-readable only. It passes 519
HarnessCore XCTest plus 17 Swift Testing tests off-box, 95/95 FastMLXHarnessTests and 155/155
SpikeCoreTests through Xcode on llmbench, Release build, two final no-issue reviews, ShellCheck,
and staged gitleaks. Its first loaded-Qwen boundary froze a 60-second continuous nominal dwell.

That v11 preflight is now terminal `FAILED` after 2/9 authenticated rows and no completed block.
The fp16 and KVarN-materialize rows each proved a 60-second nominal dwell plus exact retained
nominal/nominal equality. Block 0, position 2, `kvarn-k4v2-g128-direct` then reacquired nominal for
60 seconds after a fair warmup but changed nominal -> fair during its roughly 144-second retained
measurement. The non-promotable failure receipt SHA-256 is
`02893f30229f79861f95e8d037536ae6d0bca7855539bbc88ab2f82dd788293b`; partial receipt-list and
block-receipt SHA-256 values are
`ae9dcb7e0ef082aff6cd6509c13a9e41f85fe8937d0a203867d762559c259a56` and
`2780c3a9b4880236082feb3dcca26613f69220c854d68dc0db39f67ea0e7ac90`. The complete runner log
SHA-256 is `02ade4793feafc228a22c5a99afc8fe8ce4ab14252abaece25805dc8eb0bc291`.
Preserve v11 unchanged; its two rows and failed-row timings are diagnostic only.

The bounded loaded-path trace is now terminal `FAILED` and preserved unchanged at
`/Users/llmbench/perf-work/results/fused-compressed-kv-profile-d4102e6/qwen-8k-kvarn-direct-metal-v1`.
High Power Mode was active (`pmset powermode 2`), Foundation reported Low Power Mode false and
thermal nominal, and the host remained on 140 W AC before and after. `xctrace` reached its
240-second limit but did not publish a complete exportable trace before the launcher's bounded
save watchdog. The
preserved raw Apple Trace File is 16,441,155,536 bytes with SHA-256
`12073b786fb06d5569269500129bee3f9b1926319f9a34a9de97c5cdf24853ea`; peak recorder RSS was
107,315,600 KiB. The terminal status SHA-256 is
`982b2e8659cd53abb2c403c15e949cd6268fffa94f48c06cb85837141c2958b2`. `xctrace export --toc`
fails `Document Missing Template Error`, so the trace cannot support kernel attribution or any
promotion claim. No target, recorder, runner, watchdog, or lock remains active.

The engineering disposition is now **SHELVE KVarN's speed role and retain its already-qualified
capacity-only role**. V11's hash-bound diagnostic was 7.18 decode tok/s versus 0.46 for the same
storage materialize control, proving the direct path removes a real hot-path penalty, but fp16 was
23.48 tok/s. Clearing the frozen speed gate would require about 3.43x the observed direct decode
throughput. Prefill was 63.26 tok/s versus 533.73 for fp16, requiring about 8.02x to reach the
allowed 5% envelope. Those partial timings remain non-promotable; their magnitude plus the mapped
independent prefill/packed-attention costs rules out one credible actor-confined recovery. Do not
spend another cycle on a KVarN speed retry.

The amended Qwen 8K five-cell matrix is now terminal `COMPLETE` and independently authenticated,
but **neither direct candidate promotes**. Preserve it unchanged at
`/Users/llmbench/perf-work/results/fused-compressed-kv-qwen3-32b-loaded-a2af840/qwen-8k-v12-five-cell`.
Clean source `a2af840d6f02c3a9097e4df0372e969d18bd7bc8`, binary SHA-256
`7f187d6986906eed8d90753170f8b7a91cdc720bb860ef366bb46ff64814d4a4`, runner SHA-256
`e2f6e55bb0aeae6b1ec585f6d0d3c85b13c00879e13ed2a0fa3826ee074f8c0f`, and manifest SHA-256
`551504e541b8e9a21786be536abc24b31ea68dea1dd71b5b0d0819a04ef92591` produced 25/25 schema-v4
rows across five stable nominal/AC/non-low-power blocks. Completion, receipt-set, and runner-log
SHA-256 values are `762c276456f33bab6927af6aa38297a9949b381de868b9b70f43b218b18e8b1d`,
`bd716829efa223ffb46c25f304df81c40cbac07e29890b3f1590afd50d85bfa1`, and
`71344817efd30bcf099f62cba814c7937599ee8c9b126aafdc293764761640c6`.

Median decode/prefill tok/s were fp16 23.32/531.19, affine materialize 19.64/531.19, affine direct
24.19/333.29, KVTuner materialize 19.58/531.42, and KVTuner direct 23.19/333.34. Affine direct beat
its materialize control in all five blocks by 15.1-23.4%, but beat fp16 by at least 5% in only one
block and retained just 62.6-78.1% of fp16 prefill. KVTuner direct beat its materialize control by
18.2-18.8%, but its fp16 decode ratio ranged from 0.989 to 1.137 and its prefill ratio from 0.626
to 0.876. The fail-closed reducer therefore reports both all-block gates `false`. This is clean
negative/dominated evidence: no 8K speed tier or default changes. Continue Qwen 32K for long-context
runtime/capacity behavior, with KVarN measured separately as capacity-only, then record Qwen's
authenticated near-128K refusal. Preserve every v7-v12 and trace-v1 boundary unchanged.

**Content practice:** `docs/content/` now has 12 pieces. Keep writing one dated content piece per
notable spike, including negative results.

## Open Work Queue

1. **Fused compressed-domain KV attention** — current top engine gate. Preserve v7-v12 and the
   failed trace; never resume, overwrite, or promote partial evidence. KVarN's direct speed role is
   SHELVED and its capacity-only role remains. The exact five-cell Qwen 8K matrix completed and
   failed the unchanged all-block speed gate; retain both affine and KVTuner direct cells as
   negative/dominated 8K evidence. Next run Qwen 32K with the same authenticated speed controls,
   include a separately authenticated KVarN capacity-only row, and record the authenticated 128K
   refusal (maximum context 40,960). Then qualify source-locked Llama-3.3-70B at
   8K/32K/near-128K without the Qwen-specific KVTuner schedule unless independently calibrated.
   Both first families share
   Q64/KV8/D128; add a popular materially different geometry before any broad/default support
   claim. Synthetic geometry, implementation tests, or partial loaded rows cannot promote a dial
   tier.
   [Task](task-inbox/2026-07-12-fused-compressed-kv-attention.md).
2. **Exact prefix/session cache + request-start stack** — hot-cache TTFT, positive commit, eager
   warmup, template/tokenize cache; SSD later. Design over batching/cache ownership.
   [Task](task-inbox/2026-07-12-exact-prefix-session-cache.md).
3. **Continuous-batching serving route** — production OpenAI-compatible route, real disconnect
   propagation, dynamic policy boundaries. [Task](task-inbox/2026-07-14-continuous-batching-serving-route.md).
4. **Absorbed MLA** — reduce expanded DeepSeek-style cache; Python MLX and Swift GLM code are
   useful oracles. [Task](task-inbox/2026-07-09-absorbed-mla-kv-cache.md).
5. **Sampled-generation foundation -> sampler fusion** — define RNG/distribution contracts before
   porting the L1/L3/L1b/L3b stack. [Task](task-inbox/2026-07-12-sampled-generation-sampler-fusion.md).
6. **Learned/mixed weight-quant sweep** — affine vs official MLX dynamic/DWQ/oQ4e, output ordinary
   MLX checkpoints for the Swift loop. [Task](task-inbox/2026-07-12-learned-weight-quant-frontier.md).
7. **Operability and watchdogs** — large-prefill capacity, runtime admission control, long-context
   Metal watchdog. [Task](task-inbox/2026-07-14-long-context-metal-watchdog.md).
8. **TurboQuant Spike B closure** — bounded second-failure rule after stronger affine/KVarN/fused
   baselines exist. [Task](task-inbox/2026-07-09-turboquant-spike-b-outlier-channels.md).

**Public evidence/community platform — ROADMAP CAPTURED (2026-07-15).** The
[oMLX public-ecosystem review](reference/2026-07-15-omlx-public-ecosystem-review.md) is an internal
product reference for proof-led sites, community benchmark exploration, shareable results, update
channels, and release/community feedback loops. Public fast-mlx material must report fast-mlx's own
reproducible evidence; competitor performance and feature litmus stays private. The
[implementation task and agent spawn packets](task-inbox/2026-07-15-public-evidence-community-platform.md)
define a parallel product lane: a dial/frontier website, versioned public fast-mlx evidence schema,
automated lab collection, explicit-opt-in community submissions, machine-readable agent surfaces,
and a signed stable/RC/dev updater. Any same-box competitor reproduction remains a separate private
engineering input. No site, public ingest, or updater has been implemented yet.

**Parallel product track — high priority:** begin planning the adaptive macOS dial UI now, while
engine profiling continues. The product surface must make first-run model install, chat/serve,
hardware-fit guidance, measured quality-versus-performance selection, live metrics, updates, and
recovery easy on both low- and high-end Apple Silicon. It must remain optional and add zero linked,
initialized, or resident overhead to headless serving. Implementation is not yet present and stays
gated on stable service/catalog/metrics contracts. [UI task](task-inbox/2026-07-18-adaptive-macos-dial-ui.md);
[website/community task](task-inbox/2026-07-18-website-benchmark-community.md).

Every flywheel cycle: classify as `EXACT`, `LOSSY_FRONTIER`, or `EXPERIMENTAL`; implement behind a
flag; run lane-appropriate exactness/quality/perf checks; promote to a dial tier or shelve with a
dated verdict; write a content piece. Lossy candidates may promote only with quantified useful loss
above the non-garbage floor.

## Watch / Intake

The public-only Sol monitor is active daily at 08:15 local time:
[`2026-07-18-ecosystem-intelligence-watch.md`](task-inbox/2026-07-18-ecosystem-intelligence-watch.md).
It watches official repos/releases for MLX, MLX Swift/LM, mlx-serve, oMLX, llama.cpp; Hugging Face;
arXiv; GitHub Trending/topics; and TLDR AI as a weak lead only. Private mlx-serve/oMLX comparison is
an internal litmus for feature/perf/moat gaps. Public docs should publish only fast-mlx results.
Repo-side automated intake, dedupe, and PROMOTE/SHELVE ledgering remain pending.

Current upstream watch date: 2026-07-18. [`mlx-serve v26.7.9`](https://github.com/ddalcu/mlx-serve/releases/tag/v26.7.9)
and [`oMLX v0.5.2.dev1`](https://github.com/jundot/omlx/releases/tag/v0.5.2.dev1) are upstream
claims/leads, not local fast-mlx evidence.

## Invariants

- MLX state is non-Sendable: keep it actor-confined. `@unchecked` / `nonisolated(unsafe)` are banned.
- Precision loss is measured teacher-forced against a locked context, never by free-running drift.
- Temp-0 spec-decode must be byte-identical. A lossy spec path is a correctness bug.
- The cacheLimit invariant is load-bearing: whenever `iogpu.wired_limit_mb` is raised, set explicit
  `Memory.cacheLimit` from `CapacityModel.recommendedCacheLimitBytes`.
- MLX-coupled tests require Xcode on `llmbench`; SwiftPM CLI tests cannot load the MLX metallib.
- Preserve user/authored changes and do not infer fast-mlx implementation status from incumbent or
  upstream dependency claims.

## Test Commands

```sh
# Off-box pure HarnessCore + capacity CLI:
cd spike && swift test --filter HarnessCoreTests
swift run fastmlx-capacity --box m3Ultra512
```

```sh
# On-box MLX/SpikeCore/engine checks:
bash spike/scripts/sync_llmbench.sh
ssh llmbench@192.168.1.252 'cd ~/fast-mlx-spike && DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild test -scheme fast-mlx-spike-Package -destination "platform=macOS" -skipPackagePluginValidation -only-testing:SpikeCoreTests'
```

Models on `llmbench`: `~/perf-work/models/` (Qwen3-32B-4bit and 8-bit staged; staged
Llama-3.3-70B-Instruct-4bit checkpoint; no Qwen3-32B BF16 target). Python: `~/harness-venv`
(`transformers<5`).

## Commit / Checkin

Docs -> `main`. Features -> branch, merge `--no-ff` after verification. Commit messages end with
`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Never commit secrets, transient logs,
runtime cache paths, or machine-local state.
