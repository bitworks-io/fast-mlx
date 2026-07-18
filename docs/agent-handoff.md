# Agent Handoff

Last reviewed: 2026-07-18

For the next Codex/Claude Code/human agent. Decision-focused; link durable artifacts instead of
rediscovering status from chat.

## Project Purpose

**fast-mlx** is an Apple-Silicon MLX inference platform: Swift engine, planned native macOS
product, Python/research plane, engine-agnostic conformance and precision-loss harness, and an
optimization dial with quantified quality loss. The product goal is to serve every model supported
by the pinned MLX / MLX-Swift stack, with optimization investment prioritized for popular families
and high-leverage operator workflows. Qwen3-32B is representative/model-specific evidence only.
Broad support claims require model-support and architecture matrices plus at least one second,
materially different popular family. Do not hardcode model names in shared paths.

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
Shelved/rejected cells remain model-specific controls. This is Qwen3-32B-4bit evidence only.

**Content practice:** `docs/content/` now has 11 pieces. Keep writing one dated content piece per
notable spike, including negative results.

## Open Work Queue

1. **Fused compressed-domain KV attention** — current top engine gate. Carry forward KVarN i8 plus
   shared affine/KVTuner storage primitives. Prove full 32K/128K end-to-end behavior and the
   batch-compaction poison case before any speed claim. Add a second materially different popular
   family before broad support. [Task](task-inbox/2026-07-12-fused-compressed-kv-attention.md).
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

Models on `llmbench`: `~/perf-work/models/` (Qwen3-32B-4bit and 8-bit staged; no Qwen3-32B BF16
target). Python: `~/harness-venv` (`transformers<5`).

## Commit / Checkin

Docs -> `main`. Features -> branch, merge `--no-ff` after verification. Commit messages end with
`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Never commit secrets, transient logs,
runtime cache paths, or machine-local state.
