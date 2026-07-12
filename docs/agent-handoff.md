# Agent Handoff

Last reviewed: 2026-07-11

For the next Codex/Claude Code/human agent. Decision-focused; links to the durable specs rather than duplicating them.

## Project Purpose

**fast-mlx** — the fastest, most-optimized MLX inference platform for Apple Silicon, superseding the incumbent Zig `mlx-serve`. A Swift engine + macOS app + Python train/research plane + an engine-agnostic conformance/precision-loss harness. The product wedge is the **optimization dial with quantified precision loss** ("dial in speed, see exactly what accuracy you trade"); the moat is the **technique-integration flywheel** (absorb a new inference technique → measure its speed↔quality frontier → promote to a dial tier or shelve with a dated negative result). First production deployment: **Concierge** (bitworks' shopping assistant). Dev/bench box: **M5 Max 128GB** (`llmbench@192.168.1.252`); production target up to M3 Ultra 512GB. Full design: [`docs/superpowers/specs/2026-07-08-fast-mlx-platform-design.md`](superpowers/specs/2026-07-08-fast-mlx-platform-design.md).

## Current state (2026-07-11)

**Shipped to `main`:**
- **Engine seed** (`spike/`): compiled decode core (`CompiledMLXDecoder` + `CompiledKVCache`) at 155.4 tok/s (≥ Zig), single-owner `InferenceActor`, Swift 6 strict-concurrency clean.
- **The harness spine** (`HarnessCore`, pure/Foundation): equivalence+engagement+acceptance triad, teacher-forced KL + perplexity + long-context tail-p95, versioned corpus (incl. a 24,151-token entry), provenance/JSONL. Established baseline: **4-bit affine KV vs bf16 = tail-p95 1.665 nats @24K, ppl +21.4%**.
- **System-aware context operability**: the per-arch KV **capacity memory model** (`HarnessCore/CapacityModel.swift` — dispatches KV/token by `model_type`; naive formula is wrong 4×–71× for hybrid-linear/SWA/MLA/Mamba2), the `SystemProfiler` (real host introspection, MLX-free), and the **`fastmlx-capacity` CLI** (`swift run fastmlx-capacity [--box …]`). Spec: [`2026-07-09-system-aware-context-operability.md`](superpowers/specs/2026-07-09-system-aware-context-operability.md). Catalog (context caps + memory + Nemotron-3-Ultra/Ornith resolved) in platform spec §9.

**TurboQuant KV-quant — COMPLETE + SHELVED** (merged to `main`, `6e82e2b`; plan [`2026-07-09-turboquant-kv-quant.md`](superpowers/plans/2026-07-09-turboquant-kv-quant.md)). The flywheel's first novel technique ran the full loop: Google TurboQuant (arXiv:2504.19874) built exactly (`LloydMaxCodebook` + `TurboQuantCodec` Haar/LUT/QJL — Spike A verified against the paper's Theorem-2 distortion table), integrated as `TurboQuantKVCache` (materialize-then-attend, behind the `tq2.5`/`tq3.5` flag), measured, and **SHELVED as a dated negative result** — uniform-v1 loses to 4-bit affine on Qwen3-32B (tqB3 tail-p95 1.797/ppl +32.6% vs 1.665/+21.4%; tqB2 catastrophic). Verdict: [`verdicts/2026-07-09-turboquant-firstrun.md`](superpowers/verdicts/2026-07-09-turboquant-firstrun.md). The fp16 default path is unchanged (60/60 regression, tests green).

### ▶ Gated next step — TurboQuant Spike B (outlier channels)
The paper's near-losslessness depends on outlier channels (32ch@3b+96@2b) that uniform-v1 deferred — the one lever that could move tqB3 under the baseline. Extend `TurboQuantCodec`'s bit allocation, re-measure through the same harness, then promote or fully shelve. The codec + cache + tier plumbing are already in place behind the flag. Task-inbox: [`2026-07-09-turboquant-spike-b-outlier-channels.md`](task-inbox/2026-07-09-turboquant-spike-b-outlier-channels.md). Route engine/MLX work to deep-reasoner (fable).

**Speculative decoding (PLD) — PROMOTED + MERGED** (`main`, `5deb1d0`; plan [`2026-07-09-speculative-decoding-pld.md`](superpowers/plans/2026-07-09-speculative-decoding-pld.md), verdict [`2026-07-09-pld-firstrun.md`](superpowers/verdicts/2026-07-09-pld-firstrun.md)). The **first decode-speed multiplier** beyond the GPU-bound base loop (we're at Zig parity on the base loop; PLD is the first of mlx-serve's multipliers to land). Prompt-lookup decoding: **exact (byte-identical at temp 0, proven 80/80 + 120/120)**, no draft model. Measured on Qwen3-32B-4bit: **echo/agent +97.5% (28.25 → 55.80 tok/s, 98% accept)**; code −3.7%, prose −2.1% (non-target overhead honestly characterized). Ships as a per-request `--spec pld` toggle (opt-in; default path unchanged). The framework (`HarnessCore/SpecDecode/` — `SpecDrafter`/`SpecAccept`/`SpecEmit`/`PLDGate`) is what **DSpark/DFlash will reuse**.
- **▶ Gated follow-up — PLD gate-tuning** before global default-on: keep non-target shapes ≈ neutral (preserve submit-first pipelining on fallback steps; make the gate disable faster on low yield). Task-inbox: [`2026-07-11-pld-gate-tuning.md`](task-inbox/2026-07-11-pld-gate-tuning.md).

**Also shipped since:** the [quality-metrics explainer](reference/quality-metrics-explained.md) + a **published Artifact** (`https://claude.ai/code/artifact/168d9b15-96e7-4f30-babf-b7ea64441438` — a user-facing "how we measure quality" page), and the **dial-as-informed-consent** refinement (platform spec §4: noticeable-but-valuable tiers with quantified loss + a hard garbage floor; PrismML 1-bit captured as a device-tier research candidate in the intake).

## ▶ Open work queue — pick the next flywheel cycle

Prioritized; each is a self-contained next step. Owner's north star: **match then beat the optimized mlx-serve** — the base loop is at Zig parity and PLD is the first multiplier landed; the rest are the multipliers still to test.

1. **PLD gate-tuning** — immediate, small ([`task-inbox/2026-07-11-pld-gate-tuning.md`](task-inbox/2026-07-11-pld-gate-tuning.md)). Make non-target shapes ≈ neutral (preserve submit-first pipelining on fallback steps; gate disables faster on low yield) so PLD can be default-on.
2. **DSpark** — EAGLE-3-style spec-decode (a trained drafter), the next decode multiplier; **reuses `HarnessCore/SpecDecode/`** (SpecDrafter/SpecAccept/SpecEmit/PLDGate). Design in [`reference/mlx-serve-archive/`](reference/mlx-serve-archive/) + the [carry-forward backlog](reference/2026-07-08-carry-forward-performance-backlog.md).
3. **Continuous batching + sampler fusion** — the remaining mlx-serve throughput multipliers (carry-forward backlog). NB: PLD is single-in-flight-KV only → it disables under batching (a named invariant in the spec-decode plan).
4. **TurboQuant Spike B** — outlier channels ([task-inbox](task-inbox/2026-07-09-turboquant-spike-b-outlier-channels.md)); the one lever that could un-shelve TurboQuant.
5. **Absorbed-MLA KV cache** — 71× DeepSeek-R1 KV reduction ([task-inbox](task-inbox/2026-07-09-absorbed-mla-kv-cache.md)); makes R1 viable at long context.
6. **Operability backlog** — chunked-prefill capacity measurement + runtime admission control (system-aware spec §7).
7. **PrismML 1-bit** — device-tier extreme-compression research (the informed-consent frontier); intake candidate.

Every one runs the same loop: implement behind a flag → triad + precision-loss harness → **promote to a dial tier or shelve with a dated verdict** (`docs/superpowers/verdicts/`) → write a `docs/content/` piece.

## For a Codex (OpenAI) agent picking this up

- **Methodology is [`AGENTS.md`](../AGENTS.md)** (the shared working agreement — read it first: operating mode, user-story/acceptance discipline, verification-before-completion, TDD, the multi-agent workflow, durable state). `CLAUDE.md` holds only Claude-Code-specific additions.
- **Sub-agents:** pinned in `.codex/agents/` (`codebase_explorer`, `docs_researcher`, `implementation_worker`, `reviewer`, `verifier`) with prompt packets in `.codex/prompts/`. **The model tiers named throughout this doc (haiku/sonnet/opus/fable) are Claude-specific — map to your own:** cheap for search/recon, mid for scoped edits, **your most capable for engine/MLX-coupled work** (`SpikeCore`, the compiled decode path, spec-decode, quantizers — these have repeatedly needed the top tier and defeated weaker ones).
- **The build gotcha that will bite you:** anything importing MLX (`SpikeCore`, the engine) **cannot** be tested with `swift test` — SwiftPM's CLI doesn't emit the MLX metallib. Use `xcodebuild … -skipPackagePluginValidation` on the on-box Mac (per Test Commands). Pure `HarnessCore` tests off-box with `swift test`. `spike/scripts/sync_llmbench.sh` pushes the tree + stamps the harness git SHA into evidence.
- **Load-bearing invariants:** (1) MLX state is non-Sendable → **actor-confined**, `@unchecked`/`nonisolated(unsafe)` BANNED; (2) precision-loss is measured **teacher-forced** (context-locked), never free-running; (3) spec-decode must be **byte-identical at temp 0** — a lossy spec path is a *bug*, not a trade; (4) the cacheLimit invariant (below).

## Key Components

- `spike/Sources/SpikeCore/` — the MLX engine (decode core, KV caches, TurboQuant codec). **Non-Sendable MLX state is actor-confined** (`MLXArray` is a non-Sendable class; `@unchecked`/`nonisolated(unsafe)` are BANNED).
- `spike/Sources/HarnessCore/` — **pure, Foundation-only** (no MLX): triad, quality metrics, corpus, provenance, `CapacityModel`/`ModelArchProfile`/`SystemProfile`, `LloydMaxCodebook`. Builds+tests off-box.
- `spike/Sources/SystemProfiler/` — Metal+Darwin host introspection (MLX-free). `spike/Sources/fastmlx-capacity/` — the capacity CLI.
- `spike/Sources/fastmlx-harness/` — the measurement CLI (`verify`/`bench`/`kl`) + `SwiftEngineDriver` (MLX). `scripts/harness_reference.py` — the bf16 Python reference.

## Test Commands

```sh
# Off-box (this host / FluffyMBA) — pure HarnessCore + capacity CLI:
cd spike && swift test --filter HarnessCoreTests
swift run fastmlx-capacity --box m3Ultra512
```
```sh
# On-box (llmbench) — anything importing MLX (SpikeCore, the engine). swift test CANNOT load the MLX metallib; xcodebuild is required:
bash spike/scripts/sync_llmbench.sh
ssh llmbench@192.168.1.252 'cd ~/fast-mlx-spike && DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild test -scheme fast-mlx-spike-Package -destination "platform=macOS" -skipPackagePluginValidation -only-testing:SpikeCoreTests'
```
Models on llmbench: `~/perf-work/models/` (Qwen3-32B-4bit + bf16 reference staged). Python: `~/harness-venv` (transformers<5).

## Known Risks / Open Items

- **Long on-box measurement runs** (long-context KL, multi-shape bench) stall the driving agent in monitor-waits and don't reliably auto-resume — check `~/fast-mlx-spike/harness-evidence.jsonl` + `pld-shapes.csv` on the box for partial results, then resume the finalization (verdict + commit) explicitly.
- **Model routing:** haiku=scout, sonnet=builder, opus=main/judgment, fable=deep-reasoner; always pass `model` explicitly; re-route UP one tier on failure. Engine/MLX-coupled work → deep-reasoner (builders have escalated on it).
- **cacheLimit invariant:** whenever `iogpu.wired_limit_mb` is raised, set an explicit `Memory.cacheLimit` (never the 1.5× default — the "7K wall" mechanism). Policy: `CapacityModel.recommendedCacheLimitBytes`.
- **Backlog** (`docs/task-inbox/`): absorbed-MLA KV cache (71× DeepSeek-R1 lever, unbuilt); chunked-prefill capacity measurement gate; runtime admission control.
- **Content-library practice:** after each notable spike/optimization, write a `docs/content/` piece (blog/whitepaper source). 6 pieces so far ([`docs/content/README.md`](content/README.md) indexes them).

## Commit / Checkin

Docs → `main`. Features → branch, merge `--no-ff` after verification. Commit messages end with a `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer. Never commit secrets, runtime cache paths, or machine-local state.
