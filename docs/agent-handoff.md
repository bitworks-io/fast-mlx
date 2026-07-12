# Agent Handoff

Last reviewed: 2026-07-12

For the next Codex/Claude Code/human agent. Decision-focused; links to the durable specs rather than duplicating them.

## Project Purpose

**fast-mlx** — the fastest, most-optimized MLX inference platform for Apple Silicon, superseding the incumbent Zig `mlx-serve`. A Swift engine + macOS app + Python train/research plane + an engine-agnostic conformance/precision-loss harness. The product wedge is the **optimization dial with quantified precision loss** ("dial in speed, see exactly what accuracy you trade"); the moat is the **technique-integration flywheel** (absorb a new inference technique → measure its speed↔quality frontier → promote to a dial tier or shelve with a dated negative result). First production deployment: **Concierge** (bitworks' shopping assistant). Dev/bench box: **M5 Max 128GB** (`llmbench@192.168.1.252`); production target up to M3 Ultra 512GB. Full design: [`docs/superpowers/specs/2026-07-08-fast-mlx-platform-design.md`](superpowers/specs/2026-07-08-fast-mlx-platform-design.md).

## Current state (2026-07-12)

**Shipped to `main`:**
- **Engine seed** (`spike/`): compiled decode core (`CompiledMLXDecoder` + `CompiledKVCache`) at 155.4 tok/s (≥ Zig), single-owner `InferenceActor`, Swift 6 strict-concurrency clean.
- **The harness spine** (`HarnessCore`, pure/Foundation): equivalence+engagement+acceptance triad, teacher-forced KL + perplexity + long-context tail-p95, versioned corpus (incl. a 24,151-token entry), provenance/JSONL. Established baseline: **4-bit affine KV vs bf16 = tail-p95 1.665 nats @24K, ppl +21.4%**.
- **System-aware context operability**: the per-arch KV **capacity memory model** (`HarnessCore/CapacityModel.swift` — dispatches KV/token by `model_type`; naive formula is wrong 4×–71× for hybrid-linear/SWA/MLA/Mamba2), the `SystemProfiler` (real host introspection, MLX-free), and the **`fastmlx-capacity` CLI** (`swift run fastmlx-capacity [--box …]`). Spec: [`2026-07-09-system-aware-context-operability.md`](superpowers/specs/2026-07-09-system-aware-context-operability.md). Catalog (context caps + memory + Nemotron-3-Ultra/Ornith resolved) in platform spec §9.

**TurboQuant KV-quant — COMPLETE + SHELVED** (merged to `main`, `6e82e2b`; plan [`2026-07-09-turboquant-kv-quant.md`](superpowers/plans/2026-07-09-turboquant-kv-quant.md)). The flywheel's first novel technique ran the full loop: Google TurboQuant (arXiv:2504.19874) built exactly (`LloydMaxCodebook` + `TurboQuantCodec` Haar/LUT/QJL — Spike A verified against the paper's Theorem-2 distortion table), integrated as `TurboQuantKVCache` (materialize-then-attend, behind the `tq2.5`/`tq3.5` flag), measured, and **SHELVED as a dated negative result** — uniform-v1 loses to 4-bit affine on Qwen3-32B (tqB3 tail-p95 1.797/ppl +32.6% vs 1.665/+21.4%; tqB2 catastrophic). Verdict: [`verdicts/2026-07-09-turboquant-firstrun.md`](superpowers/verdicts/2026-07-09-turboquant-firstrun.md). The fp16 default path is unchanged (60/60 regression, tests green).

### Gated closure — TurboQuant Spike B
The paper's outlier allocation remains a finite second test, now broadened by the 2026-07-12
audit to include K-high/V-low, boundary-layer, and QJL ablations. It no longer leads the KV
queue: fused compressed-domain attention and KVarN/asymmetric affine controls must establish
the speed/quality baseline first. If no bounded recipe beats them at equal effective bytes,
fully shelve TurboQuant. Task: [`2026-07-09-turboquant-spike-b-outlier-channels.md`](task-inbox/2026-07-09-turboquant-spike-b-outlier-channels.md).

**Speculative decoding (PLD) — PROMOTED + GATE-TUNED** (`main`, framework merge `5deb1d0`, gate-tuning feature `bb5b06f`, merge `29cc453`; plan [`2026-07-09-speculative-decoding-pld.md`](superpowers/plans/2026-07-09-speculative-decoding-pld.md), [verdict + 2026-07-11 resolution](superpowers/verdicts/2026-07-09-pld-firstrun.md)). The **first decode-speed multiplier** beyond the GPU-bound base loop (we're at Zig parity on the base loop; PLD is the first of mlx-serve's multipliers to land). Prompt-lookup decoding is **exact** with no draft model: byte-identical 120/120 at temp 0 in both verify modes. Clean-SHA Qwen3-32B-4bit result: **echo/agent +100.5% (28.28 → 56.70 tok/s, 98.3% accept)**, code **+3.2%**, zero-draft prose **+0.1%**. Fallback rounds now preserve the base submit-first pipeline; the gate judges a cold partial window after four samples and waits 32 steps before probing. This **clears the performance gate for a default-on product policy**, but does not itself flip a runtime default: the harness still selects `--spec pld` explicitly and `RunConfig` still defaults to no spec decoder. The framework (`HarnessCore/SpecDecode/` — `SpecDrafter`/`SpecAccept`/`SpecEmit`/`PLDGate`) is what **DSpark/DFlash will reuse**.

**Also shipped since:** the [quality-metrics explainer](reference/quality-metrics-explained.md) + a **published Artifact** (`https://claude.ai/code/artifact/168d9b15-96e7-4f30-babf-b7ea64441438` — a user-facing "how we measure quality" page), and the **dial-as-informed-consent** refinement (platform spec §4: noticeable-but-valuable tiers with quantified loss + a hard garbage floor; PrismML 1-bit captured as a device-tier research candidate in the intake).

**Sol optimization-landscape audit — COMPLETE (2026-07-12).** The full plans/verdicts/
inbox/intake/carry-forward portfolio was reconciled against the current Swift code and
current primary sources. It corrected four status errors (native MTP and prefix/SSD cache are
not implemented here; Python MLX has shipped absorbed MLA; PrismML artifacts are real), put
Qwen3-32B EAGLE-3/DSpark first for execution, and ranked the KVarN/asymmetric storage-quality
gate plus fused compressed-domain KV attention ahead of TurboQuant B. The EAGLE gate has now
executed RED as recorded below. Dated brief:
[`2026-07-12-sol-optimization-landscape.md`](reference/2026-07-12-sol-optimization-landscape.md).

**Qwen3-32B EAGLE-3 Phase 0 — COMPLETE + SHELVED** (clean feature SHA `1a70c4d`; plan
[`2026-07-12-qwen3-32b-eagle3-preflight.md`](superpowers/plans/2026-07-12-qwen3-32b-eagle3-preflight.md),
[verdict](superpowers/verdicts/2026-07-12-qwen3-32b-eagle3-preflight.md)). The full draft
checkpoint and both target shard sets were authenticated. The MLX head passed the pinned
PyTorch/speculators parity gate (cosine `0.9999816`, argmax 100%), but greedy output changed on
both pairings: 4-bit first mismatch index 17; 8-bit index 7. The 8-bit replay proves an
immediate shape boundary: the same sequential prefix predicts `279` with a one-token target
probe and `264` with `[current,draft]`. 4-bit drift appears only after histories that processed
and rolled back rejected future tokens. Apparent rates are invalid because outputs differ.
No `k=3`/multi-shape bench or Swift port ran. Reopen only for deterministic target verification
or a compatible product-size DSpark/DFlash/MTP checkpoint.

## ▶ Open work queue — pick the next flywheel cycle

Prioritized by the 2026-07-12 Sol audit. The north star remains **match then beat optimized
mlx-serve**: the base loop is at Zig parity and exact, gate-tuned PLD is the first multiplier.

1. **Continuous batching + decode-first chunked prefill** — the largest remaining service
   throughput multiplier (Zig prior ~2.8× aggregate 1→8). Preserve drain-before-batch-join;
   speculation stays off in the batched arm. [Task](task-inbox/2026-07-12-continuous-batching-chunked-prefill.md).
2. **KVarN K4V2 + asymmetric affine/KVTuner storage-quality gate** — the strongest new KV
   candidate; compare actual packed bytes and teacher-forced quality before Metal investment.
   [Task](task-inbox/2026-07-12-kvarn-kv-frontier.md).
3. **Fused compressed-domain KV attention for the selected format** — stop materializing the
   full cache before attention; prove an end-to-end 32K/128K win, not just a Metal
   microbenchmark. [Task](task-inbox/2026-07-12-fused-compressed-kv-attention.md).
4. **Exact prefix/session cache + request-start stack** — restore the incumbent's agent-loop
   TTFT path (hot cache, positive commit, eager warmup, template/tokenize cache; SSD later).
   Design over batching/cache ownership. [Task](task-inbox/2026-07-12-exact-prefix-session-cache.md).
5. **Absorbed MLA** — exact 71× reduction versus the current expanded DeepSeek-V3 cache;
   Python MLX now ships it and pinned Swift GLM code is a second oracle.
   [Task](task-inbox/2026-07-09-absorbed-mla-kv-cache.md).
6. **Sampled-generation foundation → sampler fusion** — separate from batching; define RNG
   and distribution contracts before porting the Zig L1/L3/L1b/L3b stack.
   [Task](task-inbox/2026-07-12-sampled-generation-sampler-fusion.md).
7. **Learned/mixed weight-quant sweep** — affine vs official MLX dynamic/DWQ and oQ4e,
   producing ordinary MLX checkpoints for the existing Swift loop.
   [Task](task-inbox/2026-07-12-learned-weight-quant-frontier.md).
8. **Operability** — measured large-prefill capacity + runtime admission control (system-aware
   spec §7); reliability/capacity, not a decode-speed claim.
9. **TurboQuant Spike B closure** — bounded outlier/asymmetry/boundary matrix, then fully
    shelve on a second loss. [Task](task-inbox/2026-07-09-turboquant-spike-b-outlier-channels.md).
10. **Device/workload-specific research** — PrismML Ternary/Bonsai, then EpiCache/KVzip and
    XGrammar after their exact-cache/sampler prerequisites. [Intake](reference/performance-technique-intake.md).

**Blocked/deferred trained speculation:** EAGLE-3 is shelved by the dated exactness verdict;
no compatible Qwen3-32B DSpark, DFlash, or native-MTP checkpoint was runnable. Do not compare
raw Qwen3-8B control tok/s with the product target. Reopen only under the recovery gate in the
verdict. The bounded recovery seed is
[`2026-07-12-shape-stable-spec-verify.md`](task-inbox/2026-07-12-shape-stable-spec-verify.md).

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
Models on llmbench: `~/perf-work/models/` (Qwen3-32B-4bit and Qwen3-32B-8bit staged; no
Qwen3-32B BF16 target). Python: `~/harness-venv` (transformers<5).

## Known Risks / Open Items

- **Long on-box measurement runs** (long-context KL, multi-shape bench) can stall the driving agent in monitor-waits and do not reliably auto-resume. Inspect the SHA-stamped evidence bundle for partial rows before restarting; record the final artifact names, hashes, and harness SHA in `docs/verification-evidence.md`.
- **Model routing:** haiku=scout, sonnet=builder, opus=main/judgment, fable=deep-reasoner; always pass `model` explicitly; re-route UP one tier on failure. Engine/MLX-coupled work → deep-reasoner (builders have escalated on it).
- **cacheLimit invariant:** whenever `iogpu.wired_limit_mb` is raised, set an explicit `Memory.cacheLimit` (never the 1.5× default — the "7K wall" mechanism). Policy: `CapacityModel.recommendedCacheLimitBytes`.
- **Performance/flywheel backlog:** the ranked queue above is authoritative for optimization
  cycles; the dated Sol brief contains the complete active/deferred/rejected performance
  disposition. The broader platform work remains required by the platform spec: serving/API
  driver, conformance, task-benchmark layer, soak/recovery, and remaining memory controls.
  Do not infer implementation from an upstream dependency: native MTP and prompt caching
  remain unwired in fast-mlx.
- **Content-library practice:** after each notable spike/optimization, write a `docs/content/` piece (blog/whitepaper source). 9 pieces so far ([`docs/content/README.md`](content/README.md) indexes them).

## Commit / Checkin

Docs → `main`. Features → branch, merge `--no-ff` after verification. Commit messages end with a `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer. Never commit secrets, runtime cache paths, or machine-local state.
