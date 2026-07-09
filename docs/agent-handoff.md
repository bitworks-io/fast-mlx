# Agent Handoff

Last reviewed: 2026-07-09

For the next Codex/Claude Code/human agent. Decision-focused; links to the durable specs rather than duplicating them.

## Project Purpose

**fast-mlx** — the fastest, most-optimized MLX inference platform for Apple Silicon, superseding the incumbent Zig `mlx-serve`. A Swift engine + macOS app + Python train/research plane + an engine-agnostic conformance/precision-loss harness. The product wedge is the **optimization dial with quantified precision loss** ("dial in speed, see exactly what accuracy you trade"); the moat is the **technique-integration flywheel** (absorb a new inference technique → measure its speed↔quality frontier → promote to a dial tier or shelve with a dated negative result). First production deployment: **Concierge** (bitworks' shopping assistant). Dev/bench box: **M5 Max 128GB** (`llmbench@192.168.1.252`); production target up to M3 Ultra 512GB. Full design: [`docs/superpowers/specs/2026-07-08-fast-mlx-platform-design.md`](superpowers/specs/2026-07-08-fast-mlx-platform-design.md).

## Current state (2026-07-09)

**Shipped to `main`:**
- **Engine seed** (`spike/`): compiled decode core (`CompiledMLXDecoder` + `CompiledKVCache`) at 155.4 tok/s (≥ Zig), single-owner `InferenceActor`, Swift 6 strict-concurrency clean.
- **The harness spine** (`HarnessCore`, pure/Foundation): equivalence+engagement+acceptance triad, teacher-forced KL + perplexity + long-context tail-p95, versioned corpus (incl. a 24,151-token entry), provenance/JSONL. Established baseline: **4-bit affine KV vs bf16 = tail-p95 1.665 nats @24K, ppl +21.4%**.
- **System-aware context operability**: the per-arch KV **capacity memory model** (`HarnessCore/CapacityModel.swift` — dispatches KV/token by `model_type`; naive formula is wrong 4×–71× for hybrid-linear/SWA/MLA/Mamba2), the `SystemProfiler` (real host introspection, MLX-free), and the **`fastmlx-capacity` CLI** (`swift run fastmlx-capacity [--box …]`). Spec: [`2026-07-09-system-aware-context-operability.md`](superpowers/specs/2026-07-09-system-aware-context-operability.md). Catalog (context caps + memory + Nemotron-3-Ultra/Ornith resolved) in platform spec §9.

**In flight — `feat/turboquant` branch (NOT merged):** implementing Google TurboQuant KV-quant (the flywheel's first novel technique) per [`docs/superpowers/plans/2026-07-09-turboquant-kv-quant.md`](superpowers/plans/2026-07-09-turboquant-kv-quant.md).
- **Phase 1A + 1B COMPLETE + reviewed** (commits `7f85f67`→`e094c60`): `LloydMaxCodebook` (pure, matches paper constants); the MLX quantizer core (`SpikeCore/TurboQuant/{TurboQuantParams,TurboQuantCodec}.swift`) — Haar Π + Lloyd-Max LUT + QJL residual + `TurboQuantTier{tqB2,tqB3}`. **Spike A (make-or-break) PASSED**: the codec reproduces the paper's Theorem-2 distortion table (d·D_prod 0.175/0.0514 vs paper 0.18/0.047), unbiasedness slope 0.9959, correlated-regime margin 2.855×. 13 SpikeCoreTests green on-box. Spike A resolution recorded in [`docs/reference/turboquant-algorithm.md`](reference/turboquant-algorithm.md#spike-a-resolution).

### ▶ RESUME HERE — TurboQuant Phase 2 (next)
On `feat/turboquant`, per the plan's Phase 2/3:
1. **Task 6a (do first — flagged by Phase 1B):** the codec is proven for **unit-norm** inputs; real K/V vectors are not unit-norm. Extend the codec to store per-vector `‖x‖` and normalize before quantize / rescale on dequant (paper §1.1 approach), with a non-unit-norm round-trip test.
2. **Task 6:** `TurboQuantKVCache` — store `(idx, signs, γ, ‖x‖)` per token, dequant-on-read → materialize K/V (materialize-then-attend); parallel `CompiledKVCache`, keep fixed-shape/chunked buffers, apply the 8 GiB `Memory.cacheLimit` bound.
3. **Task 7:** wire into the decode path behind the `tqB2`/`tqB3` tier; lossy-triad equivalence on a **real Qwen3-32B checkpoint** (non-crash + short-prefix + canary + engagement-delta).
4. **Task 8 (Phase 3):** measure `tqB2`/`tqB3` KL/ppl/tail-p95 vs bf16 on corpus v2 + KV bytes/token; **promote to a dial tier iff it beats the 1.665@24K baseline at smaller KV size, else shelve** with a dated negative result. Then a content-library piece.
- Route Phase 2+ to **deep-reasoner (fable)** — engine/MLX-coupled. Fresh session recommended (this one is long).

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

- **TurboQuant norm handling** (Phase 2 Task 6a, above) — the one correctness gap before the cache is built.
- **Model routing:** haiku=scout, sonnet=builder, opus=main/judgment, fable=deep-reasoner; always pass `model` explicitly; re-route UP one tier on failure. Engine/MLX-coupled work → deep-reasoner (builders have escalated on it).
- **cacheLimit invariant:** whenever `iogpu.wired_limit_mb` is raised, set an explicit `Memory.cacheLimit` (never the 1.5× default — the "7K wall" mechanism). Policy: `CapacityModel.recommendedCacheLimitBytes`.
- **Backlog** (`docs/task-inbox/`): absorbed-MLA KV cache (71× DeepSeek-R1 lever, unbuilt); chunked-prefill capacity measurement gate; runtime admission control.
- **Content-library practice:** after each notable spike/optimization, write a `docs/content/` piece (blog/whitepaper source). 4 pieces so far.

## Commit / Checkin

Docs → `main`. Features → branch, merge `--no-ff` after verification. Commit messages end with a `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer. Never commit secrets, runtime cache paths, or machine-local state.
