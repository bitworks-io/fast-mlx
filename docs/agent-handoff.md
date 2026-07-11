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

**TurboQuant KV-quant — COMPLETE + SHELVED** (merged to `main`, `6e82e2b`; plan [`2026-07-09-turboquant-kv-quant.md`](superpowers/plans/2026-07-09-turboquant-kv-quant.md)). The flywheel's first novel technique ran the full loop: Google TurboQuant (arXiv:2504.19874) built exactly (`LloydMaxCodebook` + `TurboQuantCodec` Haar/LUT/QJL — Spike A verified against the paper's Theorem-2 distortion table), integrated as `TurboQuantKVCache` (materialize-then-attend, behind the `tq2.5`/`tq3.5` flag), measured, and **SHELVED as a dated negative result** — uniform-v1 loses to 4-bit affine on Qwen3-32B (tqB3 tail-p95 1.797/ppl +32.6% vs 1.665/+21.4%; tqB2 catastrophic). Verdict: [`verdicts/2026-07-09-turboquant-firstrun.md`](superpowers/verdicts/2026-07-09-turboquant-firstrun.md). The fp16 default path is unchanged (60/60 regression, tests green).

### ▶ Gated next step — TurboQuant Spike B (outlier channels)
The paper's near-losslessness depends on outlier channels (32ch@3b+96@2b) that uniform-v1 deferred — the one lever that could move tqB3 under the baseline. Extend `TurboQuantCodec`'s bit allocation, re-measure through the same harness, then promote or fully shelve. The codec + cache + tier plumbing are already in place behind the flag. Task-inbox: [`2026-07-09-turboquant-spike-b-outlier-channels.md`](task-inbox/2026-07-09-turboquant-spike-b-outlier-channels.md). Route engine/MLX work to deep-reasoner (fable).

**Speculative decoding (PLD) — IN FLIGHT on `feat/spec-decode`** (plan [`2026-07-09-speculative-decoding-pld.md`](superpowers/plans/2026-07-09-speculative-decoding-pld.md)). The first decode-speed multiplier beyond the GPU-bound base loop — we're at Zig parity on the *base* loop, but all of mlx-serve's multipliers (spec-decode, batching, sampler fusion) are still absent. PLD is exact (byte-identical at temp 0), needs no draft model, and is the framework DSpark/DFlash will reuse.
- **Tasks 1–5 DONE + verified** (`36f61bb`→`97f7cfb`): the pure framework (`HarnessCore/SpecDecode/` — `SpecDrafter`/`PromptLookupDrafter`, `SpecAccept.walk` exact accept-walk, `PLDGate`) + the engine integration (`CompiledKVCache.truncate` KV rollback; the batched verify-forward + PLD decode loop in `CompiledMLXDecoder`; `SpecDecodeConfig`). Off-box HarnessCore 108 + on-box SpikeCore 20 tests, 0 failures.
- **▶ RESUME HERE — Tasks 6–7:** (6) wire a `--spec pld` flag through `RunConfig`/driver so `bench` records tok/s spec-on/off and `verify` asserts equivalence; **(equivalence gate)** PLD-on must be **byte-identical** to PLD-off at temp 0 on Qwen3-32B-4bit — STOP if not (correctness bug); (7) bench tok/s on echo/agent + code + prose shapes → promote/shelve + verdict + content. Route to deep-reasoner (fable), on-box. **Honor the Phase-1 flags: `gate.record()` every step; bound the drafter's backward scan for long contexts** — Task 5 was committed just before an infra auth error interrupted it, so confirm it handled those before measuring.

**Also shipped since:** the [quality-metrics explainer](reference/quality-metrics-explained.md) (+ a published Artifact), and the **dial-as-informed-consent** refinement (platform spec §4: noticeable-but-valuable tiers with quantified loss + a hard garbage floor; PrismML 1-bit captured as a device-tier research candidate in the intake).

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
