# Performance run — 2026-07-06

Disciplined, benchmark-gated performance work across three tracks: (A) concurrency, (B) DSpark speculative decoding, (C) an independent review (fable/deep-reasoner) of other missing levers. Every kept change had to pass **byte-equivalence (or a distribution-tolerance test) AND show a measured win** on real hardware; anything that didn't move the needle was dropped, not committed.

All benchmarks ran on the **M5 / 128 GB** target (`llmbench@192.168.1.252`), which was provisioned from bare (Zig 0.16, mlx-c/mlx/webp, full model set) at the start of the run. Measurement harness: `perf_probe.py` (streaming, temp>0-aware, concurrency-aware — it fills the gap that every existing `bench.sh` cell runs `temp=0`, which never exercises the stochastic sampler path production/agent traffic actually takes).

## Results at a glance

| Lever | Track | Status | Measured (gemma-4-e2b-it-4bit, M5) | Gate |
|---|---|---|---|---|
| **L1 batched spec-verify** | C | ✅ committed | temp0.7 PLD echo **+11.8%** (531.73→594.37); greedy PLD **+7.1%** (byte-identical) | byte-equiv PASS + 4 hermetic + win |
| **L3 top-p fast path** | C | ✅ committed | top_p=0.95 sampled decode **+5.0%**; stochastic PLD echo **+10.8%** | hermetic kept-set identity + win |
| **L1+L3 additive** | C | ✅ committed | top_p=0.95 stochastic PLD echo **+19.4%** (466.93→557.73) | byte-equiv PASS on combined binary |
| **L5 batch-over-spec** | A | ❌ rejected by data | PLD-on aggregate (514, serialized) BEATS batched (274–403) on echo at conc 2–4 | would REGRESS the common workload |
| **DSpark drafter** | B | 🟡 milestone-1 | loader **validated on real 8B checkpoint** (64 tensors bind) + backbone forward + 8 tests | perf validation = milestone-2 |
| **L2 kv-quant bandwidth** | C | 📋 scoped | (est. 15–40% decode @16–32K ctx on 27B) | not implemented — see below |
| **Track A chunked-prefill / paged-KV** | A | 📋 scoped | — | not implemented — see below |
| **L4 GDN conv fuse / L7 MoE async prefill** | C | 📋 scoped | (est. 0.5–3%) | not implemented |

## Validated + committed wins

### L1 — Batched speculative-verify accept walk (`perf/l1-batched-spec-verify`)
The stochastic (temp>0) PLD/verify accept walk tested drafts one position at a time — a full sampler chain (incl. a full-vocab `applyTopP` sort) **plus a blocking GPU sync per draft**. For m drafts that's m sorts + m pipeline drains, serialized. `pldAcceptWalk` now runs the chain once over the whole `[1,1+m,V]` verify logits, gathers the m draft probabilities in a single `take_along_axis`+eval, and walks acceptance on the CPU — preserving exact RNG draw order/count and the `accepted` correction index. Greedy stays byte-identical; stochastic preserves the marginal.
- **Bonus bug fixed:** `mlx_topk` (no axis) FLATTENS the whole array — invisible while every `applyTopK` caller passed 2-D `[1,V]`, but a silent shared-mask bug on the new `[1,S,V]` path. Added `mlx_topk_axis` (mlx-c 0.6.0).
- Gate: `tests/test_pld_equivalence.sh` byte-identical + 4 hermetic accept-walk tests + `zig build test` 652/702.

### L3 — Top-p fast path when top-k active (`perf/l3-topp-fast-path`)
`applyTopP` sorted the full `[.,V]` logits (262K on Gemma) every sampled token; but top-k runs first (Gemma default top_k=64), so the nucleus lives inside the k top values `mlx_topk` already produces. `applyTopKTopP` computes the cutoff from those k values, skipping the O(V log V) sort. Used only when both top-k and top-p are active; kept-token SET proven identical to `applyTopK∘applyTopP` (hermetic test, 4 combos); greedy untouched.
- **Measurement lesson (documented):** invisible unless top-p is actually engaged — `top_p=1.0` disables it; must measure at `top_p=0.95`.

### Additive (`perf/integration`)
L1+L3 stacked: **+19.4%** on the primary agent workload (top_p=0.95 stochastic PLD echo), vs +10.8% for L3 alone — the levers compose. The combined binary passes byte-equivalence.

## Rejected by data — L5 (batch-over-spec scheduling)
The fable review proposed suspending PLD when ≥2 slots are active so concurrent requests batch (the existing 1.5×). The M5 concurrency baseline showed the **opposite**: PLD-on aggregate throughput (514 tok/s, flat/serialized) *beats* batched-no-PLD (274 @conc2, 403 @conc4) on the echo workload, because PLD's 2.6× is so strong that even serialized it wins. Blanket-suspending PLD to batch would **regress** the common echo/agent case. Correctly dropped — the benchmark rejected it before any code was written. (It may still help at higher concurrency or on non-echo traffic; that would need a runtime-PLD-yield-gated policy, not a blanket switch.)

## Track B — DSpark
- **Complete 1370-line port spec** at `docs/dspark-port-spec.md` (derived from the real checkpoint's config + safetensors index and the DeepSpec source; two self-caught+re-verified tracing errors documented for the implementer).
- **Milestone-1 code** (`perf/dspark`, `src/dspark.zig`): config parse, loader binding all 64 tensors, the `Qwen3DSparkAttention` backbone forward (non-causal 7-token parallel block, EAGLE-3 multi-layer tap [1,9,17,25,33]), Markov (rank-256) + confidence-head helpers, 8 tests, compiles clean, no regressions.
- **Loader VALIDATED against the real checkpoint** (M5, `DSPARK_TEST_MODEL`-gated): all 64 tensors of `dspark_qwen3_8b_block7` bind with matching shapes (`layers=5, block_size=7, taps=5`). This also caught + fixed a latent `readToEndAlloc` compile bug in `loadDspark` (hidden by Zig's lazy analysis — it was never called through a test path before). Milestone-2 finding: the loader hardcodes the `<dir>/dspark/` sidecar path (mtp.zig convention) but this checkpoint ships flat — a path-contract decision for milestone-2.
- **Milestone-2 (remaining, marked `// MILESTONE 2` in source):** persistent draft KV cache, the Leviathan verify loop (`Generator.nextDspark`), transformer multi-layer raw-hidden capture, scheduler dispatch, `--dspark` flag, and the equivalence + acceptance-floor parity tests against `dspark_qwen3_8b_block7`. **Performance validation of DSpark requires milestone-2** — milestone-1 is the loader+backbone foundation (now proven to load real weights), not yet a runnable drafter.

## Scoped, not implemented (honest status)

- **L2 kv-quant bandwidth (highest remaining single-model upside).** `updateAffine` (transformer.zig:562-567) builds all 6 quant views but then dequantizes to dense and returns dense only — so the fused quant-attention path engages on ~0 layers, and every decode token pays a full K/V dequant (~3× KV traffic). Fix = populate the quant triple in the `DenseKVView` return + engage `quantAttention`. Est. **15–40% decode at 16–32K context** with kv-quant on — the only lever that moves the bandwidth-bound 27B/31B. **Risk:** `mlx_repeat_axis` on the quant triples (kv_quant.zig:529-554) may materialize a copy (needs a Metal-capture check before trusting the win). Bounded, gated experiment; deferred for budget.
- **Track A chunked-prefill / paged-KV.** `runPrefill` (scheduler.zig:3261) prefills the whole prompt before the decode tick → a new arrival freezes in-flight streams (head-of-line blocking). Paged/block KV would replace the dense-pad concat (transformer.zig:4180) and let dense concurrency scale past the measured ~1.94× ceiling. Both are larger scheduler/attention refactors; scoped, not started.
- **L4 GDN conv1d fuse / L7 MoE async prefill eval.** Small (0.5–3%), byte-equivalent/tolerance; scoped.
- Corrections from the fable review (avoided wasted effort): the "GDN prefill 2.8× gap" is stale (measurement artifact; the fused GDN Metal kernel already landed — no reference to catch). Large dense decode is at ~79% of the bandwidth ceiling — only fewer-bytes-moved (L2) can move it, not dispatch reduction.

## Checked in

Branches pushed to `bitworks-io/mlx-serve` (per-change for single value + `perf/integration` for additive):
- `perf/l1-batched-spec-verify`, `perf/l3-topp-fast-path`, `perf/integration`, `perf/dspark`

Artifacts: `BenchmarkLog.md` narrative entries; `docs/perf-csvs/perf-2026-07-06-*.csv` (baseline, L1, L3, integration); `docs/dspark-port-spec.md`; this summary. Full raw logs + CSVs on the M5 under `~/perf-work/`.

## Recommended next steps (ranked)
1. **DSpark milestone-2** (user-requested track): loader load-test → verify loop → acceptance/speedup parity vs the published mlx-dspark numbers (Qwen3-8B ~1.6×). The spec + m1 foundation make this the highest-value next unit.
2. **L2 kv-quant dispatch-hole** — bounded, biggest single-model decode win on the 27B; gate on the Metal-capture materialization check.
3. **Track A chunked-prefill** — the concurrency latency fix (bounded scheduler refactor); paged-KV after.

---

## Update 2 (continuation) — drafter/MTP wins, second review, scale-out plan

**L1b+L3b (committed):** a second fable review found the drafter + MTP stochastic accept walks still ran the per-draft sort+sync loop L1 removed from PLD (the **default-on** paths), and L1's `probsAllPos` never got L3's fusion. Fixed in one patch (with the critical `mlx_topk`→`mlx_topk_axis` switch, red-on-revert tested). gemma-e2b top_p=0.95: drafter echo **+2.7%** (scales with block_size), PLD echo **+6.5%**. **Cumulative L1+L3+L1b+L3b = +27.2%** on stochastic PLD echo vs baseline. All equivalence gates + hermetic tests pass.

**Model-scaling (committed):** validated L1+L3 on production-class Qwen3.6-27B — +1.4% sampled / +3.5% PLD. The sampler wins scale **inversely with model size** (fixed vocab-bound cost — big fraction of a fast 2B decode, small on a bandwidth-bound 27B). Honest framing: these wins are for **fast** models (small dense, low-active MoE, drafters); the big dense 27B/32B wants L2.

**Second fable review — verdict:** L1b+L3b was the missed top lever; beyond it + the already-scoped items, nothing new materially outranks. Two below-the-line MoE trims noted (gate+up pack, GDN qkvz permute; 1–3%).

**L2 correction (still scoped, now de-risked):** fable confirmed the dispatch-hole but corrected the risk — `quantAttention`'s `mlx_repeat_axis` **materializes** (not stride-0), so naive L2 inflates 4-bit K to ~bf16 and the win evaporates. The fix must reshape Q into KV-head groups (pure views), not repeat K. Without this, L2 benchmarks as a wash and gets wrongly rejected. This is the corrected spec for L2 implementation.

**Frontier scale-out (`docs/big-model-scale-out-plan.md`):** DeepSeek-V4, GLM-5.2, Kimi-K2.7-Code verified (raw sources). All DeepSeek-lineage MLA/MoE; **all have DSpark drafters** → Track-B DSpark is strategic future-proofing. DeepSeek-V4-Flash already served via `ds4` (fits the M5 at 2-bit); GLM/Kimi need net-new MLA arch + 370–600 GB cluster HW.

**Upstream PR readiness:** the sampler/spec-decode wins (L1+L3+L1b+L3b) are a complete, tested, benchmark-backed unit on `perf/integration` — PR-ready for `ddalcu/mlx-serve` on your go. L2 (corrected), DSpark-m2, Track-A remain scoped roadmap.
