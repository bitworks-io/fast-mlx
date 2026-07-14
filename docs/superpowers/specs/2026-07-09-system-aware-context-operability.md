# fast-mlx — System-Aware Context Operability Spec

- **Status:** DRAFT for owner review — **§7 in-scope items landed on `main` (2026-07-09)**: the §2 memory model (spine), §3 host profiler, §3.1 cacheLimit policy, and the §4/§5 tunable+advisor logic ship as `HarnessCore` + `SystemProfiler` + the `fastmlx-capacity` CLI, verified on-host. Consuming surfaces (admin API, tooltips, macOS app) + the named backlog remain.
- **Date:** 2026-07-09
- **Owner:** brian@bitworks.io
- **Parent spec:** [`docs/superpowers/specs/2026-07-08-fast-mlx-platform-design.md`](2026-07-08-fast-mlx-platform-design.md) (§4 dial, §5 eval loop, §9 catalog)
- **Research inputs (this session):** catalog context caps + system introspection + large-context mitigations; the ["7K wall" content piece](../../content/2026-07-09-the-wall-that-wasnt.md) (the allocator-hoarding incident this feature exists to prevent in production).
- **Confidence convention (load-bearing):** ✅ = read from a primary source (HF `config.json`, official docs, or vendored Swift source) this session; ⚠️ = inferred / secondary / not independently re-derived. **These markers are carried verbatim from research into every table below — do not launder ⚠️ into a clean number.**

---

## 1. User story + acceptance criteria

**Story.** An operator deploying fast-mlx on a specific Mac (128GB M5 Max bench box → 512GB M3 Ultra production) sets a context-length ceiling for a model and needs the tool to (a) default sanely (32K), (b) allow raising it up to the model's true maximum, and (c) **tell them, before they commit, whether *this box* can actually hold that** — and if not, name the binding constraint and the mitigation. The same advice must extend to the other knobs that consume memory (concurrency, KV-quant tier, model choice), not context alone.

**Acceptance criteria (observable):**
- A1. On startup the engine introspects the host (chip, P/E cores, total RAM, GPU wired limit, current GPU usage, internal-vs-external/NVMe disk, free space) and logs a structured **system profile**.
- A2. For a `(model, context, concurrency, KV-quant tier)` request, a **capacity model** predicts peak footprint (weights + KV + transient prefill + allocator headroom) and classifies it **green / yellow / red** against the box's available headroom — using a **per-architecture** KV formula, not one formula for all models.
- A3. Context default is **32K**, admin-tunable up to `min(model native max, what the hardware can hold)`. For a model whose native max is **below** 32K (Phi-4 = 16K), the effective default is the native max, not 32K.
- A4. When a setting is yellow/red, the operator sees a **hint that names the binding constraint** (physical RAM? the wired limit? the model's own max? a lossy tier needed?) and the **highest-leverage mitigation** (drop KV to a lossy tier / reduce concurrency / pick a lighter-footprint model / raise the wired limit).
- A5. The catalog records, per model: native max context, license (✅/⚠️), per-arch KV/token, and 32K/128K/max memory-fit on 128/256/512GB.

**Proof method.** The capacity model (A2, the spine) is **pure Swift logic in `HarnessCore`, TDD**, verified against the research's confirmed per-model numbers (e.g. Qwen3-30B 3.0 GiB @32K; Gemma-3-27B 2.91 GiB @32K; DeepSeek-R1 152.5 GiB @32K as-implemented). Introspection (A1) is verified by running the profiler on `llmbench` and diffing against `environments.md`'s known topology. The advisor/tunable/app surfaces (A3–A4) are **spec** — they consume the model but don't exist yet (see §7 scope).

---

## 2. The spine — a per-architecture KV memory model (first-class, built now)

This is the one place correctness lives. A single formula is **wrong for 5 of 14 catalog models** (measured against real configs + the vendored Swift arch code) — off by 4×–71×. So the model **dispatches the KV-bytes/token formula by `model_type`** (+ layer-type pattern), exactly as the Swift arch layer already dispatches *cache type* per layer (`MambaCache` vs `KVCacheSimple` vs `RotatingKVCache`).

**Independent variables:** `(model, context, concurrency, KV-quant tier)`. "Under load" = **concurrency**: KV is per-sequence, so N streams multiply the KV term. **Terms of peak footprint:**

```
peak ≈ weights(fixed)  +  N_concurrent × KV(model_type, context, kv_quant)
        +  transient_prefill_peak(context, chunk)      # least-characterized — see §6/backlog
        +  allocator_cache_headroom                     # bounded by an explicit Memory.cacheLimit (§3)
```

Dropping the transient-prefill term is exactly the mistake that killed processes at the 7K wall — keep it, even if its coefficient is initially a measured upper bound rather than a derived constant.

### 2.1 KV-per-token dispatch by architecture class

| Class | `model_type` (catalog members) | KV/token rule | Vendored-source basis |
|---|---|---|---|
| **uniform-GQA** | Qwen3 dense/MoE, Llama, Mistral, GLM-4.5-Air/4.6, Phi-4, Qwen3-235B | `n_layers × n_kv_heads × head_dim × 2 × bytes` — every layer grows | ✅ standard `KVCacheSimple`/`StandardKVCache` |
| **hybrid-linear** | `qwen3_next`/`qwen3_5_moe` (Qwen3.6-35B-A3B, Ornith) | only `n_layers ÷ full_attention_interval` layers grow (GQA); rest are GatedDeltaNet → **fixed** recurrent state | ✅ `Qwen35.swift` `newCache()`: `MambaCache` for linear layers, `KVCacheSimple` for the 1-in-`interval` attention layers |
| **interleaved-SWA** | `gemma3` (Gemma-3-27B) | `n_global × kv × hd × 2 × bytes × ctx` **+** `n_local × kv × hd × 2 × bytes × window` (local capped at window=1024) | ✅ `Gemma3Text.swift` `newCache()`: 5 `RotatingKVCache` : 1 `StandardKVCache` |
| **MLA-as-implemented** | `deepseek_v3` (DeepSeek-R1) | decompressed **per-head** cache (128 heads × (qk_rope 64 + qk_nope 128 + v 128)) — **huge**. *Absorbed*-MLA (cache only `kv_lora_rank 512 + qk_rope 64`) is 71× smaller but **unbuilt** (§7 backlog) | ✅ `DeepseekV3.swift` decompresses via `kv_b_proj` **before** the cache write |
| **hybrid-Mamba2+MoE** | `nemotron_h` (Nemotron-3-Ultra) | only "select" attention layers grow (`n_attn × kv × hd × 2 × bytes × ctx`); Mamba2 = O(1) SSM state; MoE-FFN = no attention. ⚠️ **attention-layer count unconfirmed — do not multiply blind** | ✅ `NemotronH.swift` + `NemotronHTests.swift` exercise Mamba/Attn/MoE block mixes |
| **novel-compressed** (**out of scope**) | `deepseek_v4` (DeepSeek-V4-Flash) | CSA/HCA, per-layer `compress_ratios` — **not MLX-servable today** (ds4/GGUF only). The tunable does not apply; see §8 | ✅ no `DeepseekV4.swift` in the vendored `mlx-swift-lm` rev |

### 2.2 Per-model data (bf16 KV, GiB=1024³, effective per §2.1)

| Model | Native max ctx | `model_type` class | KV/tok (eff.) | KV @32K | KV @max | License |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-2507 | ✅ 262,144 (⚠️ ~1M via opt-in YaRN) | uniform-GQA (48L) | 96 KiB | 3.0 GiB | 24.0 GiB | ✅ Apache-2.0 |
| Qwen3.6-35B-A3B | ✅ 262,144 | hybrid-linear (10 of 40L) | **20 KiB** (naive 80) | 0.625 GiB | 5.0 GiB | ⚠️ verify |
| GLM-4.5-Air | ✅ 131,072 | uniform-GQA (46L) | 184 KiB | 5.75 GiB | 23.0 GiB | ✅ MIT |
| Gemma-3-27B | ✅ 131,072 | interleaved-SWA (10 global + 52 local) | 80 KiB/tok + 0.406 GiB fixed | 2.91 GiB | 10.41 GiB | ⚠️ Gemma (legal sign-off) |
| Qwen3-32B (dense) | ✅ 40,960 field / ⚠️ 32,768 "native" | uniform-GQA (64L) | 256 KiB | 8.0 GiB | 10.0 GiB @40,960 | ✅ Apache-2.0 |
| Llama-3.3-70B | ✅ 131,072 (default NTK) | uniform-GQA (80L) | 320 KiB | 10.0 GiB | 40.0 GiB | ⚠️ Llama Community (<700M MAU) |
| Mistral-Small-3.2-24B | ✅ 131,072 | uniform-GQA (40L) | 160 KiB | 5.0 GiB | 20.0 GiB | ✅ Apache-2.0 |
| **Phi-4-14B** | ✅ **16,384 — hard ceiling** | uniform-GQA (40L) | 200 KiB | **N/A (<default)** | 3.125 GiB @16K | ✅ MIT |
| Qwen3-235B-A22B | ✅ 40,960 field / ⚠️ 32,768 native (⚠️ Instruct-2507 262,144) | uniform-GQA (94L) | 188 KiB | 5.875 GiB | 7.34 GiB @40,960 | ✅ Apache-2.0 |
| **DeepSeek-R1** | ✅ 163,840 | **MLA-as-impl (61L)** | **4.88 MiB** (compressed 68.6 KiB, 71× — unbuilt) | **152.5 GiB (!)** / 2.14 GiB absorbed | 762.5 GiB / 10.73 GiB | ✅ MIT |
| GLM-4.6 | ✅ 202,752 | uniform-GQA (92L) | 368 KiB | 11.5 GiB | 71.16 GiB | ✅ MIT |
| DeepSeek-V4-Flash | ✅ 1,048,576 | novel-compressed — **out of scope** | not derived (ds4-only) | — | — | — |
| **Nemotron-3-Ultra** | ✅ 262,144 (⚠️ "1M" = training recipe, not shipped default) | hybrid-Mamba2+MoE | ⚠️ 1 KiB/tok/attn-layer, **attn-layer count unconfirmed** | not derived | not derived | ⚠️ OpenMDW-1.1 (legal sign-off) |
| **Ornith-1.0-397B** | ✅ 262,144 | hybrid-linear (`qwen3_5_moe`) | ⚠️ ~ (interval assumed) | 0.9375 GiB | 7.5 GiB | ⚠️ MIT (confirm at pull) |

### 2.3 Weights + KV @32K memory-fit (the two terms the advisor needs)

| Model | Weights (4-bit ⚠️ 0.5B/param rule) | +KV@32K | 128GB | 256GB | 512GB |
|---|---|---|---|---|---|
| Qwen3-30B-A3B-2507 | 15.25 | 3.0 | ✓ | ✓ | ✓ |
| Qwen3.6-35B-A3B | 17.5 | 0.625 | ✓ | ✓ | ✓ |
| GLM-4.5-Air | 53 | 5.75 | ✓ | ✓ | ✓ |
| Gemma-3-27B | 13.5 | 2.91 | ✓ | ✓ | ✓ |
| Qwen3-32B | 16 | 8.0 | ✓ | ✓ | ✓ |
| Llama-3.3-70B | 35 | 10.0 | ✓ (tight) | ✓ | ✓ |
| Mistral-Small-24B | 12 | 5.0 | ✓ | ✓ | ✓ |
| Phi-4-14B | 7 | 3.125 @16K | ✓ | ✓ | ✓ |
| Qwen3-235B-A22B | 117.5 | 5.875 | ✗ | ✓ | ✓ |
| **DeepSeek-R1** | ⚠️ 335.5 | **152.5 (as-impl)** | ✗ | ✗ | ✗ **(needs absorbed-MLA even @512GB)** |
| GLM-4.6 | ⚠️ ~177.5 | 11.5 | ✗ | ✓ (borderline) | ✓ |
| DeepSeek-V4-Flash | ✅ 86.7 (2-bit, ds4) | — | ✓ (ds4, not MLX) | ✓ | ✓ |
| Nemotron-3-Ultra | ~275 (NVFP4) | — | ✗ | ✗ | ✓ |
| Ornith-397B | 198.5 | 0.94 | ✗ | ✓ | ✓ |

---

## 3. Startup system introspection

On boot the engine builds a **system profile** (logged, and fed to the capacity model). All APIs ✅ confirmed against Apple docs or the vendored mlx-swift source unless marked.

| Field | Swift API | Notes |
|---|---|---|
| Chip / arch | `MTLDevice.name` / `.architecture.name` | mlx-swift `GPU.deviceInfo()` already uses this |
| P/E core counts | `sysctlbyname("hw.perflevel0.physicalcpu")` / `hw.perflevel1.physicalcpu` | undocumented-but-standard MIB |
| Total RAM | `ProcessInfo.processInfo.physicalMemory` (or `hw.memsize`) | mlx-swift uses `HW_MEMSIZE` |
| GPU wired **limit** | read `sysctlbyname("iogpu.wired_limit_mb")` (unprivileged); **write = root** (LaunchDaemon to persist) | `0` = system default (~66–75% RAM), *not* unlimited |
| GPU **current usage** | `MTLDevice.currentAllocatedSize` / `MLX.Memory.snapshot()` | live number, compared to the ceiling for headroom |
| Recommended working set | `MLX.GPU.deviceInfo().maxRecommendedWorkingSetSize` | don't hand-roll — mlx-swift wraps `recommendedMaxWorkingSetSize` |
| MLX allocator ceiling | `MLX.Memory.cacheLimit` / `.memoryLimit` | **already in use** in the harness (`8 << 30`) |
| Disk internal/external + free | `URLResourceKey.volumeIsInternalKey`, `.volumeAvailableCapacityForImportantUsageKey` | for the SSD-KV-paging decision |
| NVMe vs external interconnect | IOKit walk → `kIOPropertyMediumTypeKey` / `kIOPropertyPhysicalInterconnectTypeKey` | **hard** (IORegistry + DiskArbitration); ⚠️ re-verify constant spellings vs current SDK |
| Process mem headroom | `os_proc_available_memory()` | ⚠️ needs a tiny C/ObjC shim (not in the Darwin Swift module) |
| Memory-pressure events | `DispatchSource.makeMemoryPressureSource(.all, queue:)` → `.warning`/`.critical` | the runtime safety hook (§6) |

### 3.1 The cacheLimit invariant (production, not a hint)

The three memory levers are distinct and causally chained: **`iogpu.wired_limit_mb` (system ceiling)** → `MLX.Memory.memoryLimit` **defaults to 1.5× that** → `cacheLimit` defaults to `memoryLimit`. This exact chain is the **"7K wall" mechanism**: raising the sysctl to 115GB on `llmbench` silently entitled MLX's buffer cache to hoard ~all of it, and an O(context²) allocator pattern turned that into an OOM.

> **INVARIANT:** whenever the engine (or an operator) raises `iogpu.wired_limit_mb`, it **must** set an explicit `Memory.cacheLimit` sized for serving — never leave the 1.5× default in force. This is required, not advisory.

⚠️ **Do not** build any decision on the newer `WiredMemoryManager` ↔ `iogpu.wired_limit_mb` relationship — the research could not confirm whether `mlx_set_wired_limit` drives the same OS lever as the sysctl. Treat `WiredMemoryManager` (ticket admission / hysteresis) as **backlog** (§7), pending a source-level confirmation.

---

## 4. The context-length tunable

- **Default 32K.** Admin-tunable per model.
- **Effective default** = `min(requestedDefault=32768, model.nativeMax)`. Phi-4 (16,384) therefore defaults to 16K — the UI must not assume every model clears the 32K floor.
- **Ceiling** = `min(model.nativeMax, hardwareHolds(model, kv_quant, concurrency))`. On the 128GB bench box "up to model max" is **frequently physically impossible** (a 262K context on a uniform-GQA 70B is tens of GiB of KV on top of weights) — so the true ceiling is what the hardware holds, and **the advisor's job is to name which of the two bounds is binding** (§5). That naming *is* the feature's value, not a caveat.
- **Extension provenance matters.** Some maxes are shipped defaults (Llama-3.3 NTK, DeepSeek-R1 YaRN-40, Gemma-3 RoPE-rescaled); others are ⚠️ opt-in YaRN edits (Qwen dense/235B "→131K/1M"). The catalog flags which, so raising the ceiling past the *shipped* max surfaces a "requires a config edit + re-validation" note rather than silently promising it.

---

## 5. The capacity advisor (generalized — "any tunable", per the brief)

Not context-only. A **capacity advisor** takes a candidate `(model, context, concurrency, kv_quant)` config, runs the §2 memory model, and classifies against headroom. Any pressure-relevant knob queries it.

- **Headroom** = `hardwareHolds` = `min(wired_limit_effective, physicalMemory) − weights − OS/other reserve`. (`hardwareHolds` is the same function §4's ceiling uses.)
- **Classification** (thresholds are **config**, not magic numbers — defaults shown):
  - 🟢 **green** — predicted peak ≤ ~70% headroom: fits with slack for allocator churn + the transient term.
  - 🟡 **yellow** — ~70–90%: fits but fragile (a concurrency spike or a long prompt can tip it); advisor names the nearest mitigation.
  - 🔴 **red** — >90% or > headroom: will not hold; advisor names the **binding constraint** + the highest-leverage fix.
- **Binding-constraint naming** (the differentiator): the advisor reports *why* — `physical RAM` / `wired limit (raise it, root)` / `model native max` (can't go higher) / `native max < 32K` (Phi-4) / `MLA-as-implemented` (DeepSeek-R1: the KV itself is the wall) — and the **ranked mitigation** from §6 (usually: drop KV to a lossy tier → reduce concurrency → pick a lighter-footprint model).
- **Reuses existing signals:** `MLX.Memory.snapshot()` for live usage and `DispatchSourceMemoryPressure` for runtime escalation — the advisor is the *predictive* front end; the pressure source is the *reactive* backstop (§6.5).

---

## 6. Mitigations under load (ranked by leverage)

Discriminator: does the lever cut **steady-state KV/token** (extends what fits), bound **transient peak** (safely reach what already fits), or provide **safety** (prevents crashes)?

| # | Mitigation | Cuts | Feasibility | Disposition |
|---|---|---|---|---|
| **1** | **KV-cache quantization** (8-bit / turbo4; TurboQuant 2-bit deferred) | steady-state, 2×–4× | ✅ proven in Zig; Swift/MLX port is shaped engineering | **Primary lever.** The dial's KV-quant tier *is* the main context-extender. TurboQuant impl is its own plan. |
| **1b** | **Absorbed/compressed MLA cache** (DeepSeek-class) | steady-state, **71×** for R1 | ⚠️ unbuilt in MLX/mlx-swift; real lift | **Backlog (gated).** The only thing that makes R1 viable at long context. Out of scope for *this* feature. |
| **2** | **Architecture-aware model selection** (favor hybrid-linear / SWA for long-context tiers) | steady-state, 4×–6× | ✅ free — correct accounting (§2) | **In scope** — the advisor surfaces it ("Qwen3.6-35B holds 32K in 0.6 GiB vs Qwen3-32B's 8 GiB"). |
| **3** | **SSD/NVMe KV paging** | steady-state, **conditional** | ⚠️ omlx's is a **cross-request prefix cache** (agentic reuse), *not* a mid-generation max-context extender; mid-gen eviction at 10–50ms/step is unsolved. Distinct from weight-streaming (ds4). | **Backlog.** Valuable for the agentic/repeated-prefix profile (Concierge), not a drop-in context extender. |
| **4** | **Chunked prefill** | **transient peak only** — per vLLM's own docs it does *not* extend capacity | ✅ cheap; already a §5 commitment | **Reframe + gate** (see parent-spec §5 edit). Serving path already prefilled 32K *unchunked*; fused SDPA avoids the O(L²) driver. **Measure peak at 64K/128K/262K before investing.** |
| **5** | **Runtime memory-pressure response** (`Memory.cacheLimit` bound, `DispatchSourceMemoryPressure`, evict cache on `.warning`) | safety only | ✅ already partly in use | **In scope** — the reactive backstop; foundational, not a multiplier. |

*(Attention-sink / StreamingLLM retention is **not** implemented in any catalog model — only Gemma-3 has true SWA. arXiv:2309.17453 is the citable source if ever added as a new capability.)*

---

## 7. Scope line + backlog gates

**In scope now — LANDED on `main` 2026-07-09:**
1. ✅ The **§2 per-arch KV memory model** — pure `HarnessCore` `CapacityModel`/`ModelArchProfile`/`SystemProfile`, 98 tests, verified vs confirmed numbers (merge `7f9c088`).
2. ✅ The **§3 system-introspection profiler** — `SystemProfiler.probe()` (chip/cores/RAM/wired-limit/GPU-alloc/disk via Metal+sysctl, MLX-free), on-host verified (merge `2cdef35`). *Remaining increments:* NVMe/interconnect (IOKit) + `os_proc_available_memory` shim — left as marked TODOs.
3. ✅ The **§3.1 cacheLimit invariant** — pure `recommendedCacheLimitBytes` policy (merge `2cdef35`). Continuous-service loaders and benches now set and record an explicit 8 GiB MLX cache limit; *remaining:* apply the advisor-selected value at the production engine startup on every wired-limit raise.
4. ✅ The **§4 context tunable** + **§5 capacity-advisor** logic — `effectiveDefaultContext`/`contextCeiling`/`classify` (green/yellow/red + binding constraint) exposed via the **`fastmlx-capacity` CLI** (live host + `--box` planning). *Remaining:* the admin API / tooltips / macOS-app surfaces that consume this — deferred with those components.
5. ✅ The **catalog update** (§2.2/§2.3 → parent §9, merge `520ff99`).

**Named backlog (with gates):**
- **Runtime admission control** — the dense-Qwen3 continuous runtime now atomically reserves
  measured KV geometry in bytes, including allocation rounding, per-row metadata, and a
  conservative five-copy membership-transition envelope; it releases the reservation on every
  removal path. This closes admission for that explicit runtime, not for the whole engine.
  General architecture-aware host admission, `WiredMemoryManager` tickets/hysteresis, and the
  production API remain backlog. *Gate:* confirm `mlx_set_wired_limit` ↔ sysctl relationship
  from MLX core source first (§3.1), then define non-dense state envelopes independently.
- **Absorbed-MLA caching** — the 71× DeepSeek-R1 lever. *Gate:* its own design/plan; measure against the as-implemented baseline through the harness.
- **Chunked-prefill capacity value** — *Gate:* extend `CtxProbe` `generate` to one-shot 64K/128K/262K prefill and **measure the transient peak** before any chunked-prefill capacity claim (parent §5).
- **SSD/NVMe KV paging** — cross-request prefix cache for the agentic profile; separate design.

---

## 8. Honesty cases (each with a disposition)

- **DeepSeek-R1** — as the inherited arch caches K/V today (decompressed MLA), it is **152.5 GiB KV @32K — not viable at long context on any box**, including 512GB, until absorbed-MLA is built. Catalog it that way; do not paper over it with the compressed-theoretical number.
- **Phi-4-14B** — native max **16,384 < 32K default**; `effectiveDefault = min(requested, nativeMax)`. The design must not assume every model clears the default floor.
- **DeepSeek-V4-Flash** — **not MLX-servable** (no `deepseek_v4` in the vendored rev; `mlx-lm` support is an open PR; `mlx-community` weights raise `KeyError`). Served today only via the separate `ds4` engine. **The MLX-based tunable does not apply to it** — a real scoping boundary, stated, not an edge case.
- **Nemotron-3-Ultra** — real (`nemotron_h`, same family as existing support); **plausibly loadable** by existing Swift code (⚠️ pending a Codable-field check for its MoE additions + confirming the block-pattern parser scales). 256K shipped default; ⚠️ "1M" is a training claim. ~275GB NVFP4 → **512GB only**. License **OpenMDW-1.1** → legal sign-off before commercial use.
- **Ornith-1.0** — real (DeepReinforce), `qwen3_5_moe`, **post-trained on Gemma-4/Qwen-3.5**; loads through the existing `Qwen35MoE.swift` registration → **zero net-new engine work**. 9B/35B/397B; 397B ~198.5GB @4-bit (fits 256GB). ⚠️ MIT reported — confirm at pull. Note: 397B *total* params are all resident (active-param count affects decode speed, not memory).
