# Decision Record: fast-mlx Engine Language & Lineage

- **Status:** PROPOSED — recommendation: **Hybrid** (carry Zig engine into fast-mlx + Swift macOS app); awaiting owner approval
- **Date:** 2026-07-07
- **Owner:** brian@bitworks.io
- **Decision:** What engine sits under fast-mlx, and in what language — carry the existing Zig (mlx-serve) engine forward, or clean-room rewrite (Swift/mlx-swift or C++/mlx-core)?

## Context

fast-mlx aims to be a maximally-optimized MLX inference solution for Apple Silicon (up to
M3 Ultra 512GB; dev limited to 256GB), driving commercial portfolio products. Surface: CLI +
OpenAI-compatible API + native macOS app. Stated differentiators: a **tunable optimization
dial with quantified precision loss**, curated model support (incl. Nemotron-H, Qwen, Gemma4,
etc.), and big-memory targeting.

The incumbent baseline is `bitworks-io/mlx-serve` — a **fork of `ddalcu/mlx-serve` (MIT)**,
written in **native Zig against the mlx-c C API**. The bitworks fork's recent perf work
(turbo KV-quant, DSpark port, observability, chunked prefill) is **new (built ~2026-07-06),
unpushed upstream, and may never be**. The user is weighing moving that work into fast-mlx and
retiring the mlx-serve fork vs. a clean-room rewrite in "the most efficient OSX stack language."

Because upstream is **MIT**, none of the options are legally forced: forking/rebranding/closing
and commercial resale are all permitted (attribution only). So the choice is an
engineering/strategy call, not a licensing one.

## Research Findings (confirmed 2026-07-07)

### 1. Google TurboQuant vs. the engine's "turbo" KV-quant

- The engine's `--kv-quant turbo2/turbo4` (`src/kv_quant.zig`) applies a **deterministic
  Sylvester-Hadamard rotation (with per-layer sign flips) → affine group-quant (group 64) at
  2/4-bit → dequant ×Hᵀ on read**; data-free; power-of-two head_dim. **No repo text references
  Google TurboQuant.** This is a **QuaRot-family** scheme ([QuaRot arXiv:2404.00456](https://arxiv.org/abs/2404.00456)),
  built on the QuIP# randomized-Hadamard primitive ([arXiv:2402.04396](https://arxiv.org/abs/2402.04396)).
  The "turboquant" name is an internal label, not an implementation of the paper.
- **Google TurboQuant** ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874); Zandieh et al.,
  Google Research/DeepMind + NYU): an **online, data-free _vector_ quantizer** — random rotation
  → Beta-distributed coordinates → closed-form per-coordinate **optimal scalar quantizer**;
  `_prod` variant adds a 1-bit QJL residual for unbiased inner products. Within ~2.7× of the
  information-theoretic distortion bound. KV-cache results: **quality-neutral at 3.5 bits/ch,
  marginal degradation at 2.5-bit** (Llama-3.1-8B, LongBench-E). NOT a weight method. The
  widely-cited "6×/8× on H100" figures are **unverified vs. the primary source and irrelevant
  to Apple Silicon** (CUDA-throughput, not quality-at-equal-bits).
- **Naming caveat:** "PolarQuant" is a *sibling* Google KV method ([arXiv:2502.02617](https://arxiv.org/abs/2502.02617),
  polar/angular transform) that TurboQuant benchmarks against — it is **cited, not used inside**
  TurboQuant. Secondary sources conflate them; do not.
- **Build verdict (bit-width dependent):** implementing real TurboQuant is a **plausibly
  meaningful upgrade at the 2-bit tier** (optimal non-uniform quantizer + zero per-group metadata
  tax — the current scheme pays ~25% of the bit budget on scale/zero-point at group-64/2-bit +
  inner-product debiasing). At **4-bit it is roughly lateral/redundant** — post-Hadamard data is
  near-Gaussian, uniform affine is already near-optimal, and an Apple-Silicon study
  ([arXiv:2605.05699](https://arxiv.org/abs/2605.05699)) shows near-zero perplexity loss from the
  current family at int4. Cost: TurboQuant's Beta-derived non-uniform quantizer + QJL is materially
  more complex than affine RTN. **Recommendation: prototype for a future 2-bit tier; not needed for 4-bit.**

### 2. mlx-swift capability & performance reality (late 2025 / 2026)

- **Same ceiling as Zig.** mlx-swift binds **mlx-c** — the *same* C API the Zig engine uses
  (per ml-explore/mlx-c + mlx-swift). Custom Metal kernels (JIT `metal_kernel` / `MLXFast`),
  fused `scaledDotProductAttention`/`rmsNorm`/`RoPE`, full affine quant + `QuantizedKVCache`,
  safetensors, lazy graph, streaming — **all confirmed present**. Custom C++ `Primitive`
  subclasses are unavailable in **both** Swift and Zig (symmetric; matters for training, not
  inference). Maintained by Apple (ml-explore); version-parity with core MLX (~0.31.x).
- **Performance is the risk, and it's unproven.** Apple's own `mlx-swift-lm` shows MoE decode
  **7.3× slower** than Python mlx-lm (issue #124: Qwen3.5-35B-A3B @ 11.7 tok/s Swift vs 85 tok/s
  Python) — root-caused to a global `NSRecursiveLock` around `eval()`, synchronous per-token
  GPU→CPU extraction, and no MoE kernel fusion. Third-party **SwiftLM** (native Swift server that
  markets "TurboQuant") benchmarked at **~20 tok/s vs 54 tok/s Python** on Qwen3.6-35B-A3B (~half).
  These are **fixable eval-loop design choices**, not inherent Swift penalties — but **no public
  Swift MLX engine matches Python mlx-lm / vLLM-mlx today**.
- **Concurrency ceiling is language-independent:** core MLX is **not thread-safe for concurrent
  `eval`** (open mlx issues #2133/#2067/#3078) — caps batched/concurrent serving in *any* MLX engine.
- **Architecture head-start:** `mlx-swift-lm` (MIT) already registers ~40 arch types incl.
  `gemma4`, `qwen3_moe`, `qwen3_5_moe`, `mamba2`, `nemotron_h`, `lfm2`, `jamba`, `deepseek_v3`.
  Legal to leverage with attribution (may not satisfy a *strict* clean-room policy).

### 3. Zig engine surface area (re-port anchor)

- **83,338 LOC first-party Zig** across 41 files in `src/`. **~18K LOC is media generation**
  (LTX video 5.9K, Krea/Flux/gen image ~7.2K, Kokoro TTS 2.8K, ltx_audio 1.4K) — **out of scope
  for fast-mlx v1**. The **LLM-serving core is ~45K LOC**.
- Breakdown: transformer core + archs **10.9K** (25 model_types; 4 forward branches:
  standard/MoE/SSM/BERT); continuous-batching scheduler **3.8K**; API servers
  (OpenAI/Anthropic/Ollama/WS) **19.6K**; speculative decoding **4.0K** (PLD/drafter, DSpark, MTP);
  KV-quant + prefix/tokenize cache **2.5K**; tokenizer + JSON grammar/schema/regex **3.3K**;
  generate/sampling **6.1K**; vision towers (SigLIP + Qwen-VL) **2.5K**; model loading/registry **4.7K**.
- **Custom Metal (19 kernels, 11.9K LOC) is confined to the vendored `lib/ds4` path** (DeepSeek-V4
  GGUF engine, from antirez/ds4). First-party `src/` authors **no raw Metal** — it composes mlx-c
  ops + `mlx_fast_metal_kernel` JIT. So the mlx-c ceiling is clearly sufficient for top-tier perf.
- **Thin test corpus:** ~814 dedicated test LOC + ~710 inline test blocks. A rewrite (or carry-forward)
  should treat test/equivalence coverage as net-new investment.
- **Native deps to re-bind:** mlx-c, ds4 (C++), jinja2 (~30K LOC C++, chat templates), llama.cpp
  (GGUF), stb_image, libwebp.

## Options Under Evaluation

1. **Carry Zig engine forward** — relocate/rebrand bitworks work as fast-mlx core; retire fork;
   add product layer (dial, precision-loss quantification, curated models, M3-Ultra) + thin Swift app.
2. **Clean-room Swift (mlx-swift)** — one Apple-native language across engine+CLI+API+app.
3. **Clean-room C++ (mlx-core)** — theoretical ceiling; fragmented app layer.

(A hybrid — Zig/proven engine core + native Swift app + product layer — is being considered explicitly.)

## Quantified Analysis & Recommendation

_(Fable 5 synthesis, 2026-07-07. Team assumption: ~1.5 FTE senior, AI-assisted, reference code in hand. Estimates are ±40%-class.)_

### Performance delta vs the proven Zig baseline

| Path | Decode (steady-state) | Time to perf parity | Status |
|---|---|---|---|
| **Carry Zig forward** | 0% (is the baseline) | 0 | Proven |
| **Clean-room Swift** | Day-1 −15% to −60% dense, up to **−85% MoE**; mature −0–5% | +3–6 mo perf work after functional bring-up | **Unproven** — no public Swift MLX engine reaches Python-mlx parity |
| **Clean-room C++** | Mature ~0% | gated by bring-up | Unproven, low-surprise |
| **Hybrid (Zig core + Swift app)** | 0% | 0 | Proven |

Small fast models (7B/4-bit, 80–120 tok/s) punish a sloppy host loop most — exactly Swift's known failure modes (global lock around `eval()`, sync per-token GPU→CPU pull). Concurrency ceiling is **identical on all paths** (core MLX `eval` not thread-safe); only the Zig path has a proven 3.8K-LOC scheduler engineered around it.

### Re-port effort to feature+perf parity with the ~45K-LOC Zig LLM core

| | Carry/Hybrid | Clean-room Swift | Clean-room C++ |
|---|---|---|---|
| Engine subtotal (engineer-months) | **1.5–3** | **13–24** | **16–27** |
| + Product layer (dial+loss harness, SwiftUI app, CLI) | +4–7 | +3.5–6 | +3.5–6 |
| **Total to v1 (engineer-months → calendar)** | **≈5.5–10 → 4–7 mo** | **≈17–30 → 12–20 mo** | **≈20–33 → 14–22 mo** |

**Rewrite premium: ~11–21 extra engineer-months ≈ $220–520K fully-loaded + 8–13 months' delay — spent to reach a tok/s figure you already have**, with real risk of shipping slower.

### Risk ranking

- **Carry/Hybrid:** Low perf/schedule risk; **High hiring/bus-factor risk** (sole ownership of ~45K Zig LOC, small pool, pre-1.0 toolchain). Mitigate: pin toolchain, document internals, strict API boundary, engine-agnostic conformance harness.
- **Clean-room Swift:** **High perf-parity risk** (nobody has publicly done it); Low hiring risk; Apple maintains arch library.
- **Clean-room C++:** Medium perf risk, highest effort, fragmented app layer.

### Recommendation: **Hybrid** — carry the Zig engine into fast-mlx; Swift only for the app

"Most efficient OSX stack language" is a **category error at the engine level**: Zig/Swift/C++ all drive the identical mlx-c → JIT-Metal path, and the first-party Zig source authors zero raw Metal yet hits top-tier perf — the binding is not the bottleneck; eval-loop/scheduler/quant *design* is, and only the Zig engine provably has it. So both owner instincts are right at different layers: **"move it into fast-mlx and retire mlx-serve" is correct for the engine; "Swift" is correct for the app.** The differentiator (dial + quantified precision loss) is measurement/UX work, engine-agnostic, ~zero tok/s content, and sits unbuilt.

**Insurance:** build the precision-loss harness against the OpenAI API surface (engine-agnostic) — it doubles as a conformance/regression suite that makes a future Swift migration cheap and measured. **Re-evaluate in ~12 months** if Apple closes the mlx-swift-lm eval-loop gap (issue #124) or Zig bus-factor becomes a hiring blocker. Do not rewrite before that trigger.

### fast-mlx v1 scope

- **IN:** LLM text serving, *curated* catalog (~8–12 models: Qwen3/3.5 dense+MoE, Llama-3.x, Gemma4, Nemotron-H, Mamba2, DeepSeek-V3 as the 512GB flagship via vendored ds4); OpenAI-compatible API + CLI (`serve/run/bench/verify`); SwiftUI macOS app (chat, server control, dial UI); **dial v1** = weight-quant preset × KV-quant tier (fp16/8/turbo4) × spec-decode toggle (PLD+DSpark) × chunked prefill, each preset shipping with *measured* loss; 256/512GB big-memory configs.
- **PREVIEW (unmeasured):** Qwen-VL vision (excluded from dial quality claims).
- **OUT:** all media-gen (~18K LOC: LTX video, Flux/Krea, Kokoro TTS — future); Anthropic/Ollama/WS as *supported* endpoints (park code); 2-bit KV tier + MTP spec-decode (experimental flags); training/fine-tuning; non-macOS.
- **OPEN:** owner named **Nemotron 3 Ultra** and **Ornith** for the catalog — MLX availability + memory fit not yet verified (Nemotron-H ≠ "Nemotron 3 Ultra"; "Ornith" unrecognized). Resolve in catalog research.

### TurboQuant call

Defer. The engine's "turbo" is QuaRot-family; at the v1 4-bit tier it is already near-quality-neutral (real TurboQuant is lateral there). TurboQuant's genuine edge is the **2-bit tier** (current scheme burns ~25% of the bit budget on group-64 metadata; TurboQuant is quality-neutral at ~3.5 bits/ch). Schedule a bounded **3–6 week spike** when building the v1.1 2-bit "max-context" KV tier for 512GB long-context; ship only if the harness shows it beating turbo2 on the measured frontier. Name it accurately (do not repeat SwiftLM's conflation).

### The dial (the actual product)

- **Three-layer metric stack per dial point:** (1) median/p95 **KL divergence vs fp16 logits** on a fixed mixed corpus (cheap, most sensitive); (2) **perplexity delta** on held-out text; (3) **task-benchmark deltas** (MMLU-lite, GSM8K, HumanEval, RULER-style long-context) for the headline accuracy number. Speculative decode is *exact* → reported speed-only ("turbo with zero loss").
- **Per-model Pareto frontier:** grid dial coordinates (weight-quant × KV-tier × spec-decode × context), measure tok/s + TTFT on reference M3 Ultra 256/512GB, ship as signed JSON in the catalog; the app renders it — every notch shows real measured numbers.
- **Default "fastest with unnoticeable loss":** auto-select fastest point where KL ≤ ~0.05 median, ppl delta ≤ 1%, task deltas within CI; label tiers Lossless / Balanced ≤1% / Turbo ≤3% / Max-fit.
- **Engine-agnostic by construction** (drives OpenAI API + debug logprobs) → survives an engine swap, doubles as port-conformance suite.
- **Trust:** per-model *and* per-hardware numbers with error bars, re-run in CI on every engine change, published methodology, local `fast-mlx verify` quick mode.

### Executive summary

1. Language won't buy speed — Zig/Swift/C++ share the mlx-c/Metal ceiling; your Zig engine is the only one that provably reaches it (Swift's best public showing is 7.3× behind on MoE).
2. **Do the Hybrid:** move the Zig engine into fast-mlx, retire mlx-serve, wrap it in a SwiftUI app — v1 in ~4–7 months at 1.5 FTE.
3. A clean-room rewrite costs ~11–21 extra engineer-months (~$220–520K) + 8–13 months' delay to reach today's tok/s, with real risk of shipping slower.
4. Spend the saved months on the dial + precision-loss harness — the real differentiator, engine-agnostic, and the insurance that makes a future Swift port cheap if Apple fixes their eval loop (re-check ~12 mo).
5. v1 = curated LLM catalog + OpenAI API + CLI + macOS app + measured dial; media-gen out; TurboQuant deferred to the v1.1 2-bit tier.

## Update 2026-07-07 (later): Expanded scope — domain-specific training & research velocity

Owner clarified the roadmap includes **domain-specific training/fine-tuning** and **fast adoption of new compression/perf research**, and intends to **hard-fork mlx-serve into a private repo** (MIT permits: private + closed-source + commercial resale; only obligation is retaining the copyright/permission notice for derived code). Two corrections/additions:

- **Correction on custom ops (I got this wrong earlier).** MLX exposes THREE mechanisms, not two: (a) `metal_kernel` JIT custom *forward* kernels — all four languages; (b) new core `Primitive` subclass (new graph node w/ `vjp`/`jvp`) — C++-only *and* recompiles core, C++-locked **even from Python**, rarely on the critical path; (c) `custom_function`/`custom_vjp` (attach a custom gradient to a composed op — the mechanism actually used to adopt a new research method with a custom backward pass) — **present in ALL four, including mlx-c/Zig** (verified verbatim in `mlx/c/transforms.h`). So Zig is **not** locked out of custom differentiable ops; it lacks *scaffolding* (nn/optimizers/LoRA/data pipeline), not primitive capability.

- **Training capability by language (2026):** Python/mlx-lm = full stack (LoRA/DoRA/full-FT, QLoRA, optimizers, eval harness, learned-quant AWQ/GPTQ/DWQ, fuse/GGUF export; DPO/ORPO/GRPO via community `mlx-lm-lora`/`mlx-tune`). Swift/mlx-swift = real but narrower (`MLXNN`/`MLXOptimizers` + `llm-tool lora train/test/eval/fuse`, LoRA/QLoRA only). Zig/mlx-c & C++ = autodiff primitives only, zero training scaffolding, zero precedent. New research lands **Python-first** (MLX core ships C++/Python in lockstep — v0.32.0 shipped 2026-07-07; Swift trails days–weeks; mlx-c trails further, no tagged releases).

**Refined architecture — two planes, not one language:**
- **Serving plane (keep Zig):** hard-fork the Zig engine private. It's proven, and its **mandatory-TDD + equivalence/soak discipline** (per mlx-serve `CLAUDE.md`: format-corpus invariants, per-arch equivalence, 24h soak) **materially blunts the bus-factor risk** flagged above — a well-tested codebase is safely modifiable by a new engineer or by Claude. The substantial **Swift macOS app already exists** (`app/Sources/MLXServe/`: menu-bar, chat, RAG, agent engine, VM sandbox), talking to the engine over HTTP/SSE — so Hybrid is already the de-facto architecture fast-mlx inherits.
- **Research/training plane (add Python):** Python `mlx-lm` (+ vetted community `mlx-lm-lora`/`mlx-tune`) as a **separate subsystem** for domain-specific fine-tuning, eval, learned-quant, and prototyping new compression/perf research → exports quantized safetensors the Zig engine serves. When a technique proves out, implement it in-engine as a Metal / `custom_vjp` kernel (Zig can).
- **Why not switch the engine language:** future-proofing comes from the two-plane split, **not** the engine's language. Training stays in Python regardless of serving language; switching Zig→Swift/C++ would not improve research adoption and would cost the proven engine + working app. The only case for switching is single-codebase unification on Swift (engine+app+train) — but Swift training is narrower, trails research, and serving perf is unproven (7.3× MoE gap); poor trade for "max efficiency + fast research adoption."

**TurboQuant re-confirmed** from the source the owner pointed to: mlx-serve's `CLAUDE.md` + `tests/test_turboquant_equivalence.sh` name the feature "TurboQuant," and the engineer's own notes describe Hadamard-rotation-then-affine (helps on outliers, can hurt on smooth data) — **QuaRot-family, not Google TurboQuant** (arXiv:2504.19874). Owner's skepticism confirmed.

**Refined recommendation:** hard-fork Zig engine (private) + keep existing Swift app + add a Python train/research plane. fast-mlx = a *platform* spanning engine (Zig) · app (Swift) · train/research/eval + dial harness (Python) · curated catalog. Engine language stays Zig.

## Update 2026-07-08 (later): Maturity/tooling criterion + custom-Metal benchmark verdict

Owner reprioritized the engine-language decision on **maturity + availability of tooling** (not overhead), and asked to critically verify a Reddit-cited "custom Metal beats MLX" benchmark.

**Benchmark verdict — RunAnywhere MetalRT (assessment HOLDS):** The cited benchmark is RunAnywhere's closed-source C++ **MetalRT** (high-confidence; Reddit thread was inaccessible to fetch, but the "1.10–1.19× vs mlx-lm decode" fingerprint + the founder's identical HN self-post confirm it). Critically: closed-source, no published harness (**not reproducible**); all models **≤4B on an M4 Max 64GB** (never the 30B–235B / M3 Ultra regime); headline cherry-picks its 3 wins and drops Llama-3.2-3B where **mlx-lm won +14%**; the edge shrinks with size (+19% @0.6B → **−12% @3B** → +9% @4B) — a fixed host-overhead effect that vanishes at scale; **zero independent reproduction**. Independent triangulation: rival vendor BaseRT measured the custom edge going from +35% @0.6B to a **loss @26B MoE**; MLX beats llama.cpp ~1.3–1.5× on Qwen3-35B-A3B and **1.5× on Qwen3-235B-A22B @ M3 Ultra 512GB** (24 vs 16 tok/s); Apple-Silicon engine gaps close within months (stale-claim risk). **Conclusion: no reproducible custom-engine win over MLX in the large-MoE/M3-Ultra regime — MLX is strongest exactly there.** The real ~10–20% small-model single-stream decode edge is host-overhead, gone by ~26–30B, and requires the least-mature path (closed custom Metal). Detail: `docs/research/2026-07-08-mlx-inference-competitive-landscape.md` + this session.

**Language re-ranking under maturity/tooling (REVERSES the prior "stays Zig"):** overhead ranks Zig = C++ = Rust at the floor, but **maturity + tooling** ranks **Swift ≈ C++ ≈ Python > Rust ≫ Zig** — Zig is weakest (pre-1.0 0.16, thin Metal profiling, smallest ecosystem + talent pool). Stay on MLX regardless (custom-Metal is least-mature and a mirage at scale).

**Revised recommendation (maturity/tooling priority): Swift engine.** For a macOS-first product with an existing substantial Swift app: best-in-class tooling (Xcode/Instruments Metal profiling), Apple-backed mlx-swift + mlx-swift-lm arch library, near-floor overhead (ARC negligible in bandwidth-bound large-MoE decode), stack unification (engine+CLI+app), largest talent pool. Risk: unproven Swift serving perf (Apple's mlx-swift-lm 7.3× behind on MoE) — but that's a known eval-loop design flaw (lock + sync GPU→CPU extraction) the Zig engine already solves; **port the proven design**. C++ = alternative if fragmenting the app is acceptable (more mature language, mlx-core-native custom primitives).

**The fork:** (A) keep proven Zig artifact (works/tested/overhead-floor; weak *language* maturity) vs **(B) rewrite in Swift** (mature/tooled/hireable/unified; rewrite + prove serving perf). Owner's stated priority tilts to **(B)**. NEXT STEP: quantify the Swift rewrite (effort w/ mlx-swift-lm head start + eval-loop design port + de-risking plan) before committing — do NOT hard-fork the Zig engine until (A) vs (B) is settled with numbers.

## Architectural principle 2026-07-08: MLX-as-extensible-base (verified from MLX docs)

Confirmed against ml-explore.github.io/mlx: MLX is extremely active (v0.32.0 shipped 2026-07-07; a CUDA backend now exists alongside Metal) and is designed to be **extended**, not just consumed. Three tiers of going beyond the default op graph: (1) `mx.compile` (auto-fusion of existing ops), (2) `mx.fast.metal_kernel()` (JIT custom Metal kernel from source, no rebuild — exposed in Python/C++/**Swift `MLXFast`**/**mlx-c Zig**), (3) custom C++ `Primitive` (new graph node with `vjp`/`jvp`, CMake + nanobind). Docs show custom kernels beating **naive op composition**: grid_sample **8× fwd / 40× bwd** (M1 Max), axpby custom Primitive **~2×**. Apple itself ships fused kernels (`fast.scaled_dot_product_attention`, `rms_norm`, `rope`, `layer_norm`) for the same reason.

**Implications:** (a) The performance ceiling is **not "stock MLX"** — it's "MLX + fused custom kernels for your hot paths" (fused dequant-matmul, fused sampling, custom attention/MoE-routing). This dissolves the "custom engine vs MLX" dichotomy — MetalRT/uzu write custom Metal *outside* MLX (losing the mature base); you write custom fused kernels *inside* MLX (keeping Apple's primitive kernels + ecosystem + tooling) = strictly better on the maturity/tooling priority. (b) **Language-agnostic** — `metal_kernel` is available in Swift AND Zig, so this does NOT affect the Swift-vs-Zig decision; it only confirms "use MLX" never meant "capped at stock speed." (c) The engine's own TODO already has a "fused quant-attention Metal kernel (~5% decode upside)" — this pattern is the right home for custom-kernel perf work in the spec. You won't out-write Apple's GEMM/`fast`-attention; you WILL beat the default graph by fusing your specific patterns.

## Update 2026-07-08 (Swift vs C++, + MLX multi-backend): the decision hinges on deployment target

Owner narrowed to Swift-vs-C++ and asked "does C++ buy anything" + "is MLX more than Metal on OSX." Research (2 docs-researchers, primary sources):

**MLX is genuinely multi-backend now (Apple-first, CUDA-second).** Real Apple-sponsored **CUDA backend** (Awni Hannun, Jul 2025: *"same codebase... Apple silicon, or in the cloud on Nvidia GPUs"*); `mlx[cuda]`, `precompiled_cuda_kernel`, NCCL; 2025-beta gaps (quant matmul, FFT, MoE gather) closing through mid-2026. BUT Apple-Silicon stays flagship: Vulkan `wontfix`, ROCm community-only/unmerged, WWDC26 JACCL (Thunderbolt RDMA) is Mac-cluster-only. Not a vendor-agnostic JAX/PyTorch — Apple-first with CUDA as a resourced 2nd tier.

**Plot twist — Swift is no longer Apple-locked.** mlx-swift shipped a **Linux+CUDA build path ~Jan 2026** (v0.30.2). All three host languages span Metal+CUDA from ONE source tree (mutually-exclusive per-platform build artifacts, not one fat binary). Maturity of that cross-hardware path: **Python > C++ > Swift** (Swift's is ~6mo old, opt-in, cross-platform ask still open).

**Quantified Swift vs C++** (refines the 2026-07-07 numbers with primary evidence): engine subtotal **Swift ~13–24 vs C++ ~16–27 em** (~15% / ~3 em C++ premium); to-v1 **Swift ~17–30 vs C++ ~20–33 em**. Premium concentrated in the **arch/tokenizer breadth layer**: Swift has `mlx-swift-lm` (**59 archs**, Apple-maintained, MIT) + first-party `swift-transformers`/`swift-jinja`; C++ has NO Apple LLM library (every arch a one-time port → your permanent maintenance; precedent: `lemon-mlx-engine`, real MIT C++ mlx-core engine, ~46 archs, but depends on a non-upstream fork + drags Rust via `tokenizers-cpp`). Serving core (~27K LOC) + the perf-critical eval loop are **hand-built in BOTH** (mlx-swift-lm is the 7.3× MoE-regression code; port the proven design either way). **Correction:** Xcode/Instruments Metal GPU profiling is **language-agnostic** (attaches to Metal command buffers regardless of Swift/C++) — NOT a Swift advantage as previously stated. Swift's real edges: 59-arch library, stack unification (existing Swift app), first-party tokenizer, ~3 em cheaper.

**Does C++ buy anything?** Two narrow things: (1) more mature/less-contingent path to MLX's CUDA/cloud future (native `mlx-core` layer vs young mlx-swift binding) — **but largely redundant with the two-plane architecture, since the Python plane already gives a mature NVIDIA-cloud path**; (2) mlx-core-native (custom `Primitive` autodiff — the one C++-locked mechanism — + tracks core's cadence directly; mlx-c has never cut a tagged release). Costs: 59-arch head-start, stack unification (permanent C++↔Swift-app FFI seam), +~3 em, thin C++ MLX docs (ops-only).

**DECISION HINGES ON DEPLOYMENT TARGET (open question to owner):**
- **Apple-Silicon-first** (Mac/edge/on-device; cloud handled by the Python plane) → **Swift** (recommended on current evidence: more complete MLX tooling, unification with existing app, 59-arch head-start, ~3 em cheaper; CUDA path exists as a hedge).
- **Single unified engine must run natively on Mac AND NVIDIA cloud** → **C++** (native layer for portability + custom-op headroom; accept arch-porting cost). Python is the *most* mature cross-hardware path if a compiled host isn't otherwise required.

NEXT: owner to confirm deployment target (Apple-first vs cross-hardware single-engine). That answer selects the language; then spec the platform / de-risk with a Swift decode-loop spike.

## Update 2026-07-08 (RESOLVING → Swift): Claude-authored + limited human review

Owner added the decisive constraint: **implementation will be Claude-written source with limited human review**, and clarified NVIDIA is an **objective, not a hard threshold** (commercial customers may have NVIDIA hardware/cloud).

**This settles the engine language: Swift.** Under limited human review, the COMPILER is the reviewer, and Swift catches classes of bugs C++ ships silently:
- **Memory safety** (ARC, bounds checks, optionals, value semantics) — Claude cannot compile use-after-free/buffer-overrun/null-deref; in C++ these compile and fail silently as UB (the failure mode limited review misses).
- **Swift 6 strict concurrency** — compile-time data-race detection. Serving engine is deeply concurrent (scheduler/batching/SSE); that's where the 7.3× MoE regression lived (lock + sync stall) and where AI-authored code is riskiest. C++ has no equivalent. (mlx-serve pins `-swift-version 5` to ESCAPE this checker; for AI code, run Swift 6 mode deliberately BECAUSE it's strict — the friction is the safety net.)
- **Legible diagnostics** vs C++ template-error soup (MLX is template-heavy) → Claude self-corrects without human unblocking.
- **Complexity/maintainability** (owner's axes): C++ has more for Claude to get right every time (manual lifetimes, ownership, templates, CMake); Swift refactors are compiler-checked; stack unification (engine+CLI+app, one language) removes a C++↔Swift FFI seam (AI-bug source); Swift leans on first-party maintained libs (mlx-swift-lm 59 archs, swift-transformers) = less from-scratch code.
- **Reframes maturity/hireability:** with humans barely reviewing, human talent-pool size matters less than AI-authorability → tilts further to Swift.

**NVIDIA objective met WITHOUT C++:** objective not threshold; Swift has its own young Linux+CUDA path; and the **architecture handles NVIDIA without the engine language** — Python plane (`mlx[cuda]`) is the mature cross-hardware path, and the engine-agnostic OpenAI-API + dial lets NVIDIA customers be served by the Python plane or a proxied runtime (their vLLM) behind the same product surface. Swift engine owns Mac/edge; platform handles NVIDIA. C++'s only edge (mature native cross-hardware) is redundant with this and costs the memory-unsafety limited review can least afford.

**DECISION: Swift engine** (pending owner confirmation). Claude-authored-spec implications: (1) the engine-agnostic conformance + precision-loss harness is the automated reviewer standing in for limited human review — invest heavily; (2) port the proven Zig eval-loop design (submit-first async, no sync per-token readback) so Swift expresses a known-good concurrency design rather than discovering one under the strict-concurrency checker.

**Net language journey (for future readers):** Zig (incumbent, overhead-floor, proven) → reopened by *maturity/tooling* priority → narrowed to Swift-vs-C++ → **Swift**, decided by *Claude-authored + limited-review* (memory + concurrency safety). Two-plane platform stands: **Swift engine · Swift app · Python train/research + dial harness · curated catalog**; MLX as the extensible base (custom fused Metal kernels for hot paths); NVIDIA via the Python plane + engine-agnostic API.
