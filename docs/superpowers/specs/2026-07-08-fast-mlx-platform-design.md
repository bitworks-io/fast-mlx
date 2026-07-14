# fast-mlx — Platform Design Spec

- **Status:** DRAFT for owner review
- **Date:** 2026-07-08
- **Owner:** brian@bitworks.io
- **Companion decision record:** [`docs/decisions/2026-07-07-engine-language-and-lineage.md`](../../decisions/2026-07-07-engine-language-and-lineage.md) (full rationale for every verdict summarized here)
- **Competitive context:** [`docs/research/2026-07-08-mlx-inference-competitive-landscape.md`](../../research/2026-07-08-mlx-inference-competitive-landscape.md)

---

## 1. What fast-mlx is (and is not)

fast-mlx is a **maximally-optimized MLX inference platform for Apple Silicon** (primary dev/bench box: **M5 Max 128GB** / `llmbench`; production target up to M3 Ultra 512GB), built to drive commercial portfolio products. Its **first production deployment is Concierge** (bitworks' AI shopping assistant) — fast-mlx will take over model serving on the M5 Max box that runs the now-retired mlx-serve daemon. It is a **platform, not a single binary**: a Swift inference engine, a native macOS app, a Python train/research plane, and — the spine that ties them together and makes the whole thing safe to iterate — an **engine-agnostic conformance + precision-loss harness**.

**The differentiator is measurement, not raw speed.** The competitive scan is unambiguous: nobody has an independently-reproduced single-stream decode win over MLX, and the whole field *advertises* speed while **nobody quantifies the accuracy they trade for it**. fast-mlx's wedge is the **tunable optimization dial with quantified precision loss** — "dial in the speed you want, see exactly what accuracy you trade, defaulting to the fastest setting with loss you won't notice." That is engine-agnostic product/measurement work, and it is open white space.

**The durable moat is integration velocity + quantification, not any single optimization.** Inference is improving at breakneck speed and from *all corners* — large and small labs, independent researchers, TLDR / GitHub / one-off blog posts. In a market moving this fast, a specific optimization is a *depreciating asset*: someone matches or beats it next month. What compounds is the **machinery to absorb that firehose safely and quantify each technique's speed↔quality tradeoff**, so users get absolute market-leading performance *with the quality cost made explicit*. That machinery — the harness (§6), the dial (§4), and the intake loop that connects them — **is the product**; the engine is its substrate.

**Operating model — the technique-integration flywheel:** monitor the field → implement a candidate behind a flag → run it through the equivalence/engagement/acceptance triad (correctness, §6.1) and the precision-loss + perf harness (quantified tradeoff, §4/§6.3) → **promote** it to a measured dial tier or **shelve** it with a dated negative result → repeat. This is why the harness must be fast, trustworthy, and engine-agnostic: it is the gate every candidate passes through. v1 delivers this loop as internal machinery; the roadmap (§13) opens it up to **user-defined custom harnesses** and **dynamic, runtime-adaptive dialing**.

**Non-goals for v1:** beating MLX's primitive kernels; media generation (image/video/TTS); being a general cross-vendor framework; competing on benchmark tok/s headlines.

---

## 2. Verdict summary (the decisions this spec builds on)

| Decision | Verdict | One-line rationale |
|---|---|---|
| Separate project vs extend mlx-serve | **Separate platform** | Product/dial/catalog is the value; MIT permits a private hard fork |
| Engine base | **MLX (extensible base)** | Nobody beats MLX at scale; `mx.fast.metal_kernel` lets us fuse hot paths *above* stock MLX without leaving the mature base |
| Engine language | **Swift (mlx-swift)** | Decided by *Claude-authored + limited human review*: memory safety + Swift 6 strict-concurrency = the compiler is the reviewer. Also: stack unification with the app, 59-arch head-start (mlx-swift-lm), ~3 em cheaper than C++ |
| Why not C++ | Rejected for this use | Memory-unsafe + silent-UB is the worst fit for AI-authored, lightly-reviewed code; its one edge (native cross-hardware) is redundant with the Python plane |
| Why not Zig (incumbent) | Retired | Overhead-floor and proven, but weakest on the owner's stated *maturity/tooling* priority (pre-1.0, thin ecosystem) |
| Training / research | **Python plane (mlx-lm + community)** | Research lands Python-first; keep training out of the serving engine |
| NVIDIA / cloud | **Via the Python plane + engine-agnostic API** | `mlx[cuda]` is the mature cross-hardware path; the OpenAI-API + dial let NVIDIA customers be served without a C++ engine. Objective, not a hard threshold |
| TurboQuant | **Defer to v1.1 2-bit tier** | The engine's "turbo" is QuaRot-family, not Google TurboQuant; real TurboQuant only helps at 2-bit |

The engine language is settled — **GO**. The Swift decode-loop spike + follow-on optimization (2026-07-08→09) hit the bar: after compiling the decode step (`MLX.compile` + a purpose-built compile-friendly `CompiledKVCache`), Swift reached **155.5 tok/s ≥ same-session Zig 153.65** on the *fastest* MoE model (the worst case for host overhead). The host-loop overhead is **eliminated — decode is now GPU-bound**; Swift 6 strict concurrency held clean (zero unsafe hatches; the compiler caught a real bug); temp=0 equivalence passes. The **compiled decode path (`CompiledKVCache` + `CompiledMLXDecoder`) is the engine's decode core.** (Diagnosis correction: the overhead was C++ per-token graph re-traversal/kernel re-encoding, *not* Swift ARC — Swift itself was never the bottleneck.) Open: re-measure on dense Qwen3-32B (Concierge's production model — expected easier). Full detail: [spike verdict](../verdicts/2026-07-08-swift-spike-verdict.md).

---

## 3. Architecture

Four components + the harness spine. Each communicates through a **well-defined, engine-agnostic interface** (the OpenAI-compatible HTTP API + a debug logprobs surface), which is what makes the harness portable and a future engine swap cheap.

```
        ┌───────────────────────────────────────────────────────┐
        │  macOS app (SwiftUI)  ── chat · server control · DIAL  │
        └───────────────┬───────────────────────────────────────┘
                        │  local HTTP (OpenAI-compatible) + SSE
        ┌───────────────▼───────────────┐        ┌──────────────────────────────┐
        │  fast-mlx ENGINE (Swift)      │        │  PYTHON PLANE                │
        │  • mlx-swift + custom Metal   │        │  • mlx-lm: LoRA/DoRA/FT      │
        │    fused kernels (hot paths)  │        │  • eval + learned-quant      │
        │  • single-owner inference     │        │  • dial harness (measurement)│
        │    actor, continuous batching │        │  • NVIDIA/cloud via mlx[cuda]│
        │  • OpenAI/Anthropic API + SSE │        │    or proxied runtime        │
        └───────────────┬───────────────┘        └──────────────┬───────────────┘
                        │                                        │
        ┌───────────────▼────────────────────────────────────────▼──────────────┐
        │  ENGINE-AGNOSTIC CONFORMANCE + PRECISION-LOSS HARNESS (the spine)       │
        │  drives the HTTP API + logprobs → works across Swift engine, Python     │
        │  plane, or any proxied runtime. The automated reviewer + the dial's     │
        │  measurement instrument. §5, §6.                                        │
        └────────────────────────────────────────────────────────────────────────┘
                        ▲
        ┌───────────────┴───────────────┐
        │  CURATED MODEL CATALOG (signed) │  weights + measured speed/quality frontier per model×hardware
        └─────────────────────────────────┘
```

**Managed via CLI + API, with the macOS app for convenience** (per the original brief). CLI subcommands: `serve`, `run`, `bench`, `verify`, `pull`, `list`, `dial`.

---

## 4. The optimization dial + precision-loss quantification (the product)

The dial exposes a small number of orthogonal knobs and, for each combination, **ships a measured speed↔quality point** rather than an unquantified promise.

**Dial axes (v1):** weight-quant preset × KV-quant tier (`fp16` / `8-bit` / selected
compressed formats), plus exact execution controls such as spec-decode and service policy.
Weight/KV compression occupies the measured quality-loss frontier; an exact control has no
quality-loss coordinate and instead reports its speed/TTFT/TPOT/fairness frontier. Prefill chunk
size is an exact scheduling/cancellation control, not permission to change model quality.
(**Measured through the flywheel:** PLD spec-decode → **PROMOTED** (`--spec pld`) —
distribution-preserving / byte-identical, **2× decode (+100.5%) on high-repetition/agentic
workloads**, with gate-tuned non-target results of code +3.2% and zero-draft prose +0.1%
([verdict + resolution](../verdicts/2026-07-09-pld-firstrun.md)). Exact dense-Qwen3 continuous
batching → **PROMOTED as a measured service-policy building block**: on the canonical
same-workload burst, solo PLD wins C=1 while batch-no-spec wins C=2/4/8 by
45.8%/58.6%/74.7%, with a clean 24-hour stability gate
([verdict](../verdicts/2026-07-14-continuous-batching-chunked-prefill.md)). Neither product
default is wired yet. Uniform-v1 TurboQuant KV → **SHELVED** after adding more measured loss
than the same-weights fp16-KV baseline without a realized packed-memory win; outlier allocation
is separately gated. KVarN/asymmetric affine, 2-bit KV, DSpark, and MTP remain unpromoted until
their lane-specific gates run.) The **KV-quant tier is also the primary large-context memory
lever** — the same knob that trades quality for speed extends how much context fits per GB, so
the dial and the context tunable share it (governed by the capacity advisor in the
[system-aware operability spec](2026-07-09-system-aware-context-operability.md)).

**Three-layer precision-loss metric stack, computed per dial point:**
1. **KL divergence vs fp16 logits** (median + p95) on a fixed mixed corpus (prose / code / long-context) — cheap, most sensitive, the primary signal.
2. **Perplexity delta** vs fp16 on held-out text.
3. **Task-benchmark deltas** on small fixed subsets (MMLU-lite, GSM8K, HumanEval, RULER-style long-context) — the headline "accuracy" number.
   - Speculative decode is **distribution-preserving (exact)** → reported as *speed-only, zero loss* (a marketing gift, and provable via the equivalence harness).

**Per-model Pareto frontier.** Grid the dial coordinates, measure decode tok/s + TTFT on reference hardware (M3 Ultra 256/512GB), and ship the measured frontier as **signed JSON in the catalog**. The app renders it as the dial — every notch shows real, measured numbers, never extrapolations.

**Default policy — "fastest with unnoticeable loss":** auto-select the fastest point where KL ≤ ~0.05 median AND ppl delta ≤ 1% AND task deltas within benchmark CI. Tier labels by measured loss: **Lossless / Balanced (≤1%) / Turbo (≤3%) / Max-fit** (biggest model that fits the box, loss stated).

**Informed consent, not a nanny (dial philosophy).** The default is the *safe* setting, not the *only* one. The dial exposes the **full measured frontier**, including points with **noticeable** loss, and lets a user knowingly opt into them **when the payoff justifies it for their context** — a footprint-constrained device where a more-aggressive quant is the difference between *runs at all* and *doesn't*, a latency/throughput-critical path where a large speedup is worth a stated quality cost, or a loss-tolerant workload (drafting, casual chat). The non-negotiable: every such point is **quantified** (the metric stack above, shown at selection time) so the choice is eyes-open, never a hidden downgrade — bounded by two rails:
- **A hard garbage floor.** Settings that produce incoherent/degenerate output (fail the coherence canary / non-crash / non-NaN lossy-triad floor, §6.1) are **refused outright and never offered** — we don't let a user stretch a model into uselessness, however fast or small.
- **The capacity advisor** (system-aware operability spec) surfaces *why* an aggressive point is worth it on *this* box ("fits at 2-bit KV; would OOM at fp16"), so the trade is contextual, not abstract.

This is the wedge, productized: not "we hide compression behind one quality bar," but "we measure the whole trade and let you spend it — down to a floor that protects you from garbage." The **device/footprint frontier** — extreme-compression tiers for small Macs, candidates like 1-bit weight schemes (PrismML, tracked in the [technique intake](../../reference/performance-technique-intake.md)) — lives in exactly this informed-but-gated region.

**Engine-agnostic by construction:** the harness drives the OpenAI API plus a debug logprobs endpoint, so the dial's numbers are reproducible against the Swift engine today, the Python plane on NVIDIA, or a customer's proxied runtime — and the same harness doubles as the port-conformance suite that de-risks any future engine change.

---

## 5. Eval loop — the known-good concurrency design, expressed in Swift 6

The Zig engine already proved a correct, high-overlap serving loop; we **port the design, not the syntax**, and let Swift 6 strict concurrency enforce the invariants the Zig code hand-rolled. (Source: mlx-serve `scheduler.zig`, `generate.zig`.)

- **Single-owner inference actor.** A dedicated actor owns the mlx-swift model, weights, and *all* array lifetimes; **every** MLX call — forward, sample, eval, even array deallocation — is actor-isolated. Request-handling code is `Sendable`, never imports mlx-swift types, and consumes tokens as an `AsyncThrowingStream`. (Swift actors are built for exactly the "one owner, everyone else asks nicely" invariant the Zig thread+condvar code hand-rolls — and the "MLX-from-two-threads crash class" disappears structurally.)
- **Continuous-batching slots — dense-Qwen3 probe path proven 2026-07-14.** Per-request state
  lives off the shared weights in a pure scheduler plus actor-confined MLX runtime. Decode runs
  before bounded prefill; a stable shared membership executes one batched greedy forward and
  demultiplexes by request ID. Qwen3-32B B1→B2→B1 and B3→B2 are byte-identical, and the
  same-workload C=1/2/4/8 frontier plus 24-hour soak passes. The current executor admits only
  dense `qwen3`; MoE/hybrid/recurrent/vision layouts fail closed. Speculation stays disabled in
  shared batches. The runtime is still explicit-probe-only—production request routing and the
  API surface remain open and are tracked by the
  [serving-route gate](../../task-inbox/2026-07-14-continuous-batching-serving-route.md).
- **NAMED INVARIANT — drain-pipeline-before-batch-join — PROVEN.** The pipelined decode keeps a
  one-step lookahead (a token already forwarded into the KV cache + its pending logits,
  resolved lazily for GPU overlap). When a solo slot joins a batched tick mid-generation, that
  pending state **must be drained first**—dropping it and re-forwarding appends a duplicate KV
  position and re-emits a token. The pure red-on-revert order test and final-SHA real-model
  transition probe both pass; the invariant remains mandatory for every future executor.
- **Chunked prefill — exact fairness/cancellation control, dense-Qwen3 probe path proven.** Prefill
  large prompts in bounded chunks for TTFT/scheduling fairness and a cancellation point between
  chunks. A chunk-size-1 real-model run proves decode interleaves without changing either byte
  stream. **It is NOT a large-context capacity enabler**: chunking bounds transient prefill work,
  not the KV allocation for the full sequence. Hybrid/SSM alignment has not been proven and those
  architectures remain rejected. Before relying on chunking for capacity at 64K+, extend
  `CtxProbe generate` to one-shot 64K/128K/262K prefill and measure the transient peak (see the
  [system-aware operability spec §6/§7](2026-07-09-system-aware-context-operability.md)).
- **Eager warmup.** Immediately after weight load, on the model actor, run one dummy decode-shaped `[1,1]` and one prefill-shaped `[1,8]` forward + `eval` to pre-fault weight pages and force Metal kernel JIT — so cold-start cost is paid at boot, not on the first user token (and doesn't masquerade as "experiment N is slower"). Gated for minimal-footprint deployments.
- **KV / prefix cache — composite key + positive commit gate (NOT reactive invalidation).** Lookup key is an `Equatable` struct encoding every axis that changes buffer semantics: `(token-prefix, hasTools, exact KVQuantConfig)`. Commit is *positively gated* — an entry is written only when generation **cleanly succeeded** (skip pad-only, errored, vision-bearing, zero-token). A degenerate generation's state is never eligible for reuse *by construction*, which structurally avoids the "did we remember every invalidation call site?" bug class.
- **Structured-concurrency shutdown.** Hold request tasks in a `TaskGroup` / tracked `Set<Task>` so teardown `await`s their completion/cancellation before freeing actor state (the Zig SIGSEGV bug class — detached handle-dropped workers touching freed shared state — is what structured concurrency exists to prevent). Cancellation must be observed at every `await` in the generation loop (`withTaskCancellationHandler`), or a "cancelled" stream spins on uncancelled.

---

## 6. Test harness — the spine (first-class; the reason this is safe to iterate 100× )

Because fast-mlx is **Claude-authored with limited human review** and will absorb **10s–100s of performance experiments**, the harness is not QA-after-the-fact — it is the **automated reviewer standing in for the humans who aren't there**, and the **measurement instrument** the dial depends on. It is designed engine-agnostic (drives the HTTP API + logprobs) so it guards the Swift engine, the Python plane, and any proxied runtime identically.

### 6.1 The core pattern — the equivalence + engagement + acceptance triad

**This is the single most important thing carried over from the Zig repo.** Byte-equality alone *cannot* tell "the optimization ran and matched baseline" from "the optimization silently no-op'd and baseline produced the same output" — they are identical at the output layer by definition. (mlx-serve shipped a drafter hardcoded `off` on two call sites for a **month** for exactly this reason.) So every perf/correctness-affecting feature ships a **three-way proof**:

1. **Equivalence** — first-N-token match vs a reference baseline at temp=0. *First-N, not full-output*: INT4/MoE/GatedDeltaNet float-reduction order differs between single-token-decode and batched-verify kernels, so near-tie argmaxes legitimately flip past a horizon. The horizon is a **documented, per-family, env-tunable threshold** (~30–80 tokens), not a hidden fudge. Genuinely-lossy modes (2-bit) loosen to *non-crash + non-NaN + short-prefix-match + a coherence canary* (a fixed prompt whose temp-0 answer must contain a known substring — because a misrouted quant produces fluent-looking garbage).
2. **Engagement delta** — grep the structured per-request log marker (`[spec-stats] mode=…`) and assert the count **strictly increased** for the request under test (a delta check, not a presence check — can't be fooled by a prior request's engagement). Proves it *actually ran*.
3. **Acceptance / effectiveness floor** — e.g. draft acceptance ≥ 50%. Proves it ran *well*, not degenerately (a structurally-broken draft head can engage every request, accept ~0%, gate-fall-back to regular decode, and pass both equivalence *and* engagement).

Any one alone is insufficient; all three are a **required checklist item** for every new dispatch/optimization path. Scheduling transitions (a solo stream vs. one that a concurrent request joins mid-generation) are equivalence-tested the same way numerics are.

### 6.2 Hermetic format/correctness corpus

A weight-free, in-process table of **real captured (input → expected-output) pairs** run through pure functions (tokenizer edge cases, chat-template rendering, tool-call parsing, sampling-param resolution). Two layers: (1) per-entry specific assertions; (2) **universal invariants** applied to the whole table (no control-tag leak into visible content; tool args are valid JSON; hostile-byte content is included, not just well-formed) — so *growing the corpus needs no new invariant code*. Live HTTP failures auto-dump raw model output for one-step harvesting back into the hermetic corpus.

### 6.3 Performance / benchmark harness

A **cell matrix**: {prefill, decode, echo, code} workloads × {none, PLD, DSpark} modes, mlx-swift-only by default (fast dev loop), with opt-in comparison runs. Methodology rules baked into the harness entry point:
- **Verify the release build before trusting any number.** (Zig's lesson: a Debug binary made every benchmark look like a 2–4× regression, silently. Swift analogue: assert `-c release` / optimization config at harness start.)
- **Apples-to-apples, *verified* not assumed:** same context, `temp=0, top_p=1`, no system prompt, thinking off — and check `reasoning_tokens` didn't leak rather than trusting a flag.
- **Measure rate from the live SSE stream** (count delta events, time first→last token), never from self-reported `usage`.
- **Salt prompts per run** so repeats don't silently hit a warm prefix cache and inflate numbers.
- **Drop warmup run, average the rest**; bounded liveness-aware retry on a 0-tps cell; garbage-response guard → `ERR`, don't crash.
- **Durable evidence, append-only:** narrative log (`BenchmarkLog`-style) + raw per-run CSV with enough dimensions (label, mode, concurrency, model, hardware) to diff programmatically against a prior run. **Negative results are logged with the same rigor as positive ones** (dated "measured no-op, retired") so a 100-experiment regime never re-attempts a dead end blind.

### 6.4 Soak / stability

- **Multi-workload concurrent soak** (chat + agent-recall + Anthropic + tool-calling, concurrently) with **RSS-drift-bounded** pass/fail (quantified % growth over time, not "didn't crash"); baseline = the *second* sample (exclude cold-start warm-up).
- **Liveness ≠ responsiveness:** a GPU/inference deadlock hangs without killing the process → probe `/health` answers within N seconds *separately* from PID-alive.
- **Disconnect-cancel SLA:** when a client vanishes mid-prefill, the slot is cancelled within one keepalive interval so the next request isn't queued behind ghost work (measured directly, not "eventually").
- **State-poison A/B/A:** request A succeeds → request B (degenerate/hostile) → re-send A must still succeed. Generalizes to any mutable server-side cache.
- **Load→unload→reload N times returns to baseline resident memory** — the standing guard for FFI/refcount leaks across the mlx-swift → C++ boundary (verified by *measurement*, not code inspection).

### 6.5 API-conformance matrix

A **representative-model-per-architecture-family** matrix (skip-if-missing so it degrades gracefully on a partial local cache), each model proven two independent ways: an external OpenAI/Anthropic-API spec-conformance validator + a live multi-turn agentic tool-calling case. Append-only TSV evidence trail.

### 6.6 House rules (enforced, not aspirational)

- **TDD order:** failing test → minimum pass → full suite green → iterate. A live curl is a sanity check, **not** a test.
- **Class bugs get class guards:** a bug that reveals a *class* ships with (1) the instance regression test, (2) a universal-invariant/corpus entry covering the whole class, (3) a named gotcha (symptom signature + prevention rule).
- **Toy-green/real-red is banned:** any assertion through a quantized/gathered/batched/routed path must be proven at **real checkpoint + realistic input scale** — toy dimensions passing is *not* evidence of correctness for anything involving float-reduction order, routing, or backpressure.

### 6.7 Extensibility — built for custom & dynamic harnesses (roadmap-ready)

Because the harness must absorb a firehose of techniques and — per the roadmap (§13) — eventually run *user-defined* evaluations, it is **pluggable from day one** rather than hardcoded:
- **Technique-under-test is an adapter, not a hardcode.** A new quant scheme, spec-decoder, sampler, or fused kernel plugs in behind the triad through a stable interface, without touching the harness core. This is the intake point of the flywheel (§1).
- **Metrics are pluggable.** KL-vs-fp16 / perplexity / task-suite are the built-ins; a custom metric — a domain-specific judge, a structured-output validity rate, a p99-latency SLA — implements a small interface and joins the Pareto measurement.
- **Corpora / workloads are pluggable.** The fixed mixed corpus is the default; a custom corpus (a customer's domain traffic, a specific long-context profile) drops in to re-measure the frontier for *that* workload.

This pluggability is the v1 groundwork that makes the near-future **customized harness implementations** a natural extension, not a rewrite.

---

## 7. Recurring bug-class guards (seed the harness from day one)

From the Zig repo's hard-won incidents — guard these *before* they recur:

| Class | Symptom | Prevention (built into fast-mlx from day 1) |
|---|---|---|
| Silent fallback invisible to equality | Feature no-ops, output matches baseline | Engagement-delta + acceptance-floor on every dispatch path (§6.1) |
| Toy-green / real-red | Synthetic test passes, real checkpoint fails silently-wrong | Real-scale assertions for any quant/route/batch path (§6.6) |
| Refcount leak at FFI boundary | Resident memory never drops on unload | Load/unload/reload-to-baseline memory test (§6.4) |
| Config-default drift across layers | App/CLI assumes a default that changed → wrong head count / 16GB overfill | Emit values explicitly or a guard test that fails on drift |
| Prompt-format silent downgrade | One bad byte → fallback render in wrong format → "model" looks broken | Visible-level log on any template fallback + hostile-byte corpus cases |
| Cache poisoning | Bad generation contaminates later good requests | Positive-commit-gate design (§5) + A/B/A soak test |

---

## 8. v1 scope

- **IN:** LLM text serving; curated catalog (see §9); OpenAI-compatible API + CLI (`serve/run/bench/verify/dial`); SwiftUI macOS app (chat, server control, dial UI); the dial (§4) with measured frontiers; 256/512GB big-memory configs; the full harness (§6); the ported eval loop (§5); the **internal technique-integration flywheel** (§1 — implement-behind-flag → triad + dial → promote/shelve, negative results logged) with a **pluggable** harness (§6.7) so custom metrics/corpora are an extension, not a rewrite.
- **PREVIEW (unmeasured, excluded from dial quality claims):** Qwen-VL vision.
- **OUT:** media generation (image/video/TTS); Anthropic/Ollama/WS as *supported* surfaces (park behind the OpenAI core); 2-bit KV / real TurboQuant / MTP as *supported* tiers (experimental flags); training/fine-tuning in the engine (lives in the Python plane); non-macOS engine builds (NVIDIA served via Python plane).

---

## 9. Model catalog

**Authoritative source:** the model-selection ladder in `bitworks.io-website/concierge/docs/scaling-model-options.md` (scored against a threshold matrix T1–T7 — tool-calling, license, batching support, ≥32K ctx, API compat, instruction-following, p95 latency — and an optional matrix O1–O12). **License markers are load-bearing** — a wrong license is the worst possible error. Curation, not "everything supported," is the promise. Adapted for fast-mlx's Apple-Silicon (up to 512GB M3 Ultra) scope:

**Primary (Tier A — efficiency sweet spot):**
- **Qwen3-30B-A3B-Instruct-2507** ⭐ (30.5B / 3.3B active, **Apache-2.0**, 262K→1M ctx) — *the recommendation*; best efficiency×quality×license.
- GLM-4.5-Air (~106B / ~12B active MoE, **MIT**); Gemma 3 27B dense (⚠️ **Gemma license** — legal sign-off); Qwen3.5-35B-A3B (⚠️ verify card/license, post-cutoff).

**Secondary (Tier B — dense, higher per-request quality / known-good baseline):**
- **Qwen3-32B** dense (**Apache-2.0**) — current production baseline; golden-set reference.
- Llama 3.3 70B (⚠️ **Llama Community License**, <700M MAU cap); Mistral Small 3.x 24B (**Apache-2.0**); Phi-4 14B (**MIT** — validate tool-calling; near the reliable-JSON floor).

**Frontier / 512GB flagship (Tier C — large MoE):**
- Qwen3-235B-A22B (**Apache-2.0**); **DeepSeek-V4-Flash** (284B / 13B active, already servable at 2-bit via the vendored `ds4` engine — the 512GB demo); DeepSeek-R1 (**MIT**); GLM-4.6 (**MIT**). Requires the GPU-wired-memory raise (backlog "System").

**Owner additions — RESOLVED (2026-07-09):**
- **Nemotron 3 Ultra** — ✅ real (NVIDIA, Computex 2026). `model_type: nemotron_h` — the *same* arch family fast-mlx already inherits (hybrid Mamba2 + MoE + select-attention), a scale-up to 550B/55B-active. **Plausibly loadable by existing Swift code** (⚠️ pending a Codable-field check for its MoE additions + block-pattern parser scaling). 256K shipped default (⚠️ "1M" is a training claim, not the config). ~275GB NVFP4 → **512GB only**. License **OpenMDW-1.1** → legal sign-off before commercial use.
- **Ornith** — ✅ real (DeepReinforce, `Ornith-1.0` @ 9B/35B/397B). `model_type: qwen3_5_moe`, **post-trained on Gemma-4/Qwen-3.5** → loads through the existing `Qwen35MoE.swift` registration with **zero net-new engine work**. 397B ~198.5GB @4-bit (fits 256GB). ⚠️ MIT reported — confirm at pull.
- **Context caps + per-arch KV memory + memory-fit (128/256/512GB)** now live in the [system-aware context-operability spec §2](2026-07-09-system-aware-context-operability.md) — including the finding that the naive KV formula is **wrong for 5 of 14 models** (hybrid-linear / SWA / MLA-as-implemented / Mamba2 don't do uniform GQA), and two catalog-shaping results: **DeepSeek-R1 = 152.5 GiB KV @32K as-implemented** (not viable long-context until absorbed-MLA is built) and **Phi-4's 16K max sits *below* the 32K default**.

Proven-in-Zig architectures fast-mlx inherits (mlx-swift-lm should cover most): Qwen3 / 3.5 / 3.6 dense+MoE + Qwen3-Next (DeltaNet), Gemma 3 / 4 (+MoE), Llama / Mistral, Nemotron-H (Mamba2 hybrid), LFM2, DeepSeek-V4-Flash (via `ds4`). Optimization backlog per model class: [carry-forward performance backlog](../../reference/2026-07-08-carry-forward-performance-backlog.md).

---

## 10. Implementation plan outline (→ writing-plans)

**Task 1 (GATE): Swift decode-loop spike.** Port the §5 eval-loop design for **one MoE architecture** onto mlx-swift; run under **Swift 6 strict concurrency**; benchmark decode + TTFT vs the Zig engine on the 256GB M3 Ultra using the §6.3 methodology. **Go/no-go gate:** decode within **~10–15% of the Zig engine's tok/s** (and the strict-concurrency model expresses the design cleanly) → Swift validated in code, proceed. A **≥1.5× gap** signals a design/perf wall (the mlx-swift-lm failure mode) → reassess *here*, cheaply, before building the platform around it.

Subsequent phases (sequenced in the plan): (2) harness spine first — the equivalence/engagement/acceptance triad + hermetic corpus + perf harness (so every later change is measured); (3) single-model engine end-to-end (load → API → stream) behind the harness; (4) continuous batching + the drain-before-join invariant + its test; (5) quant tiers + KV-quant + the dial's measurement pipeline; (6) spec-decode (PLD, DSpark); (7) the catalog + per-model measured frontiers; (8) the macOS app dial UI; (9) the Python plane (train/research + the dial harness as shared measurement); (10) soak/conformance/hardening.

---

## 11. Risks & open questions

- **Swift serving-perf parity is unproven** (Apple's mlx-swift-lm is 7.3× behind on MoE — a *design* flaw we avoid by porting the proven loop). **Mitigation: Task 1 spike is the gate.**
- **Swift 6 strict concurrency friction** — real, but it's the safety net working; run Swift 6 mode deliberately.
- **Catalog unknowns** — Nemotron 3 Ultra / Ornith ✅ resolved (§9). New catalog-shaping risks: the KV-memory model **must dispatch per `model_type`** (one formula is off 4×–71×), and **DeepSeek-R1 is not viable at long context as-implemented** (152.5 GiB KV @32K) until absorbed-MLA is built — both tracked in the [operability spec](2026-07-09-system-aware-context-operability.md).
- **mlx-swift CUDA path is young** — only matters if NVIDIA-as-single-engine becomes a hard requirement; today handled by the Python plane.
- **Custom Metal fusion scope** — decide per-hot-path (fused dequant-attention is the known ~5% win); don't fight Apple's primitive kernels.

---

## 12. Definition of done (v1)

Engine serves the curated catalog over the OpenAI API + CLI + app; the dial ships measured speed/quality frontiers per model×hardware with the default "fastest-unnoticeable-loss" policy; the full harness (triad + corpus + perf + soak + conformance) is green in CI and gates every change; soak is RSS-drift-bounded over ≥4h; and the whole thing is reproducible via `fast-mlx verify` on a customer's own machine.

---

## 13. Roadmap / North Star — beyond v1

The v1 flywheel (§1) is internal machinery. The North Star extends it toward a **dynamic, customizable quantified-tradeoff platform** — directional, not committed scope, captured so the v1 architecture (pluggable harness §6.7, engine-agnostic measurement, signed per-model frontiers) doesn't foreclose them:

- **User-defined / customized harnesses.** Let a customer define their own evaluation — their quality bar (custom metric or LLM-judge), their corpus/workload, their acceptable-loss policy — and receive a Pareto frontier and default dial setting tuned to *their* domain rather than a generic one. (§6.7 pluggability is the groundwork.)
- **Dynamic dialing.** Move from static, pre-measured tiers to **runtime-adaptive** selection: pick the fastest setting that meets a live per-request / per-session quality-latency target, adapting to load, prompt type, or a customer SLA. Requires a cheap online quality signal — the KL-vs-reference primitive is the candidate.
- **Continuous field-monitoring intake.** Tooling that lowers the cost of "saw a technique on TLDR/GitHub → get a quantified verdict": templated technique adapters, a standing regression scoreboard (measured expected gain per technique, negative results retained), fast promote/shelve. This is the flywheel (§1) turned into daily practice.
- **Big-memory frontier.** 2-bit KV / real Google TurboQuant for 512GB long-context "Max-fit" tiers (§2) — measured, promoted only if the harness shows a win.
- **Cross-hardware reach.** If commercial customers standardize on NVIDIA, harden the Python-plane serving path (`mlx[cuda]`) and/or the engine-agnostic proxy so the dial + measured tradeoffs travel to their hardware behind the same API.
