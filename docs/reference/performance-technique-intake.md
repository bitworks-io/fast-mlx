# Performance-Technique Intake (flywheel candidates)

The **monitor-the-field front of the flywheel** (spec §1): candidate *external* techniques to evaluate + quantify through the harness, before deciding to promote or shelve. These are **unproven for fast-mlx** and vendor-numbers unless noted. Complements [`carry-forward-performance-backlog.md`](2026-07-08-carry-forward-performance-backlog.md) (proven Zig techniques already ours).

**Process:** classify each candidate under the
[`EXACT`, `LOSSY_FRONTIER`, or `EXPERIMENTAL` verdict contract](../superpowers/verdicts/README.md)
→ implement behind a flag → run the lane-appropriate triad + quality/performance harness →
**promote to a dial tier or shelve with a dated negative result.** Exact techniques must
preserve their declared equivalence contract. Intentional approximations may trade real,
quantified quality for speed, memory, power, or model fit as long as they remain above the
garbage floor and are not dominated by a better configuration. Judge on *quantized/MoE*
numbers (our real workloads), never bf16-small-model or vendor headlines. Do not adopt on an
acceptance number without checking its denominator (DSpark lesson).

---

## 2026-07-12 Sol audit — current candidate decisions

The [full portfolio audit](2026-07-12-sol-optimization-landscape.md) reconciled this intake
against the current Swift code. Four old labels were stale: native MTP and prefix/SSD caching
exist only in adjacent/upstream engines, absorbed MLA has now shipped in Python MLX, and
PrismML has released real model artifacts. Current external candidates, ranked inside their
lane:

1. **KVarN K4V2-g128 — HIGH, new KV gate.** Calibration-free token tiles, Hadamard channel
   rotation, two-axis variance normalization, 4-bit keys, and 2-bit values. The
   [paper](https://arxiv.org/abs/2606.03458) establishes the method and 2/2-bit quality on
   smaller models. A later [official-repository](https://github.com/huawei-csl/KVarN) author
   benchmark reports Qwen3-32B AIME25 parity, roughly 4× KV capacity, and fp16-or-better
   throughput for a 16K-context burst at TP=2. Those system results are new, author-reported,
   and CUDA/Triton-only. Action: compare exact
   KVarN against affine K4V2/K8V2, KVTuner schedules, 4-bit affine, and bounded TurboQuant
   recipes at equal packed bytes; custom Metal only after the quality/size gate.
2. **KVTuner mixed K/V policy — FOLD INTO KVarN PHASE 2.** Offline sensitivity search
   selects per-layer K8V4/K8V2/K4V2 configurations; Swift consumes a small policy artifact.
   [Paper](https://arxiv.org/abs/2502.04420),
   [official code](https://github.com/cmd2001/KVTuner). This is a control-plane technique,
   not a competing attention kernel.
3. **Learned/mixed weight quantization — MEDIUM-HIGH, bounded sweep.** Official
   [MLX-LM tools](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md)
   now cover dynamic layer sensitivity, DWQ, AWQ, and GPTQ; oMLX
   [oQ](https://github.com/jundot/omlx/blob/main/docs/oQ_Quantization.md) adds a current
   external comparison. Produce ordinary MLX checkpoints and let the Swift harness decide.
4. **PrismML Ternary/Bonsai — RESEARCHABLE DEVICE TIER.** Official
   [demo/models](https://github.com/PrismML-Eng/Bonsai-demo) exist. Ternary 2-bit uses stock
   MLX; binary 1-bit currently depends on PrismML forks while upstream support is pending.
   These are separately trained model families, not a post-hoc quantizer for Qwen.
5. **EpiCache / KVzip selective retention — RESEARCH-LATER, Concierge-shaped.** Apple's
   [EpiCache](https://machinelearning.apple.com/research/epicache) and
   [KVzip](https://github.com/snu-mllab/kvzip) reduce retained conversational context, so
   they are lossy memory policies rather than numeric KV codecs. Exact prefix caching and a
   multi-turn task corpus must land first.
6. **CommVQ, AQUA-KV/xKV, SliceGPT — WATCH.** Stronger research-plane candidates than
   unstructured sparsity, but each adds training, checkpoint, cache-contract, or custom
   kernel work before the current KVarN/learned-quant gates exist.
7. **XGrammar 2 — PRODUCT-ADJACENT.** Exact structured-output masking has Apple/Metal
   support and high tool-call value, but belongs after the serving/sampler surface; it is not
   a base decode multiplier. [Official project](https://github.com/mlc-ai/xgrammar).

Generic FlashAttention, prefill/decode disaggregation on one Mac, and CUDA-only sparse/
activation-quant kernels are not new fast-mlx tasks. Upstream MLX already owns optimized
SDPA; multi-accelerator disaggregation belongs to the future scale-out plane.

---

## 2026-07-09 intake — sources: Reddit/DFlash · MTPLX · omlx

### Candidates to quantify (ranked)

**1. DFlash — block-diffusion external speculative drafter — PRIORITY: HIGH (conditional)**
- *What:* a small (~1B) **block-diffusion** draft model proposes a 16-token block; the target verifies the whole block in one forward; longest-matching-prefix accept + bonus token; per-layer KV rollback on reject. Published: [arXiv:2602.06036](https://arxiv.org/abs/2602.06036). MLX ports (Python 100%): [bstnxbt/dflash-mlx](https://github.com/bstnxbt/dflash-mlx) (748★), [Aryagm/dflash-mlx](https://github.com/Aryagm/dflash-mlx) (380★).
- *Claimed:* 4.1× on bf16 4B/9B; **1.7–1.9× on 4-bit 27B / 35B-A3B MoE** (M5 Max 64GB, vs stock `mlx_lm.stream_generate`); ~89% "acceptance." (Vendor, relayed via content-farm — unverified.)
- *Why it's the top lead:* it targets **exactly the gap DSpark left open** — external drafters
  lose at batch=1 when target verify cost and draft cost exceed accepted-token savings. The
  old Qwen3-8B/DSpark pair needed roughly 2.3 accepted drafts/round and measured 1.5–1.8,
  but that threshold is not transferable to another target/drafter pair.
- *GATE before any code:* measure each pair's draft cost, target AR cost, verify cost, and
  accepted drafts/round with the correct denominator, then calculate its own break-even.
  Judge on quantized/product-size targets, not a bf16-small headline.
- *Fold-in (not standalone):* a SWA long-context draft variant (Z-Lab checkpoint) claimed to hold accept-length past 4K — evaluate only within DFlash if pursued.
- **UPDATE 2026-07-12:** provenance is no longer the blocker. The current
  [optimized MLX port](https://github.com/bstnxbt/dflash-mlx) publishes per-run JSON and
  reports 2.78–3.06× for Qwen3.6-27B-4bit from 1K–16K output on an M5 Max; Qwen3.5-27B
  falls from 2.37× at 1K to 1.34× at 8K, exposing model/length sensitivity. The port also
  warns that MLX dispatch differences can change the AR byte stream even though it emits no
  unverified token. fast-mlx therefore keeps EAGLE-3/DSpark first for its target-compatible
  Qwen3-32B checkpoint and uses DFlash as a measured control; our byte-identical gate is
  stricter than the port's “lossless” label.
- **GATE RESULT 2026-07-12 — EAGLE-3 SHELVED:** the authenticated Qwen3-32B head passed
  PyTorch→MLX parity, but target verification changed the greedy stream on both 4-bit and
  8-bit pairings. No compatible Qwen3-32B DFlash checkpoint was found in the audited public and
  bench inventory for the same-target control. DFlash remains conditional on a product-size
  pairing and must pass the same one-token versus batched-target byte-identity gate before any
  throughput claim.

**2. omlx oQ4e — importance-calibrated (imatrix) weight quantization — PRIORITY: MEDIUM**
- *What:* activation-importance / imatrix-calibrated **weight** quant (incl. fractional bit levels), vs uniform affine. From omlx v0.5.x release notes.
- *Why:* a **novel axis** — our quant backlog covers **KV cache** (affine + Hadamard "turbo"), not importance-calibrated **weights**. Directly on-strategy for the **measured precision-loss dial** (the white space). It's an *offline checkpoint-production* step — the Swift decode loop just consumes a differently-quantized checkpoint (no loop change), so cheap to investigate.
- *Action:* scope real accuracy-vs-size numbers (not extractable from release notes) before harness time. Not urgent.
- **UPDATE 2026-07-12:** replace the release-note-only lead with the bounded
  [learned/mixed weight-quant task](../task-inbox/2026-07-12-learned-weight-quant-frontier.md).
  oQ4e is one comparison arm alongside official MLX dynamic quantization and DWQ, not the
  sole source of truth.

**3. Google TurboQuant — data-free near-optimal KV-cache vector quantization — PRIORITY: HIGH (owner-requested)**
- *What:* Google Research's TurboQuant ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874); [blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)) — an **online, data-free vector quantizer**: random-rotate → coordinates become ~Beta-distributed → apply a closed-form per-coordinate **optimal** scalar quantizer (within ~2.7× of the information-theoretic distortion bound). A `_prod` variant adds a 1-bit QJL residual for unbiased Q·K inner products. Evaluated on **KV cache**: quality-neutral at **3.5 bits/channel**, marginal at **2.5-bit**.
- *Why (owner-requested — corrects the earlier mix-up):* this is the **actual** Google TurboQuant the owner meant — **not** mlx-serve's `turbo2/4`, which was our own Hadamard-rotation-then-affine (QuaRot-family) scheme mislabeled "turboquant." The real distinction that could matter: TurboQuant's distribution-optimal non-uniform quantizer **+ zero stored per-group scale/zero metadata**, vs our affine scheme's ~25%-of-bit-budget metadata tax at group-64/2-bit, plus the inner-product debiasing.
- *Where it was expected to win:* the **2-bit / max-context tier**. The retired Zig engine
  had rotation+affine `turbo4` and 2-bit controls, while the hardened harness has a
  4-bit-weight/fp16-KV reference row but no same-weights affine-KV quality row. The current
  Swift runtime does **not** implement ordinary affine cache tiers. The new KVarN/asymmetric task owns creation of the Swift
  baseline; TurboQuant B may promote only if its clean-SHA row beats the available
  affine/KVarN frontier at equal actual packed bytes.
- *Effort:* port/adapt the Beta-optimal quantizer (+ optional QJL residual) — materially more
  than affine RTN; data-free (no calibration). Community MLX/Swift implementations now exist
  (source update below), but no upstream/core path has passed the fast-mlx harness.
- **SOURCE UPDATE 2026-07-12:** “no known MLX/Swift implementation” is stale.
  [MLX-VLM](https://github.com/Blaizzy/mlx-vlm),
  [TurboQuant+](https://github.com/TheTom/turboquant_plus), and other community ports now
  provide packed/Metal implementation leads. Upstream support and their quality/performance
  claims remain unverified for fast-mlx.
- **UPDATE 2026-07-09; clarified 2026-07-14 — DONE + SHELVED.** Built exactly (Spike A verified vs the paper's distortion table), integrated, and measured. Uniform-v1 tqB3 adds loss over the same 4-bit-weight/fp16-KV row (tail-p95 1.797/ppl +32.6% vs 1.665/+21.4%) while its actual unpacked cache is larger; tqB2 is catastrophic. No ordinary affine-KV quality row was measured. Shelved as a dated negative result; gated next step is **Spike B (outlier channels)**. Verdict: `docs/superpowers/verdicts/2026-07-09-turboquant-firstrun.md`.

**4. PrismML — 1-bit weight quantization — PRIORITY: RESEARCH-LATER (owner-flagged)**
- *What:* official Bonsai binary (1-bit) and Ternary-Bonsai (1.58-bit information stored in a
  2-bit MLX format) model families now exist in 1.7B/4B/8B sizes. These are trained low-bit
  checkpoints, **not** post-hoc quantization of arbitrary models. Source:
  [Bonsai demo and model matrix](https://github.com/PrismML-Eng/Bonsai-demo).
- *Why it's on-strategy:* the **device/footprint frontier** of the dial's *informed-consent* region (spec §4) — extreme compression that trades *noticeable* quality for *runs-at-all* on a small Mac. Exactly the "massively smaller footprint for a stated loss, gated against garbage" case: 1-bit is unlikely to clear the "unnoticeable" bar, but can still deliver value where the alternative is the model not fitting **— if it stays above the coherence/garbage floor.**
- *GATE:* benchmark Ternary-Bonsai 8B first because stock MLX can load its 2-bit format.
  Binary Bonsai remains conditional on reviewing the PrismML MLX/MLX-Swift forks or waiting
  for upstream support. Judge against same-footprint competitors, tool/code/agent tasks, and
  the coherence floor—not as a teacher-forced quantization of an unrelated Qwen reference.

### Do not re-evaluate as standalone techniques (folded / superseded / marketing)
- **MTPLX** ([youssofal/MTPLX](https://github.com/youssofal/MTPLX)) — native MTP is
  **not implemented in fast-mlx**. The pinned Swift dependency and external engines provide
  reference machinery, but the actor/compiler/harness path exposes PLD only. Fold native MTP
  into a future compatible trained-speculator gate as a lower-port-cost control; the 2026-07-12
  Qwen3-32B gate closed RED without a matching MTP checkpoint. Do not treat MTPLX marketing or
  a single condition as a comparison pair.
- **omlx "Lightning MTP"** — not a separate task. Native MTP is absent from fast-mlx and is
  now a control inside the trained-speculator gate; “verify-shape kernels” alone do not prove
  a net Apple win.
- **omlx "Adaptive Burst Decode"** (reduces per-token executor overhead) — **SUPERSEDED**: we eliminated 81.7% of per-step cost (per-token graph rebuild) outright today via `MLX.compile` + `CompiledKVCache`, rather than batching around it.
- **omlx SSD/paged KV cache** — **not already-have**. ds4 streams model weights; a
  cross-request prefix/session cache is a different state and is absent from fast-mlx. The
  exact hot-cache task precedes any cold SSD tier.
- **omlx DeepSeek-V4 MXFP4 MoE + sparse-attention Metal kernels** — narrow; revisit only if DeepSeek-V4-Flash is near-term on the catalog.
- **auto-depth calibration** (both MTPLX + omlx: search MTP/draft depth per model/HW) — minor incremental; we already have dynamic self-management (PLD yield-gate/re-enable). LOW.

### Honesty / hygiene flags
- The original Reddit trail remains unusable and irrelevant. DFlash now clears the source bar
  through its paper, official code, two MLX ports, and published per-run artifacts. Its
  benchmark results are still maintainer/self-reported until fast-mlx reproduces them.
- **Every benchmark here is vendor-self-reported / unverified**; the omlx numbers are self-*referential* (vs its own prior version, not vs stock mlx-lm). Quantify independently before adoption.
- A **prompt-injection artifact** (fake "system-reminder" pushing computer-use tools) was embedded in a fetched omlx page and ignored — flagging for hygiene.

### Sources
DFlash: [arXiv:2602.06036](https://arxiv.org/abs/2602.06036) · [bstnxbt/dflash-mlx](https://github.com/bstnxbt/dflash-mlx) · [Aryagm/dflash-mlx](https://github.com/Aryagm/dflash-mlx) · Reddit shortlink (unresolved) `reddit.com/r/LocalLLaMA/s/hU6pN7yPe4` · best-match thread (unverified) `.../comments/1skesyq/`. MTPLX: [youssofal/MTPLX](https://github.com/youssofal/MTPLX). omlx: [jundot/omlx/releases](https://github.com/jundot/omlx/releases) · [v0.5.0.dev4](https://github.com/jundot/omlx/releases/tag/v0.5.0.dev4).
