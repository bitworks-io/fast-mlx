# Performance-Technique Intake (flywheel candidates)

The **monitor-the-field front of the flywheel** (spec §1): candidate *external* techniques to evaluate + quantify through the harness, before deciding to promote or shelve. These are **unproven for fast-mlx** and vendor-numbers unless noted. Complements [`carry-forward-performance-backlog.md`](2026-07-08-carry-forward-performance-backlog.md) (proven Zig techniques already ours).

**Process:** each candidate → implement behind a flag → run the equivalence/engagement/acceptance triad + dial/perf harness → **promote to a dial tier or shelve with a dated negative result.** Judge on *quantized/MoE* numbers (our real workloads), never bf16-small-model or vendor headlines. Do not adopt on an acceptance number without checking its denominator (DSpark lesson).

---

## 2026-07-09 intake — sources: Reddit/DFlash · MTPLX · omlx

### Candidates to quantify (ranked)

**1. DFlash — block-diffusion external speculative drafter — PRIORITY: HIGH (conditional)**
- *What:* a small (~1B) **block-diffusion** draft model proposes a 16-token block; the target verifies the whole block in one forward; longest-matching-prefix accept + bonus token; per-layer KV rollback on reject. Published: [arXiv:2602.06036](https://arxiv.org/abs/2602.06036). MLX ports (Python 100%): [bstnxbt/dflash-mlx](https://github.com/bstnxbt/dflash-mlx) (748★), [Aryagm/dflash-mlx](https://github.com/Aryagm/dflash-mlx) (380★).
- *Claimed:* 4.1× on bf16 4B/9B; **1.7–1.9× on 4-bit 27B / 35B-A3B MoE** (M5 Max 64GB, vs stock `mlx_lm.stream_generate`); ~89% "acceptance." (Vendor, relayed via content-farm — unverified.)
- *Why it's the top lead:* it targets **exactly the gap DSpark left open** — external drafters lose at batch=1 because the fixed verify-forward cost needs **>~2.3 accepted tok/round** to win, and DSpark measured 1.5–1.8 and lost. If DFlash's effective accepted-length clears that on *quantized/MoE* targets, it's the real thing. Our `KVCache.truncate` (built for DSpark) already gives the rollback primitive; our L1 batched accept-walk is already exact/marginal-preserving.
- *GATE before any code:* confirm whether "89%" is **per-token or per-round**, and whether effective accepted-length clears ~2.3 tok/round on 4-bit/MoE. Judge on the **1.7–1.9× quantized number**, not 4.1× bf16-small. No Swift port exists → would need porting the draft model + block-diffusion loop.
- *Fold-in (not standalone):* a SWA long-context draft variant (Z-Lab checkpoint) claimed to hold accept-length past 4K — evaluate only within DFlash if pursued.

**2. omlx oQ4e — importance-calibrated (imatrix) weight quantization — PRIORITY: MEDIUM**
- *What:* activation-importance / imatrix-calibrated **weight** quant (incl. fractional bit levels), vs uniform affine. From omlx v0.5.x release notes.
- *Why:* a **novel axis** — our quant backlog covers **KV cache** (affine + Hadamard "turbo"), not importance-calibrated **weights**. Directly on-strategy for the **measured precision-loss dial** (the white space). It's an *offline checkpoint-production* step — the Swift decode loop just consumes a differently-quantized checkpoint (no loop change), so cheap to investigate.
- *Action:* scope real accuracy-vs-size numbers (not extractable from release notes) before harness time. Not urgent.

**3. Google TurboQuant — data-free near-optimal KV-cache vector quantization — PRIORITY: HIGH (owner-requested)**
- *What:* Google Research's TurboQuant ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874); [blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)) — an **online, data-free vector quantizer**: random-rotate → coordinates become ~Beta-distributed → apply a closed-form per-coordinate **optimal** scalar quantizer (within ~2.7× of the information-theoretic distortion bound). A `_prod` variant adds a 1-bit QJL residual for unbiased Q·K inner products. Evaluated on **KV cache**: quality-neutral at **3.5 bits/channel**, marginal at **2.5-bit**.
- *Why (owner-requested — corrects the earlier mix-up):* this is the **actual** Google TurboQuant the owner meant — **not** mlx-serve's `turbo2/4`, which was our own Hadamard-rotation-then-affine (QuaRot-family) scheme mislabeled "turboquant." The real distinction that could matter: TurboQuant's distribution-optimal non-uniform quantizer **+ zero stored per-group scale/zero metadata**, vs our affine scheme's ~25%-of-bit-budget metadata tax at group-64/2-bit, plus the inner-product debiasing.
- *Where it wins:* the **2-bit / max-context tier** (512GB long-context). At 4-bit our existing rotation+affine KV-quant is already near-lossless, so TurboQuant is ~lateral there. So implement + quantify **specifically for the aggressive 2-bit KV tier**, and promote only if the harness shows it beats `turbo4` / our 2-bit on the measured KL/perplexity frontier at equal bits.
- *Effort:* port the Beta-optimal quantizer (+ optional QJL residual) from the paper — materially more than affine RTN; data-free (no calibration). No known MLX/Swift implementation. A natural **first real customer of the `kl` metric** (harness Task 8) once the spine lands.
- **UPDATE 2026-07-09 — DONE + SHELVED.** Built exactly (Spike A verified vs the paper's distortion table), integrated, measured — **uniform-v1 loses to 4-bit affine on Qwen3-32B** (tqB3 tail-p95 1.797/ppl +32.6% vs 1.665/+21.4%). Shelved as a dated negative result; gated next step is **Spike B (outlier channels)**, the recipe the paper's near-losslessness actually depends on. Verdict: `docs/superpowers/verdicts/2026-07-09-turboquant-firstrun.md`.

**4. PrismML — 1-bit weight quantization — PRIORITY: RESEARCH-LATER (owner-flagged)**
- *What:* an extreme (~1-bit) **weight** quantization scheme (owner reference: "PrismML 1-bit"). Concrete algorithm, primary source, and real-model numbers **not yet researched** — placeholder pending a docs-researcher pass (confirm it's real, the method, whether it's post-hoc or needs QAT, and quantized-real-model accuracy-vs-size).
- *Why it's on-strategy:* the **device/footprint frontier** of the dial's *informed-consent* region (spec §4) — extreme compression that trades *noticeable* quality for *runs-at-all* on a small Mac. Exactly the "massively smaller footprint for a stated loss, gated against garbage" case: 1-bit is unlikely to clear the "unnoticeable" bar, but can still deliver value where the alternative is the model not fitting **— if it stays above the coherence/garbage floor.**
- *GATE:* research the method + real numbers first (is 1-bit even *coherent* on our model classes, or does it fall below the garbage floor? 1-bit typically needs training-time support (QAT/BitNet-style), not post-hoc RTN — confirm). Then, if promising, implement behind a flag and quantify through the harness as a **device-tier** candidate — judged not on "beats fp16" (it won't) but on "coherent + useful at a footprint nothing else reaches." Adjacent to oQ4e (weight-quant axis).

### Verdicts — do NOT re-evaluate (already-have / superseded / marketing)
- **MTPLX** ([youssofal/MTPLX](https://github.com/youssofal/MTPLX), Python+Swift, very active, "fastest") — native MTP heads + Leviathan-Chen exact rejection sampling = **already-have** (our Qwen native MTP + L1 batched accept-walk). Real bench infra (120+ result JSONs), zero architectural novelty. **Value: competitive datapoint only** (validates native-MTP nets ~1.6–2.2× on Qwen-class). Its own detailed telemetry: Qwen3.6-27B @ M5 Max 128GB = 14.16 tok/s decode, 78% accept — a single-condition file, not a comparison pair.
- **omlx "Lightning MTP"** — already-have (native MTP); our `MLX.compile` step arguably goes further than its "verify-shape kernels."
- **omlx "Adaptive Burst Decode"** (reduces per-token executor overhead) — **SUPERSEDED**: we eliminated 81.7% of per-step cost (per-token graph rebuild) outright today via `MLX.compile` + `CompiledKVCache`, rather than batching around it.
- **omlx SSD/paged KV cache** — already-have (ds4 SSD streaming for big MoE).
- **omlx DeepSeek-V4 MXFP4 MoE + sparse-attention Metal kernels** — narrow; revisit only if DeepSeek-V4-Flash is near-term on the catalog.
- **auto-depth calibration** (both MTPLX + omlx: search MTP/draft depth per model/HW) — minor incremental; we already have dynamic self-management (PLD yield-gate/re-enable). LOW.

### Honesty / hygiene flags
- **DFlash provenance is the weakest** of the three: Reddit was unreachable via every path (direct, old.reddit, .json, mirrors), so the "Reddit" framing rests on an *unverified* best-match thread (`1skesyq`) relayed through a content-farm aggregator with no real comment text recovered. **But DFlash itself clears the bar** — real arXiv paper + two independent MLX ports.
- **Every benchmark here is vendor-self-reported / unverified**; the omlx numbers are self-*referential* (vs its own prior version, not vs stock mlx-lm). Quantify independently before adoption.
- A **prompt-injection artifact** (fake "system-reminder" pushing computer-use tools) was embedded in a fetched omlx page and ignored — flagging for hygiene.

### Sources
DFlash: [arXiv:2602.06036](https://arxiv.org/abs/2602.06036) · [bstnxbt/dflash-mlx](https://github.com/bstnxbt/dflash-mlx) · [Aryagm/dflash-mlx](https://github.com/Aryagm/dflash-mlx) · Reddit shortlink (unresolved) `reddit.com/r/LocalLLaMA/s/hU6pN7yPe4` · best-match thread (unverified) `.../comments/1skesyq/`. MTPLX: [youssofal/MTPLX](https://github.com/youssofal/MTPLX). omlx: [jundot/omlx/releases](https://github.com/jundot/omlx/releases) · [v0.5.0.dev4](https://github.com/jundot/omlx/releases/tag/v0.5.0.dev4).
