# Reading the quality metrics — what we measure, and what "good" means

**Audience:** a technical reader who knows LLMs in general but not the specific vocabulary of quantization-quality measurement. **Goal:** explain what each number in our harness means, and what range of each is generally considered *imperceptible* to a user.

fast-mlx's product is a **dial**: turn up the compression (smaller/faster), and see *exactly* how much accuracy you trade. That promise is only as good as the yardstick behind it. This document is the yardstick's user manual.

---

## 1. The problem: precision loss is invisible until it isn't

When you quantize a model (store weights or the KV cache in fewer bits), the output usually *looks* fine — fluent, on-topic, confident. That's the trap. A quantized model can produce perfectly grammatical text while being subtly, or occasionally badly, *wrong* — a flipped digit in a calculation, a different function name, a broken JSON field. You cannot eyeball this. You need to measure how far the compressed model's **probability distribution over the next token** has drifted from a high-precision version of the same model.

Everything below is a way of quantifying that drift.

## 2. The setup: drift from a high-precision reference, on locked context

Two ideas underpin every metric:

**A reference model.** We run the compressed candidate *and* a high-precision **reference** (bf16 — 16-bit floating point, our stand-in for "the model at full quality") on the **same prompts**, and compare their next-token predictions. The metrics measure how far the candidate has moved *away from* that reference. So they are **relative** measures of divergence, not absolute quality scores — the reference defines "perfect."

**Teacher-forcing (context-locking).** To compare fairly, we feed **both** models the **same** token sequence and compare their predictions *position by position*. We do **not** let each model generate freely and then compare — because the moment they pick a different token they're continuing *different sentences*, and every comparison after that point is between unrelated contexts. That produces a chaotic, meaningless number that gets *worse* exactly at the aggressive-compression frontier we care about. (We learned this the hard way; see `docs/content/2026-07-09-trusting-the-instrument.md`.) Teacher-forcing keeps both models on identical context so every position is a clean apples-to-apples comparison.

---

## 3. The four metrics

### 3.1 KL divergence — the sensitive early-warning signal

**What it is.** At each token position, both models output a full probability distribution over the vocabulary ("30% *the*, 12% *a*, …"). **Kullback–Leibler (KL) divergence** measures how different two such distributions are. We compute it per position and summarize it.

**Units: nats** (natural-log units). **0 = identical distributions.** Intuitively, KL is the extra "surprise" you incur by predicting with the candidate when the truth is the reference's distribution. Rough anchors: **ln 2 ≈ 0.69 nats** is "about one bit" of divergence — a lot; **0.05 nats** is a mild, usually-invisible drift.

**Why it's our primary signal.** It's the cheapest and most *sensitive* metric — it reacts to small distribution shifts long before they show up in generated text. It's the smoke detector.

**We report two summaries, and the split matters:**
- **Median** — the *typical* position. Most tokens in real text are easy ("the", "of", a closing bracket) and both models agree on them, so the median sits low.
- **p95 / tail** — the *worst* positions. Quantization error concentrates in the **hard, high-entropy** positions (a sentence start, a branch in reasoning, a rare word). The median can look great while the tail tells the real story. **The tail is the honest headline** (see §3.4).

### 3.2 Perplexity delta — the interpretable headline

**What it is.** Perplexity is `exp(average negative log-likelihood)` — loosely, "how surprised the model is by this text." Lower is better. We report the **percentage change** in the candidate's perplexity versus the reference's, on the same text. **+1%** means the candidate finds the text 1% more surprising than the full-precision model does.

**Why it's useful.** It's a single, familiar number that trends the same direction as real quality, and it's the metric with the most published intuition (below). It falls out of the same teacher-forced pass as KL, essentially for free.

**Caveat specific to us.** Our perplexity-delta is measured on the reference's own continuation under teacher-forcing — a *sensitive relative* measure of "how much more surprised is the candidate by what the full model would say." That is stricter than the classic "perplexity on a held-out corpus" the published rules of thumb (§4) are stated against, so read our percentages as **conservative**.

### 3.3 Top-1 agreement — greedy-output equivalence

**What it is.** At each position, does the candidate's **single most-likely** next token match the reference's? We report the fraction that agree.

**Why it's useful.** At **temperature 0** (greedy decoding — the model always takes its top token), top-1 agreement directly predicts whether the two models produce **token-identical output**. It's the most concrete "will the user see the same thing" measure.

**Caveat.** Measure it **teacher-forced**, not free-running. Free-running top-1 agreement is chaotic for any lossy setting: one flipped high-entropy token early on cascades, and the rest diverges even if the models are almost identical. A low *free-running* prefix match can hide a genuinely small distributional difference.

### 3.4 Long-context tail-p95 — the worst-position, depth-accruing statistic

**What it is.** For long documents, we take the **95th-percentile** per-position KL *within each document* (the worst 5% of positions), then combine across documents. It is deliberately **tail-biased**.

**Why it exists.** Quantization divergence is a **tail phenomenon that grows with context depth**. Early in a prompt the compressed and full models agree closely; deep into a long context, small per-token KV errors have **compounded through every layer and every prior token**, and the hard positions drift. A median over a 24,000-token document is dominated by the thousands of easy tokens and reads near zero — hiding exactly the degradation the metric exists to catch. The tail-p95 at long context is our **load-bearing long-context quality number**.

---

## 4. What counts as "good" (imperceptible)?

**Honest framing first.** There is **no universal, published "the user won't notice past here" constant.** These metrics are **proxies**; the ultimate arbiters are (a) **task-benchmark accuracy** (MMLU/GSM8K/HumanEval/long-context retrieval deltas within their noise) and (b) **human evaluation**. The ranges below are practitioner rules of thumb, calibrated by *correlation with observed output quality* and cross-checked against this project's own measurements. Treat them as a triage guide, not a guarantee.

| Metric | Imperceptible | Minor (usually fine) | Noticeable | Degraded |
|---|---|---|---|---|
| **Perplexity delta** | **≤ 1%** | 1–5% | 5–15% | > 15% |
| **KL divergence (median, nats)** | ≤ ~0.05 (approaching the ~0.002 noise floor) | 0.05–0.2 | 0.2–0.7 | > 0.7 (≥ ~1 bit) |
| **Top-1 agreement (temp 0)** | ≥ ~99% | 95–99% | 90–95% | < 90% |
| **Long-context tail-p95 (nats)** | within ~2–3× the same-weights floor (~0.004) | up to a few tenths | ~0.5–2 | ≫ 2 (order-of-magnitude) |

**The single most user-facing rule of thumb:** **perplexity delta under ~1%, with task-benchmark deltas inside their confidence intervals, is the practical "unnoticeable-loss" bar** — and it is the gate fast-mlx's dial uses to auto-select its default "fastest with loss you won't notice" setting (KL median ≤ ~0.05 **and** ppl delta ≤ 1% **and** task deltas within benchmark CI).

### Why these are proxies, not promises
- KL/perplexity measure the *average* or *tail* distributional drift. A model can have a tiny average KL and still flip one catastrophic token (the wrong digit) — which is why **task benchmarks and human eval remain the final word**, and why the tail statistic matters more than the median.
- Our numbers are measured *against a reference on locked context*. They are excellent at **ranking** two settings (is tier A better than tier B?) and at **catching regressions**, which is their job in the dial. They are not a substitute for running your actual workload.

### What shifts the bar
- **Use case.** Open-ended chat and brainstorming are **forgiving** — a slightly different word rarely matters. **Code, math, tool-calling, and structured output (JSON/XML) are unforgiving** — a single flipped token breaks a compile, a computation, or a parse. The same KL that's invisible in a poem can be fatal in a function call. Budget your acceptable loss by the *hardest* thing the model does in production.
- **Sampling temperature.** At **temperature > 0**, small distributional drifts wash into the sampling noise you already accept — more forgiving. At **temperature 0** (greedy, common for code/agents), a single flipped argmax changes the output — so top-1 agreement and the tail matter most there.
- **Context length.** Loss accrues with depth. A setting that's clean at 2K can drift at 32K — always check the long-context tail, not just short-prompt medians.

---

## 5. The short version

- We compare a **compressed candidate** against a **full-precision reference**, on **identical (teacher-forced) context**, and measure how far the candidate's next-token predictions drift.
- **KL divergence (nats)** is the sensitive smoke detector; **perplexity delta (%)** is the interpretable headline; **top-1 agreement (%)** is "will the greedy output match"; **long-context tail-p95** is the worst-case that grows with depth.
- **Rule of thumb for "the user won't notice":** perplexity delta **≲ 1%**, KL median **≲ 0.05 nats**, top-1 agreement **≳ 99%**, and task-benchmark deltas within noise. Tighten this for code/math/structured/agentic workloads and for temperature-0 decoding; relax it for casual chat.
- These are **proxies calibrated to correlate with real quality** — trustworthy for ranking settings and catching regressions, but the final bar is task accuracy and human judgment on *your* workload.

## 6. Reference points from this project (concrete calibration)

Measured on Qwen3-32B with our harness (teacher-forced, vs a bf16 reference):

| Comparison | KL median | Long-ctx tail-p95 @24K | Perplexity delta |
|---|---|---|---|
| Same weights vs itself (**noise floor**) | ~0.0013 nats | ~0.004 nats | ~−0.24% |
| 4-bit weights vs bf16 (**the shipping baseline**) | 0.19 nats | 1.665 nats | +21.4% |
| 4-bit weights **+ TurboQuant tqB3 KV** (shelved) | 0.17 nats | 1.797 nats | +32.6% |
| 4-bit weights **+ TurboQuant tqB2 KV** (shelved) | — | 10.09 nats | +488% |

Reading these: the **noise floor** is where "a model vs itself" lands — indistinguishable. The **4-bit baseline** carries real, measurable loss (that's the cost of 4× smaller weights) but is a shipped, accepted trade. A KV-quant tier is judged on the *marginal* drift it adds *on top* of that baseline — tqB3 pushed the long-context tail from 1.665 → 1.797 (worse, at no size win) and was **shelved**; tqB2's tail of 10.09 is an order of magnitude past the baseline — decisively broken. This is the dial's whole point: every one of those rows is a *measured* number, not a vibe.

**One reconciliation** (don't misread the table above): our teacher-forced-relative perplexity runs **higher** than the classic *held-out* perplexity the "≤1%" rule of thumb (§4) is stated against — they are different scales. So the 4-bit baseline's +21.4% is **not** "21× past the imperceptible bar"; a standard held-out-corpus perplexity on the same 4-bit model reads far milder (good 4-bit weight quant is typically within a few percent of fp16 on held-out text). Our metric is deliberately the stricter, more sensitive one because its job in the dial is **ranking settings and catching regressions against a reference**, not producing a marketing perplexity. Use the §4 thresholds to judge the **marginal** loss a dial setting adds versus its own baseline, and cross-check absolute quality with task benchmarks.

*(Baselines: `docs/superpowers/verdicts/`; the teacher-forcing rationale: `docs/content/2026-07-09-trusting-the-instrument.md`; the tail-statistic rationale: `docs/content/2026-07-09-the-wall-that-wasnt.md`.)*
