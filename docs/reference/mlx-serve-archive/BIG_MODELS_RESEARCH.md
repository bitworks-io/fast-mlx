# Big remote-only models — perf-support research (2026-07-06)

Verification grade: ✅ = raw HF config.json/LICENSE/API (not WebFetch prose); ⚠️ = reported-only.
(A first WebFetch pass produced self-contradictory param counts — a hallucination on thin SPA pages — so the DeepSeek-V4 data below was re-derived from raw curl/API sources.)

## DeepSeek-V4 — ✅ VERIFIED (rich)
- **Exists:** `deepseek-ai/DeepSeek-V4-Flash` + `-Pro` (HTTP 200, real download counts). Bare `DeepSeek-V4` is a HF Collection, not a repo. `-Flash-DSpark` / `-Pro-DSpark` speculative checkpoints exist (mod. 2026-07-04).
- **Arch:** `model_type: deepseek_v4`. Flash **284B / 13B active** (43 layers, 256 routed+1 shared exp, top-6). Pro **1.6T / 49B active** (61 layers, 384 exp). Context **1,048,576 (1M)**.
  - Attention = **CSA (Compressed Sparse Attention) + HCA (Heavily Compressed Attention)** — MLA descendant but distinct (`num_key_value_heads:1`, `q_lora_rank`/`o_lora_rank`, `index_topk` 512/1024, `sliding_window:128`, per-layer `compress_ratios`).
  - **Native MTP head present** (`num_nextn_predict_layers:1`, real `mtp.0.*` weights). Plus **Manifold-Constrained Hyper-Connections (mHC)** (`hc_*` config, real).
  - Precision: FP4 (MoE experts) + FP8 (rest) mixed; `-Base` is FP8-only.
- **License:** ✅ **MIT** (verbatim from LICENSE), plain/unrestricted on code+weights.
- **Caps:** text-only (no vision_config). Tool calling via custom **DSML** format (no Jinja template shipped). Think modes: Non-think / Think High / Think Max.
- **Quants + hardware:**
  - MLX: `mlx-community/DeepSeek-V4-Flash-4bit` = **151.5 GB**; 2/3/5/6/8-bit + MTP-bf16 + FP4-FP8-SSD variants; Pro-4bit/8bit too.
  - GGUF (antirez/deepseek-v4-gguf, ds4's own): Flash IQ2_XXS **86.7 GB**, Flash Q4K **164.6 GB**, Flash MTP sidecar 3.8 GB; Pro IQ2 **464.6 GB**, Pro Q4K split **~900 GB (2 files)**.
  - HW: Flash 2-bit → **96 GB Mac min** (M3 Max 128GB: 26.7 tok/s; **fits the M5 via ds4 SSD streaming**). Flash 4-bit → **256 GB+** (M3 Ultra 512GB works). Pro 2-bit → **512 GB**; Pro 4-bit → **2× 512GB Mac Studios over TB5** (distributed, 11.47 tok/s).
- **mlx-serve fit + perf improvements:**
  - **Already served** by the embedded **`ds4` engine** (lib/ds4) for the `deepseek_v4` GGUF family — Flash-2bit runs on the M5 today.
  - ds4 has its OWN simple MTP sidecar path (experimental, "slight speedup"); it has NOT integrated DeepSeek's DSpark/DFlash. **DSpark drafters for V4 exist** (`DeepSeek-V4-Flash-DSpark`) → integrating DSpark into ds4 is the natural perf improvement (the Track-B DSpark port work is the MLX-side analog; ds4 would need its own C-side integration).
  - L1/L3 sampler wins are in the MLX `generate.zig` path — they do NOT apply to the ds4 engine (separate C stack). So V4's perf levers are ds4-internal (DSpark/MTP), not the shipped L1/L3.

## GLM-5.2 — ✅ VERIFIED (full analysis in docs/big-model-scale-out-plan.md)
`zai-org/GLM-5.2`, ~743B/41B MoE, 78 layers, 256 exp top-8, MLA + DeepSeek Sparse Attention, 1M ctx, **MIT**. No -Air/-Flash. DSpark drafter exists (`RedHatAI/GLM-5.2-speculator.dspark-preview`). Needs net-new MLA/DSA arch in mlx-serve; 4-bit ~370 GB → remote/cluster. ⚠️ typosquat GitHub repos exist.

## Kimi-K2.7-Code — ✅ VERIFIED (bare "Kimi-K2.7" does NOT exist)
- **Exists:** `moonshotai/Kimi-K2.7-Code` (HTTP 200, 864K downloads, mod. 2026-06-15). Bare `Kimi-K2.7` → 401. Only the **-Code** variant shipped at 2.7 (no Base/Instruct/Thinking).
- **Arch:** `model_type: kimi_k25`, text tower `DeepseekV3ForCausalLM`. **1T total / 32B active** MoE (61 layers, 384 routed+1 shared exp, top-8, sigmoid `noaux_tc` routing). **MLA attention** (q_lora 1536, kv_lora 512, qk_nope 128 / qk_rope 64 / v 128 — DeepSeek-V2/V3 MLA fields). Context **256K (YaRN)**. **Native image+video** (vision_config present, MoonViT-class). **NO native MTP** (`num_nextn_predict_layers: 0` — explicit). Native **INT4** (compressed-tensors g32; attn/shared-exp/lm_head/vision stay bf16).
- **License:** ✅ **Modified MIT** — standard MIT + one clause: display "Kimi K2.7 Code" if >100M MAU or >$20M/mo revenue. Permissive for realistic users.
- **Quants + hardware:** native INT4 in-repo. GGUF via unsloth (bf16 ~605 GB; UD-Q4_K_XL ~585 GB; UD-Q2 ~325 GB). **NO `mlx-community` release** — only individual-uploader MLX quants (3.5–5 bpw, ~426–600 GB). 4-bit ≈ **600 GB** → multi-GPU datacenter or multi-Mac cluster; NOT a single-Mac target. Moonshot ships no Apple-Silicon guidance (MLX is community-only).
- **mlx-serve fit + perf:** `model_type: kimi_k25` + DeepSeek-V3 MLA text tower → mlx-serve has **NO `kimi_k25` / MLA / DeepSeek-V3 dispatch arm** → needs **NET-NEW arch support** (MLA attention + noaux_tc sigmoid routing + YaRN + the vision tower) before it can load at all, independent of the ~600 GB question. A community EAGLE-3 drafter exists for **K2.5** (+26–60% throughput) but ⚠️ none confirmed for K2.7-Code. No native MTP → mlx-serve's MTP path would never engage. Realistic mlx-serve path: large lift (new arch), remote/cluster hardware only.

## GLM-5.2 — ✅ VERIFIED (full analysis in docs/big-model-scale-out-plan.md)
`zai-org/GLM-5.2`, ~743B/41B MoE, 78 layers, 256 exp top-8, MLA + DeepSeek Sparse Attention, 1M ctx, **MIT**. No -Air/-Flash. DSpark drafter exists (`RedHatAI/GLM-5.2-speculator.dspark-preview`). Needs net-new MLA/DSA arch in mlx-serve; 4-bit ~370 GB → remote/cluster. ⚠️ typosquat GitHub repos exist.
