# Frontier MoE scale-out plan — models too big for a 128 GB Mac

**Purpose.** Plan mlx-serve performance support + a test methodology for the current frontier open-weight models that **do not fit the 128 GB M5** — **DeepSeek-V4**, **GLM-5.2**, **Kimi-K2.7-Code** — so they can be evaluated on larger hardware (Mac Studio 512 GB / multi-GPU / multi-node). All specs below are graded ✅ (verified against raw HuggingFace `config.json`/`LICENSE`/API) or ⚠️ (reported-only — confirm the model card before committing commercially). Full research notes: `scratchpad/BIG_MODELS_RESEARCH.md`.

## The models (verified)

| Model | Total / active | Arch | Context | License | 4-bit size | Min hardware |
|---|---|---|---|---|---|---|
| **DeepSeek-V4-Flash** | 284B / 13B | `deepseek_v4` MoE, CSA+HCA attn, native MTP | 1M | ✅ MIT | ~151 GB (MLX) / 86.7 GB (2-bit GGUF) | 96 GB Mac (2-bit) / 256 GB (4-bit) |
| **DeepSeek-V4-Pro** | 1.6T / 49B | `deepseek_v4` MoE | 1M | ✅ MIT | ~465 GB (2-bit) / ~900 GB (4-bit) | 512 GB Mac / 2× over TB5 |
| **GLM-5.2** | ~743B / 41B | MoE, MLA + DeepSeek Sparse Attention | 1M | ✅ MIT | ~370 GB | remote/cluster |
| **Kimi-K2.7-Code** | 1T / 32B | `kimi_k25`, DeepSeek-V3 MLA text tower, +vision/video | 256K | ✅ Modified-MIT¹ | ~585 GB | remote/cluster |

¹ Modified-MIT: display "Kimi K2.7 Code" attribution only if >100M MAU or >$20M/mo revenue. Bare "Kimi-K2.7" does not exist — the release is **-Code** only.

## The strategic finding

**All three are DeepSeek-lineage MLA/MoE architectures, and DSpark/DFlash drafters exist for all three** (`DeepSeek-V4-Flash-DSpark`, `RedHatAI/GLM-5.2-speculator.dspark-preview`, and the DFlash family). Two consequences for mlx-serve:

1. **mlx-serve's MLX path has no MLA attention arm.** The only MLA family it serves today is **DeepSeek-V4, via the embedded `ds4` C engine (GGUF)** — not the safetensors MLX dispatch. So GLM-5.2 and Kimi-K2.7-Code would each need **net-new architecture support** (MLA + the model's sparse-attention variant + sigmoid `noaux_tc` routing + YaRN; Kimi also a vision tower) before they load at all — independent of the memory question.
2. **The Track-B DSpark port is strategic future-proofing.** The same external-drafter mechanism accelerates the *entire* frontier MoE cohort. Once a base arch is supported, its published DSpark drafter drops in. This is the highest-leverage forward investment for the frontier tier.

## Per-model serving path + applicable perf work

### DeepSeek-V4-Flash — servable on the M5 *today*
- **Path:** embedded `ds4` engine (`lib/ds4`), GGUF 2-bit (86.7 GB) fits 128 GB via SSD streaming (already shipped — issue #39). 4-bit (151–164 GB) needs a 256 GB+ box.
- **Perf levers:** (a) **integrate DSpark into ds4** — `DeepSeek-V4-Flash-DSpark` exists; ds4 today has only a weak native MTP path ("slight speedup" per its README). This is the single biggest ds4 decode win. (b) The shipped **L1/L3 sampler wins do NOT apply** — they live in the MLX `generate.zig` path, not the ds4 C stack. ds4's speculative path is separate C code.
- **Test on bigger HW:** 4-bit Flash on a 256 GB+ Mac; measure decode tok/s + DSpark acceptance vs the native-MTP baseline.

### DeepSeek-V4-Pro — distributed only
- **Path:** `ds4` supports 2-machine Thunderbolt-5 layer-split (~900 GB 4-bit across 2× 512 GB). Measured ~11.5 tok/s in ds4's own tables.
- **Perf levers:** same as Flash (DSpark-into-ds4) + the distributed-serving tuning that already exists.

### GLM-5.2 / Kimi-K2.7-Code — need net-new arch first
- **Path:** no mlx-serve support today. Options: (a) add an MLA/MoE arm to the MLX safetensors path (large — MLA attention, the model's sparse-attention variant, sigmoid routing, YaRN; Kimi + vision); (b) route via the embedded **llama.cpp engine** IF a GGUF quant exists AND llama.cpp has the arch (GLM/Kimi GGUFs exist via unsloth; verify llama.cpp arch support before assuming).
- **Perf levers (post-arch):** the model's DSpark drafter (both have one); MLA is already KV-efficient. L1/L3 apply but are marginal on a bandwidth/dispatch-bound giant MoE.
- **Hardware:** 370–600 GB at 4-bit → 512 GB Mac Studio (if still pur-chasable), multi-GPU (8×H200 / L40S class), or a multi-node cluster. Not a single consumer box.

## Test plan for additional hardware

**Prereq gate (do before sizing anything):** re-verify each model card (license + params) with a raw `curl` of `config.json`/`LICENSE` — a first WebFetch pass hallucinated contradictory param counts on all three; only raw/API sources are trustworthy. Avoid the GLM typosquat GitHub repos.

**Harness:** reuse this run's `perf_probe.py` (streaming, temp>0/concurrency-aware) + the `bench.sh` cells + the equivalence suites. All engine-agnostic over the OpenAI API.

**Per target, measure:**
1. **Load + single-stream decode** at the fitting quant (Q4_K_M floor for tool-calling; 2-bit only for capacity tests). Report tok/s + TTFT + peak unified-memory/VRAM.
2. **Long-context decode** (`perf_probe --prefill-repeat` at 16K/32K/128K) — these are 256K–1M-context models; the KV-bandwidth regime is where L2 (kv-quant) and MLA efficiency matter.
3. **Spec-decode**: DSpark drafter acceptance + speedup vs the native/MTP baseline (DeepSeek-V4 via ds4; GLM/Kimi once arch-supported).
4. **Tool-call correctness** on the golden set (Q4 floor — heavier quant erodes tool JSON before chat quality).
5. **Concurrency**: aggregate throughput + p95 tail at batch 2/4/8 (these are MoE → non-batchable in the current mlx-serve MLX path; measure the real ceiling).

**Sequencing recommendation:**
1. **DeepSeek-V4-Flash 4-bit on a 256 GB+ box** — lowest-lift (ds4 already serves it); prove the DSpark-into-ds4 win.
2. **DSpark generalization** — land the Track-B DSpark port (MLX path) + the ds4 DSpark integration; this unlocks the drafter win across the cohort.
3. **GLM-5.2 or Kimi-K2.7-Code arch support** — only if a concrete deployment needs it; scope as a net-new MLA/MoE arm (or a llama.cpp-GGUF route if arch-supported). Largest lift; defer until hardware + demand are real.

## What NOT to do (honest guardrails)
- Do not put any ⚠️ spec/license into a commercial build without a raw-source re-check.
- Do not expect the L1/L3 sampler wins to matter on these giant models — they're for fast models; the frontier levers are drafters + MTP + quantization + MLA.
- One box is not a SaaS backend; these targets are the "premium/cluster tier," per the concierge scaling doc.
