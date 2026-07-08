# EAGLE-3 Phase 0 Spike — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, on the M5, whether a *faithful* MLX EAGLE-3 draft head on Qwen3-32B-8bit beats the 15.3 tok/s decode baseline — and produce a converter + parity anchor we trust — so we can green/amber/red-gate the production `eagle3.zig` build.

**Architecture:** A throwaway Python spike (adapting kmsalah's mlx-lm EAGLE-3 prototype to the `RedHatAI/Qwen3-32B-speculator.eagle3` head + a Qwen3-32B target) with a torch head-math parity fixture as the objective fidelity gate. This is a **measurement spike, not TDD feature work**: correctness is proven by a parity oracle (cos > 0.99 vs. the authoritative `speculators` reference) and by end-to-end acceptance, not by pre-written unit tests. Spike code is committed to `experiments/eagle3/` for reproducibility but is not shipped.

**Tech Stack:** Python 3.14, MLX 0.31.2 + mlx-lm (on the box), PyTorch + `speculators` + `transformers` (fixture generation only), the `RedHatAI/Qwen3-32B-speculator.eagle3` safetensors head, Qwen3-32B-8bit MLX (already on the box).

**Execution environment:** `llmbench@192.168.1.252` (M5/128 GB). **EVERY** non-interactive SSH command MUST prefix `export PATH=/opt/homebrew/bin:$HOME/Library/Python/3.14/bin:$HOME/.local/bin:$PATH;`. There is **no `timeout`/`gtimeout`** — background long commands with `nohup ... &` + a `kill -0` poll, or rely on the tool timeout. `hf` is at `~/Library/Python/3.14/bin/hf`. No HF token is cached (RedHat head is public, so fine; do NOT depend on any gated repo).

**Box state already confirmed (2026-07-07 recon):** Qwen3-32B-8bit (32 GB) and -4bit (17 GB) present & complete; 3.1 TB free; MLX 0.31.2 + huggingface_hub 1.22 present; **torch NOT installed** (install in Task 0); Llama-3.1-8B absent + gated (deliberately not used — see Design §5 refinement).

**Refinement of the approved design (Design §5 "Step A"):** the "reproduce kmsalah on Llama-3.1-8B first" step is **dropped** (gated model, no token, and it only proves we ran his code, not that his code is correct). It is **replaced** by the stronger, lighter torch head-math parity fixture in Task 4, which uses the authoritative `speculators` reference on Qwen3 and loads only the 3 GB head.

---

## File Structure

All spike code lives under `experiments/eagle3/` in the repo, mirrored to `~/perf-work/eagle3-spike/` on the box via `rsync`.

- `experiments/eagle3/vendor/` — verbatim copies of kmsalah's 4 files (`eagle_convert.py`, `eagle_draft.py`, `eagle_generate.py`, `eagle_bench.py`), unmodified, for provenance.
- `experiments/eagle3/inspect_head.py` — dumps the RedHat safetensors manifest (keys/shapes/dtypes). One responsibility: discovery.
- `experiments/eagle3/convert_redhat.py` — RedHat `speculators` safetensors → MLX head dir (`weights.safetensors` + `config.json`). Adapted from `vendor/eagle_convert.py`.
- `experiments/eagle3/eagle_draft_qwen3.py` — the draft-head module, adapted from `vendor/eagle_draft.py` for Qwen3-32B geometry (explicit head_dim, optional QK-norm, rope_theta).
- `experiments/eagle3/dump_reference_fixture.py` — torch: load RedHat head via `speculators`, dump `(fused_hidden, prev_token) → draft_logits` fixture npz. Fidelity oracle source.
- `experiments/eagle3/check_head_parity.py` — MLX: run `eagle_draft_qwen3` on the fixture inputs, assert cos > 0.99. **The fidelity gate.**
- `experiments/eagle3/eagle_generate_qwen3.py` — target tap + chain generate, adapted from `vendor/eagle_generate.py`.
- `experiments/eagle3/bench_qwen3.py` — baseline vs eagle measurement, adapted from `vendor/eagle_bench.py`; writes `results.csv`.
- `experiments/eagle3/RESULTS.md` — the gate verdict (Task 8), feeds the case study.

---

### Task 0: Spike workspace + provision deps

**Files:**
- Create: `experiments/eagle3/vendor/{eagle_convert,eagle_draft,eagle_generate,eagle_bench}.py`
- Create: `experiments/eagle3/.gitignore` (ignore downloaded weights / fixtures / csv)

- [ ] **Step 1: Create the local workspace and vendor kmsalah's code**

```bash
mkdir -p /Users/braymond/Projects/mlx-serve/experiments/eagle3/vendor
cd /Users/braymond/Projects/mlx-serve/experiments/eagle3/vendor
for f in eagle_convert eagle_draft eagle_generate eagle_bench; do
  curl -fsSL "https://raw.githubusercontent.com/kmsalah/mlx/eagle3-speculative-decoding/python/$f.py" -o "$f.py"
done
ls -la   # expect 4 non-empty .py files
```

Expected: 4 files, each > 3 KB.

- [ ] **Step 2: Add a .gitignore so weights/fixtures/results aren't committed**

```bash
cat > /Users/braymond/Projects/mlx-serve/experiments/eagle3/.gitignore <<'EOF'
# spike artifacts — never commit weights or generated data
redhat-head/
eagle3-mlx/
*.npz
results.csv
__pycache__/
EOF
```

- [ ] **Step 3: Provision the box — download the RedHat head (public, no token)**

```bash
ssh llmbench@192.168.1.252 'export PATH=/opt/homebrew/bin:$HOME/Library/Python/3.14/bin:$HOME/.local/bin:$PATH; \
  mkdir -p ~/perf-work/eagle3-spike/redhat-head && cd ~/perf-work/eagle3-spike && \
  nohup hf download RedHatAI/Qwen3-32B-speculator.eagle3 \
    --local-dir ./redhat-head > dl_head.log 2>&1 &'
```

Poll until the `model.safetensors` is fully present (guard against the incomplete-download trap — check the real multi-GB file, NOT config.json):

```bash
ssh llmbench@192.168.1.252 'export PATH=/opt/homebrew/bin:$PATH; \
  ls -la ~/perf-work/eagle3-spike/redhat-head/ ; \
  du -h ~/perf-work/eagle3-spike/redhat-head/model.safetensors 2>/dev/null'
```

Expected when done: `model.safetensors` ≈ 3.1 GB (2,900–3,200 MB), plus `config.json`. If `model.safetensors` is missing or < 3 GB, the download is still running or stalled — re-check `dl_head.log`, and if stalled, kill the `hf download` PID, delete `*.lock`/`*.incomplete`, re-run.

- [ ] **Step 4: Provision the box — install torch + speculators + transformers (fixture generation only)**

```bash
ssh llmbench@192.168.1.252 'export PATH=/opt/homebrew/bin:$HOME/Library/Python/3.14/bin:$HOME/.local/bin:$PATH; \
  nohup pip3 install --user torch transformers safetensors speculators > ~/perf-work/eagle3-spike/pip.log 2>&1 &'
```

Verify (may take a few minutes for torch to install):

```bash
ssh llmbench@192.168.1.252 'export PATH=/opt/homebrew/bin:$HOME/Library/Python/3.14/bin:$HOME/.local/bin:$PATH; \
  python3 -c "import torch, transformers, safetensors; print(torch.__version__, transformers.__version__)"; \
  python3 -c "import speculators; print(\"speculators\", getattr(speculators,\"__version__\",\"?\"))"'
```

Expected: torch + transformers versions print. If `speculators` import fails, note it — Task 4 has a hand-written-reference fallback, but the real class is strongly preferred.

- [ ] **Step 5: Commit the vendored code + gitignore**

```bash
cd /Users/braymond/Projects/mlx-serve
git add experiments/eagle3/vendor experiments/eagle3/.gitignore
git commit -m "spike(eagle3): vendor kmsalah MLX EAGLE-3 prototype + workspace"
```

---

### Task 1: Inspect the RedHat head's real tensor manifest (the critical unknown)

**Files:**
- Create: `experiments/eagle3/inspect_head.py`

- [ ] **Step 1: Write the manifest dumper**

```python
# experiments/eagle3/inspect_head.py
"""Dump every tensor key/shape/dtype from a safetensors head, plus its config.
Resolves the three port risks: exact key names, presence of q_norm/k_norm, fc shape."""
import json, sys
from safetensors import safe_open

head_dir = sys.argv[1] if len(sys.argv) > 1 else "redhat-head"
with safe_open(f"{head_dir}/model.safetensors", framework="pt") as f:
    keys = sorted(f.keys())
    print(f"# {len(keys)} tensors")
    for k in keys:
        t = f.get_slice(k)
        print(f"{k:60s} {str(t.get_shape()):20s} {t.get_dtype()}")
print("\n# has q_norm/k_norm:", any("q_norm" in k or "k_norm" in k for k in keys))
print("# fc keys:", [k for k in keys if "fc" in k])
print("# vocab buffers:", [k for k in keys if k in ("d2t", "t2d")])
cfg = json.load(open(f"{head_dir}/config.json"))
print("\n# config.json:")
print(json.dumps(cfg, indent=2)[:2000])
```

- [ ] **Step 2: Run it on the box and CAPTURE the output into notes**

```bash
rsync -az /Users/braymond/Projects/mlx-serve/experiments/eagle3/inspect_head.py \
  llmbench@192.168.1.252:~/perf-work/eagle3-spike/
ssh llmbench@192.168.1.252 'export PATH=/opt/homebrew/bin:$HOME/Library/Python/3.14/bin:$HOME/.local/bin:$PATH; \
  cd ~/perf-work/eagle3-spike && python3 inspect_head.py redhat-head' | tee /tmp/redhat_manifest.txt
```

Expected: a full key list. **Record the exact key names** (e.g. does the decoder block live under `layers.0.self_attn.q_proj.weight`, `midlayer.self_attn.q_proj.weight`, or `transformer.*`?), whether `q_norm`/`k_norm` weights exist, the `fc.weight` shape (expect `[5120, 15360]`), `lm_head.weight` (`[32000, 5120]`), `embed_tokens.weight` (`[151936, 5120]`), `d2t` (int64×32000), `t2d` (bool×151936). This output drives Tasks 2 and 3 — paste it into `experiments/eagle3/RESULTS.md` under a "Manifest" heading.

- [ ] **Step 3: Commit the inspector**

```bash
cd /Users/braymond/Projects/mlx-serve
git add experiments/eagle3/inspect_head.py
git commit -m "spike(eagle3): safetensors manifest inspector"
```

---

### Task 2: Converter — RedHat `speculators` safetensors → MLX head dir

**Files:**
- Create: `experiments/eagle3/convert_redhat.py` (adapt `vendor/eagle_convert.py`)

- [ ] **Step 1: Write the converter for the nested speculators schema + Task-1 key names**

Key differences from `vendor/eagle_convert.py` (which assumes flat schema + no remap): (a) parse `config.json` `transformer_layer_config` for `hidden_size`, `num_attention_heads`, `num_key_value_heads`, **`head_dim`**, `intermediate_size`, `rms_norm_eps`, `rope_theta`, `vocab_size`; (b) apply the key remap discovered in Task 1 so the output keys match `eagle_draft_qwen3` module names; (c) keep `d2t` (int64) / `t2d` (bool) native dtype, downcast float weights to... **keep bf16, not f16** — the head is bf16 and the target runs bf16; write the MLX config with `hidden_state_layers` defaulted to `[2, 32, 61]` (Qwen3-32B N=64 → `{2, N//2, N-3}`) since the RedHat config omits `eagle_aux_hidden_state_layer_ids`.

```python
# experiments/eagle3/convert_redhat.py
import json, sys, numpy as np, mlx.core as mx
from safetensors import safe_open

src = sys.argv[1] if len(sys.argv) > 1 else "redhat-head"
out = sys.argv[2] if len(sys.argv) > 2 else "eagle3-mlx"

raw = json.load(open(f"{src}/config.json"))
tlc = raw["transformer_layer_config"]

# --- KEY REMAP: fill from Task 1 manifest. Map source key -> module-tree key. ---
# Example (ADJUST to the real manifest): speculators ships flat `layers.0.*`;
# eagle_draft_qwen3 expects `midlayer.*`. If names already match, use identity.
def remap(k: str) -> str:
    k = k.replace("layers.0.", "midlayer.")
    return k

weights = {}
with safe_open(f"{src}/model.safetensors", framework="numpy") as f:
    for k in f.keys():
        arr = f.get_tensor(k)
        if k in ("d2t", "t2d"):
            weights[k] = mx.array(arr)            # keep native int64/bool
        else:
            weights[remap(k)] = mx.array(arr)     # bf16 preserved by mlx

import os; os.makedirs(out, exist_ok=True)
mx.save_safetensors(f"{out}/weights.safetensors", weights)

cfg = {
    "hidden_size": tlc["hidden_size"],
    "intermediate_size": tlc["intermediate_size"],
    "num_attention_heads": tlc["num_attention_heads"],
    "num_key_value_heads": tlc["num_key_value_heads"],
    "head_dim": tlc["head_dim"],                       # EXPLICIT — do not derive
    "rms_norm_eps": tlc["rms_norm_eps"],
    "rope_theta": tlc["rope_theta"],
    "draft_vocab_size": int(weights["lm_head.weight"].shape[0]),
    "target_vocab_size": int(weights["t2d"].shape[0]),
    "has_qk_norm": any("q_norm" in k or "k_norm" in k for k in weights),
    "hidden_state_layers": [2, 32, 61],
}
json.dump(cfg, open(f"{out}/config.json", "w"), indent=2)
print("wrote", out, cfg)
```

- [ ] **Step 2: Run the converter on the box; verify shapes**

```bash
rsync -az /Users/braymond/Projects/mlx-serve/experiments/eagle3/convert_redhat.py \
  llmbench@192.168.1.252:~/perf-work/eagle3-spike/
ssh llmbench@192.168.1.252 'export PATH=/opt/homebrew/bin:$HOME/Library/Python/3.14/bin:$HOME/.local/bin:$PATH; \
  cd ~/perf-work/eagle3-spike && python3 convert_redhat.py redhat-head eagle3-mlx'
```

Expected: prints `wrote eagle3-mlx {...}` with `head_dim: 128`, `draft_vocab_size: 32000`, `target_vocab_size: 151936`, `has_qk_norm: <true/false>`. If a `remap` key doesn't resolve (KeyError on `lm_head.weight`), the remap table is wrong — fix against the Task-1 manifest.

- [ ] **Step 3: Commit**

```bash
cd /Users/braymond/Projects/mlx-serve
git add experiments/eagle3/convert_redhat.py
git commit -m "spike(eagle3): RedHat speculators->MLX head converter"
```

---

### Task 3: Adapt the draft head module for Qwen3-32B geometry

**Files:**
- Create: `experiments/eagle3/eagle_draft_qwen3.py` (adapt `vendor/eagle_draft.py`)

- [ ] **Step 1: Copy `vendor/eagle_draft.py` and apply the three Qwen3 fixes**

Concrete changes (the rest of the module — `fc`, the concat-and-residual-on-fused forward, SwiGLU MLP — stays as `vendor/eagle_draft.py`):

1. **Explicit head_dim** — in `EAGLEAttention.__init__`, replace `self.head_dim = config.hidden_size // config.num_attention_heads` with `self.head_dim = config.head_dim`. Q proj output = `num_attention_heads * head_dim` (= 64×128 = 8192), K/V proj output = `num_key_value_heads * head_dim` (= 8×128 = 1024). Input stays `2 * hidden_size` (= 10240).
2. **Optional Qwen3 QK-norm** — if `config.has_qk_norm`, add `self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)` and `self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)`, and in `__call__` apply them per-head to q and k AFTER reshape to heads and BEFORE RoPE (mirrors mlx-lm's Qwen3 attention order). If `has_qk_norm` is false, this is a no-op.
3. **rope_theta / n_rep from config** — ensure `RoPE(dims=self.head_dim, base=config.rope_theta)` uses `config.rope_theta` (1e6), and GQA `n_rep = num_attention_heads // num_key_value_heads` (= 8).

Add the matching fields to the `EAGLEConfig` dataclass: `head_dim: int`, `has_qk_norm: bool = False` (and ensure it's populated from the converted `config.json`).

- [ ] **Step 2: No standalone test — correctness is proven by Task 4.**

This module is exercised by the parity gate; do not hand-write a unit test whose expected values you'd be guessing. Commit as-is.

```bash
cd /Users/braymond/Projects/mlx-serve
git add experiments/eagle3/eagle_draft_qwen3.py
git commit -m "spike(eagle3): draft head adapted for Qwen3-32B (head_dim, QK-norm, rope)"
```

---

### Task 4: Head-math parity fixture — THE fidelity gate

**Files:**
- Create: `experiments/eagle3/dump_reference_fixture.py` (torch)
- Create: `experiments/eagle3/check_head_parity.py` (MLX)

- [ ] **Step 1: Write the torch reference dumper (authoritative `speculators` head)**

Load the RedHat head via the real `Eagle3Speculator`/draft submodule, feed a FIXED-seed input, dump draft logits. Feed the decoder-layer input directly (post-`fc` fused hidden of width `hidden_size`, plus the token-embedding input the midlayer concatenates) so the fixture isolates the head math. If the `speculators` API can't be driven head-only, fall back to constructing the head forward from the loaded state_dict per `vendor/eagle_draft.py`'s math (documented risk: a hand reference could share a bug — prefer the real class).

```python
# experiments/eagle3/dump_reference_fixture.py
import numpy as np, torch, json
torch.manual_seed(0)
src = "redhat-head"
cfg = json.load(open(f"{src}/config.json")); tlc = cfg["transformer_layer_config"]
H = tlc["hidden_size"]; L = 4
# fixed inputs: 3 tapped hiddens (pre-fc) + prev token ids for the embedding path
aux = torch.randn(1, L, 3 * H, dtype=torch.bfloat16)      # concat of 3 target-layer hiddens
prev_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
# --- load the RedHat head with the real speculators class and run its draft forward ---
# (fill in per speculators API: build Eagle3Speculator from redhat-head, call the
#  draft model's forward with (aux, prev_ids) to get draft_logits [1, L, 32000])
# draft_logits = ...
np.savez("fixture.npz",
         aux=aux.float().numpy(), prev_ids=prev_ids.numpy(),
         draft_logits=draft_logits.float().numpy())
print("wrote fixture.npz", draft_logits.shape)
```

- [ ] **Step 2: Run the dumper on the box (loads only the 3 GB head, not the target)**

```bash
rsync -az /Users/braymond/Projects/mlx-serve/experiments/eagle3/dump_reference_fixture.py \
  llmbench@192.168.1.252:~/perf-work/eagle3-spike/
ssh llmbench@192.168.1.252 'export PATH=/opt/homebrew/bin:$HOME/Library/Python/3.14/bin:$HOME/.local/bin:$PATH; \
  cd ~/perf-work/eagle3-spike && python3 dump_reference_fixture.py'
```

Expected: `wrote fixture.npz (1, 4, 32000)`.

- [ ] **Step 3: Write the MLX parity checker (the gate)**

```python
# experiments/eagle3/check_head_parity.py
import numpy as np, mlx.core as mx, json
from eagle_draft_qwen3 import EAGLEDraftModel, EAGLEConfig
fx = np.load("fixture.npz")
cfg_d = json.load(open("eagle3-mlx/config.json"))
model = EAGLEDraftModel(EAGLEConfig(**{k: cfg_d[k] for k in EAGLEConfig.__dataclass_fields__ if k in cfg_d}))
model.load_weights("eagle3-mlx/weights.safetensors")
H = cfg_d["hidden_size"]
aux = mx.array(fx["aux"]); prev = mx.array(fx["prev_ids"])
# split aux into the 3 tapped hiddens the model.fc expects
hs = [aux[..., i*H:(i+1)*H] for i in range(3)]
embed = model.embed_tokens(prev) if hasattr(model, "embed_tokens") else None
logits, _, _ = model(embed, hs)   # match eagle_draft_qwen3's __call__ signature
mine = np.array(logits.astype(mx.float32)); ref = fx["draft_logits"]
cos = float((mine.ravel() @ ref.ravel()) /
            (np.linalg.norm(mine) * np.linalg.norm(ref) + 1e-9))
print(f"cos={cos:.6f}  argmax_match={(mine.argmax(-1)==ref.argmax(-1)).mean():.3f}")
assert cos > 0.99, f"PARITY FAIL cos={cos} — port bug (key remap / head_dim / QK-norm / concat order)"
print("PARITY PASS")
```

- [ ] **Step 4: Run the gate**

```bash
rsync -az /Users/braymond/Projects/mlx-serve/experiments/eagle3/{check_head_parity.py,eagle_draft_qwen3.py} \
  llmbench@192.168.1.252:~/perf-work/eagle3-spike/
ssh llmbench@192.168.1.252 'export PATH=/opt/homebrew/bin:$HOME/Library/Python/3.14/bin:$HOME/.local/bin:$PATH; \
  cd ~/perf-work/eagle3-spike && python3 check_head_parity.py'
```

Expected: `cos > 0.99` and `PARITY PASS`. **If it fails**, the port has a bug — the failure isolates to the head math (independent of tap/target), so debug against the three known risks in order: (1) key remap wrong (Task 1/2), (2) head_dim not 128, (3) QK-norm missing/misordered, (4) embed⊕fused concat order or residual-on-fused vs residual-on-embed. Do NOT proceed to Task 5 until this passes — a low end-to-end acceptance is only interpretable once the head math is proven faithful.

- [ ] **Step 5: Commit both + record the parity number in RESULTS.md**

```bash
cd /Users/braymond/Projects/mlx-serve
git add experiments/eagle3/dump_reference_fixture.py experiments/eagle3/check_head_parity.py
git commit -m "spike(eagle3): torch head-math parity fixture (fidelity gate, cos>0.99)"
```

---

### Task 5: Wire the target tap + chain generation

**Files:**
- Create: `experiments/eagle3/eagle_generate_qwen3.py` (adapt `vendor/eagle_generate.py`)

- [ ] **Step 1: Adapt `forward_with_hidden_states` + `load_eagle_model` + the chain loop for Qwen3**

Changes from `vendor/eagle_generate.py`: (a) `capture_layers=(2, 32, 61)` (from the converted config); (b) confirm the mlx-lm Qwen3 module layout matches the assumed `model.model.{embed_tokens,layers,norm}` / `model.lm_head` (mlx-lm standardizes this — verify with a quick `dir()` if the forward errors); (c) Qwen3-32B dense has no sliding-window layers, so the `use_sliding`/`swa_idx` shim naturally no-ops; (d) keep the `num_draft=1` optimized single-sync path AND the `num_draft>1` path; (e) `d2t` application unchanged (`d_target = d_vocab + d2t[d_vocab]`), `t2d` ignored (confirmed inert); (f) greedy accept (`target_preds[i] == draft[i]`) — matches a temp-0 baseline.

- [ ] **Step 2: Smoke-run: generate 64 tokens, confirm coherence + non-zero acceptance**

```bash
rsync -az /Users/braymond/Projects/mlx-serve/experiments/eagle3/eagle_generate_qwen3.py \
  llmbench@192.168.1.252:~/perf-work/eagle3-spike/
ssh llmbench@192.168.1.252 'export PATH=/opt/homebrew/bin:$HOME/Library/Python/3.14/bin:$HOME/.local/bin:$PATH; \
  cd ~/perf-work/eagle3-spike && python3 eagle_generate_qwen3.py \
    --model ~/perf-work/models/Qwen3-32B-8bit --eagle-model eagle3-mlx \
    --prompt "Write a Python function that reverses a linked list." \
    --max-tokens 64 --num-draft 1'
```

Expected: coherent code output; a printed acceptance rate that is **not** ~0 (a healthy head should be well above the kmsalah-8B 0.34). If the text is garbage, the tap layers or fused-input wiring is wrong even though head-parity passed — revisit capture point (pre- vs post-layer) and layer indices (see Task 7).

- [ ] **Step 3: Commit**

```bash
cd /Users/braymond/Projects/mlx-serve
git add experiments/eagle3/eagle_generate_qwen3.py
git commit -m "spike(eagle3): Qwen3-32B target tap + chain generate"
```

---

### Task 6: Measure — the gate numbers

**Files:**
- Create: `experiments/eagle3/bench_qwen3.py` (adapt `vendor/eagle_bench.py`)

- [ ] **Step 1: Adapt the bench harness** — same metric defs as `vendor/eagle_bench.py` (tok/s excludes prefill; acceptance = accepted/draft_steps), a coding + a prose prompt set, sweeping `--num-draft 1` and `--num-draft 3`, plus a pure-baseline (mlx-lm, no draft) row. Write `results.csv` with columns `prompt_kind,num_draft,baseline_tok_s,eagle_tok_s,speedup,accept_per_step`.

- [ ] **Step 2: First validate the harness — confirm our baseline ≈ 15.3 tok/s**

```bash
rsync -az /Users/braymond/Projects/mlx-serve/experiments/eagle3/bench_qwen3.py \
  llmbench@192.168.1.252:~/perf-work/eagle3-spike/
ssh llmbench@192.168.1.252 'export PATH=/opt/homebrew/bin:$HOME/Library/Python/3.14/bin:$HOME/.local/bin:$PATH; \
  cd ~/perf-work/eagle3-spike && python3 bench_qwen3.py \
    --model ~/perf-work/models/Qwen3-32B-8bit --eagle-model eagle3-mlx \
    --max-tokens 256 --warmup 1'
```

Expected: the **baseline** row's tok/s lands near the 15.3 tok/s reference (±1). If it's wildly off, the harness/measurement is wrong — fix before trusting the eagle numbers. Then read the eagle rows: `speedup` and `accept_per_step` for num_draft 1 and 3, coding and prose.

- [ ] **Step 3: Record results + apply the gate**

Paste `results.csv` into `RESULTS.md`. Apply Design §6: **Green** ≥ 1.3× on ≥1 workload with reference-range acceptance; **Amber** healthy acceptance but 1.0–1.3×; **Red** ≤ 1.0× or stuck-low acceptance. Note: the prototype's `num_draft>1` path has per-step `.item()` syncs that understate tok/s — if `num_draft=3` acceptance is high but its tok/s lags, record that an optimized (lazy-pipeline) k=3 — which the Zig path would do — projects higher, and reason explicitly about whether that clears 1.3×.

- [ ] **Step 4: Commit the bench harness**

```bash
cd /Users/braymond/Projects/mlx-serve
git add experiments/eagle3/bench_qwen3.py
git commit -m "spike(eagle3): baseline-vs-eagle bench harness"
```

---

### Task 7: Conditional diagnosis (only if not clearly Green)

**Files:** modify `experiments/eagle3/eagle_generate_qwen3.py`; possibly download Qwen3-32B-bf16.

- [ ] **Step 1: If acceptance is low but head-parity PASSED — the tap is the suspect.** Try, one at a time, re-measuring acceptance after each: (a) **post-layer** capture (`captured[i] = h` AFTER `h = layer(...)`) instead of pre-layer; (b) alternate layer indices (the reference formula is `{2, N//2, N-3}` = `{2,32,61}`, but confirm against the head's training — try `{1, 32, 61}` / `{2, 31, 60}` neighbors only if a principled reason emerges). Head-parity passing means the head is faithful, so acceptance sensitivity here localizes to what the head is *fed*.

- [ ] **Step 2: If acceptance is still low — precision A/B.** Download Qwen3-32B-bf16 (~64 GB, fits) and re-measure acceptance; the head was trained on bf16 target hiddens, so 8-bit quantization of the *tapped* hiddens is a plausible acceptance tax. This isolates quantization from a ceiling.

```bash
ssh llmbench@192.168.1.252 'export PATH=/opt/homebrew/bin:$HOME/Library/Python/3.14/bin:$HOME/.local/bin:$PATH; \
  cd ~/perf-work/models && nohup hf download mlx-community/Qwen3-32B-bf16 \
    --local-dir ./Qwen3-32B-bf16 > ~/perf-work/eagle3-spike/dl_bf16.log 2>&1 &'
```

(Verify completion by the `du -h .../*.safetensors` multi-GB check before benching — incomplete-download trap.)

- [ ] **Step 3: If acceptance is healthy but tok/s < 1.3× — economics.** This is the DSpark story on a slower target; record the measured round-cost ratio and per-step acceptance, and project the optimized-k Zig number. No code change — a written analysis into RESULTS.md.

---

### Task 8: Write the gate verdict

**Files:**
- Create: `experiments/eagle3/RESULTS.md`

- [ ] **Step 1: Write RESULTS.md** — the manifest (Task 1), parity cos (Task 4), the `results.csv` table (Task 6), any Task-7 diagnostics, and the explicit **verdict** (Green/Amber/Red) with a one-paragraph recommendation on whether to build `eagle3.zig`. This doc feeds the perf case study either way (a rigorous negative result is a publishable result).

- [ ] **Step 2: Commit + report the verdict to the user for the gate decision**

```bash
cd /Users/braymond/Projects/mlx-serve
git add experiments/eagle3/RESULTS.md
git commit -m "spike(eagle3): Phase 0 results + gate verdict"
```

**After this task, STOP and bring the verdict to the user.** The production `eagle3.zig` build (Phase 1+) is gated on a Green (or an explicit Amber go-ahead) and gets its own `writing-plans` pass, using the constants measured here.

---

## Self-Review

**Spec coverage:** Design §5 Steps A/B/C → Tasks 0–6 (Step A "reproduce reference" refined to the Task-4 parity fixture, flagged above); target-precision A/B → Task 7 (made conditional per the recon-confirmed absence of a bf16 32B, to avoid a 64 GB download unless diagnostic); §6 gate → Task 6 Step 3 + Task 8; §9 open questions → tap layers/capture point (Task 7), t2d inert (Task 5), tensor keys (Task 1), bf16-vs-8bit (Task 7). Covered.

**Placeholder scan:** the code blocks intentionally leave two spots to fill from discovery — the `remap()` table (Task 2, filled from the Task-1 manifest) and the `speculators` head-forward call (Task 4 Step 1, filled from the real API). These are inherent to a spike against an unseen checkpoint; each is gated by an objective check (converter shape print; parity cos) so "wrong fill" fails loudly rather than silently. Every command is concrete and copy-runnable.

**Type consistency:** `EAGLEConfig` gains `head_dim` + `has_qk_norm` in Task 3 and both are written by the Task-2 converter and read by the Task-4 checker — consistent. `capture_layers=(2,32,61)` consistent between Task 2 config, Task 5 generate, Task 7 tuning. `d2t` additive / `t2d` inert consistent across Tasks 2/4/5.
