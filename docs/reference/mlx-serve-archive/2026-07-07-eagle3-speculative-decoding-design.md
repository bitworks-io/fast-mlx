# EAGLE-3 Speculative Decoding for mlx-serve — Design

- **Date:** 2026-07-07
- **Branch:** `perf/dspark-m2`
- **Status:** Approved — spike-first (Phase 0 executes next; Zig build gated behind the go/no-go)
- **Author:** brian (with Claude Code)

---

## 1. Motivation

The slow big **dense** models are where speculative decoding should finally pay off on
Apple Silicon. mlx-serve's own DSpark investigation established the round economics:

> `speedup ≈ (1 + accepted_per_round) / round_cost_ratio`

On the fast Qwen3-8B, the target verify forward dominates (`round_cost_ratio ≈ 3.2` at
~1.6 accepted), so the drafter *loses* and the runtime gate correctly disables it. On a
slow dense target the fixed drafter overhead is dwarfed by the target forward — the
opposite regime. mlx-serve's production **Qwen3-32B-8bit baseline is 15.3 tok/s decode**
(temp 0.6, coding, `bench-matrix-2026-07-06.csv`), which is squarely in the favorable
regime.

Published **EAGLE-3** draft heads for Qwen3-32B already exist (no training required).
Getting a native path to load and run them is the concrete route to a measured
speculative-decode win on the production model — the roadmap row left open by the
DSpark case study.

## 2. Context — what already exists in the repo (~70% of the machinery)

EAGLE-3 is a sibling of the **DSpark** drafter already shipped through milestone 2. The
reusable machinery:

- **`src/dspark.zig`** — an EAGLE-3-*style* multi-layer-hidden-fusion drafter (flat
  external checkpoint, own KV cache, `--dspark <dir>` flag, graceful-disable on
  mismatch). Closest structural template for an EAGLE-3 loader.
- **Multi-layer raw-hidden tap** — `ForwardCtx.capture_layers` /
  `forwardWithCaptureLayers` (`src/transformer.zig`) captures the raw post-residual,
  pre-final-norm output of specific decoder layers. Only honored in
  `forwardStandardWith` (dense path) — fine, Qwen3-32B is dense.
- **Architecture-agnostic verify** — `pldAcceptWalk` (`src/generate.zig`) is shared
  verbatim by PLD / drafter / MTP / DSpark. The Leviathan accept/reject needs no
  EAGLE-specific knowledge.
- **Partial-accept re-forward elimination** — `KVCache.truncate` (commit `92f001e`,
  +24% decode) generalizes to any pure-causal dense trunk. Qwen3-32B qualifies.
- **Draft-vocab↔full-vocab remap analog** — `drafter.zig`'s `MaskedEmbedding`
  (`ordering_2d` + `mlx_take_axis` gather + `mlx_put_along_axis` scatter) is the same
  primitive an EAGLE-3 `d2t` table needs.

Genuinely **new** for EAGLE-3 (small surface): the 2×hidden fused draft layer and the
`d2t` additive remap.

## 3. Research findings (primary source)

### 3.1 Checkpoint — `RedHatAI/Qwen3-32B-speculator.eagle3` is the target

Chosen over AngelSlim (ships a `pytorch_model.bin` pickle **and** has a `head_dim: 80`
authoring bug) and Zjcxy (pickle). RedHat's head:

- Single-shard **`model.safetensors`** — mlx-serve's existing `loadWeights` reads it
  natively; "MLX weight conversion" nearly collapses to "point the loader at the dir."
- **Apache-2.0, public, ungated.** bf16, unquantized, ~3.12 GB.
- Nested `speculators` schema: `transformer_layer_config` (hidden 5120, heads 64/kv 8,
  **head_dim 128**, intermediate 25600, 1 layer, rope_theta 1e6, rms_eps 1e-6, vocab
  151936) + `speculators_config` (verifier `Qwen/Qwen3-32B`, `speculative_tokens: 3`).
- `draft_vocab_size: 32000`. **Metadata verifies** it ships `d2t` (int64×32000) and
  `t2d` (bool×151936).

**Gaps to close during the spike:** (a) exact serialized tensor key strings are
unverified — inspect the safetensors header directly; (b) `eagle_aux_hidden_state_layer_ids`
(which target layers feed the fusion) is **not** in config.json — assume the reference
`{2, N//2, N−3}` = `{2, 32, 61}` for N=64 and confirm empirically via acceptance.

### 3.2 Algorithm — from the SafeAILab reference (`cnets.py`, `modeling_qwen3_kv.py`)

- **Tap 3 target layers** at `{2, N//2, N−3}` — raw hidden states, mid-stack,
  pre-final-norm. Fixed formula, not a config field. (Pre-vs-post-layer capture point is
  inferred as post-layer output — confirm against reference.)
- **Fusion:** concat the 3 tapped hiddens → `[B,L,3·hidden]` → `fc: Linear(3·hidden →
  hidden, bias=False)`.
- **Token fusion:** the sampled token is embedded and **concatenated** with the fused
  feature → `2·hidden` wide. The single draft decoder layer's q/k/v projections take
  `2·hidden` input; otherwise it is a standard pre-norm Llama block. Then `norm` → draft
  `lm_head` (draft-vocab wide).
- **`d2t` (draft→target):** additive offset. `target_id = draft_id + d2t[draft_id]`
  applied to the draft argmax before feeding back / verifying. (Cross-confirmed by vLLM
  PR #25917.)
- **`t2d` (target→draft):** boolean mask over full vocab; only confirmed used at
  training/vocab-construction time — **likely not needed at inference.** Confirm.
- **Verify is untouched by the vocab reduction:** once draft ids are remapped to target
  space, the target runs its normal full-vocab forward and Leviathan accept/reject
  proceeds as usual.

### 3.3 The risk (why we spike before building)

- The **only** MLX EAGLE-3 implementation is an unmerged single-author prototype
  (mlx-lm Discussion #890, `kmsalah`) — includes a PyTorch→MLX **converter** and a
  **chain-mode forward** we can validate against. It measured **1.05× at 0.34 draft
  acceptance** on an M3 Ultra — but for a **Llama-3.1-8B** target (the *fast* regime
  where DSpark already loses), and 0.34 is suspiciously low vs. the reference's 1.8–2.2
  accept-length (possible port bug, indistinguishable from the post).
- **Tree attention** (the paper's headline 6.5× lever, depth 8) is effectively blocked
  on MLX — the prototype author hit the KVCache-lacks-flexible-position-ids wall. We go
  **chain-only** (topk=1, precedented by SGLang), which caps the realistic upside at the
  ~1.4–2.2× accept-length range.

**Net:** buildability is low-risk (clear reference, clean checkpoint, machinery exists).
The dominant risk is empirical — *does the ported head hit reference acceptance on MLX,
and does the 32B economics deliver a real win?* Both are cheap to measure and expensive
to assume.

## 4. Approach — phased, spike-gated

```
Phase 0  Python measurement spike (M5 box)  ──▶  🚧 GATE  ──▶  Phase 1+  eagle3.zig (TDD)
         throwaway; produces 2 numbers               go/no-go        gated on the gate
         + a trusted converter                       from real       being green
                                                     numbers
```

## 5. Phase 0 — spike design (executes next)

Runs on the M5/128 GB bench box (`llmbench@192.168.1.252`, `~/perf-work`).

- **Step A — reproduce the reference.** Run kmsalah's prototype as-published on
  Llama-3.1-8B; confirm we reproduce ~1.05× / ~0.34. Proves harness fidelity so a low
  32B number can be trusted as real, not our bug.
- **Step B — adapt to Qwen3-32B + the RedHat head.** Convert the RedHat safetensors via
  the prototype's `eagle_convert.py` (extended for the `speculators` schema + `d2t`),
  wire the tap at `{2, 32, 61}`, the 2×hidden fused layer, and the `d2t` remap. Chain
  mode, k=3.
- **Step C — measure** on coding + prose prompts, chain mode:
  1. **Per-draft acceptance** (diagnostic — is the port faithful? target ~0.5+).
  2. **Decode throughput vs. the 15.3 tok/s baseline** (the business gate).
- **Target-precision variable.** The head was trained on **bf16** target hiddens but
  production serves **8-bit**. Measure acceptance at **both** 8-bit (drives the gate) and
  fp16 (32B fp16 ≈ 64 GB, fits 128 GB → tells us whether target precision is a lever), so
  a low number isn't silently confounded by quantization.

**Deliverables:** the two numbers, a converter we trust, and a short written result
(feeds the case-study either way).

## 6. The gate

| Verdict | Condition | Action |
|---|---|---|
| **Green** | throughput **≥ 1.3× baseline** (≈ ≥ 20 tok/s) on ≥1 realistic workload, acceptance in reference range | Build the Zig path (Phase 1+) |
| **Amber** | healthy acceptance but throughput 1.0–1.3× | The DSpark economics story again — decide if a smaller win justifies a 5th spec path |
| **Red** | throughput ≤ 1.0×, or acceptance stuck low after a fidelity chase | Stop; document the negative result for the case study |

## 7. Phase 1+ — production `eagle3.zig` (gated; re-planned after the spike)

A 5th spec path, modeled on `dspark.zig`:

- Flat-checkpoint loader `--eagle3 <dir>` + `hasEagle3Checkpoint`, graceful-disable on
  arch/shape mismatch (DSpark convention, **not** the drafter's hard-fail).
- The 2×hidden fused draft forward; the 3-layer tap via `capture_layers`; own single-layer
  dense KV cache (MTP's committed-history template — the per-step chaining is closer to
  MTP than DSpark's one-shot infill).
- `d2t` remap via the `MaskedEmbedding` gather/scatter primitives.
- Verify via `pldAcceptWalk` + `KVCache.truncate` partial-accept fast path.
- Dispatch wired into **both** streaming (`pickStreamMode`) and non-streaming priority
  chains + `Generator.InitOptions` — per the CLAUDE.md dispatch-hole gotcha, or it
  silently falls back to regular decode while passing output-equality tests.
- Tensor constants (tap layer ids, k, gate thresholds) taken from the spike.

## 8. Test strategy

- **Spike (Phase 0):** reproduce the reference number first; acceptance vs. the reference
  accept-length; A/B target precision. This is measurement, not TDD.
- **Zig (Phase 1+, TDD per CLAUDE.md):** failing test first for each unit; **byte-equivalence**
  vs. `--no-eagle3` at temp 0 (first-N tokens); an **acceptance-floor** check (a
  structurally broken head engages, accepts ~0%, and gate-falls-back to regular decode —
  equivalence + engagement alone can't catch it, exactly like the MTP sidecar guard); a
  per-request **`[spec-stats] mode=eagle3` engagement count** on chat (stream +
  non-stream) + `/v1/messages` (output-equality can't see a silent dispatch-hole
  fallback). Class guard: any new dispatch site added to both modes.

## 9. Open questions / risks tracked into the spike

1. Exact tensor key strings (unverified) — inspect the RedHat safetensors header first.
2. `eagle_aux_hidden_state_layer_ids` absent from config — assume `{2, N//2, N−3}`,
   validate via acceptance.
3. Pre- vs post-layer capture point — confirm against `cnets.py`.
4. `t2d` inference-time role — confirmed only train-time; verify it's a no-op at serve.
5. Head trained on bf16 target hiddens vs. 8-bit production target — quantified by the
   precision A/B.
6. Chain-only caps upside at ~1.4–2.2×; tree is out of scope (MLX position-ids limitation).

## 10. References

- EAGLE-3 paper: arXiv 2503.01840.
- Reference impl: `SafeAILab/EAGLE` (`eagle/model/cnets.py`,
  `eagle/model/modeling_qwen3_kv.py`).
- Checkpoint: `RedHatAI/Qwen3-32B-speculator.eagle3` (HF). `d2t` semantics cross-confirmed
  by vLLM PR #25917; `speculators` schema per docs.vllm.ai + blog.vllm.ai 2025-12-13.
- MLX prototype (converter + chain forward): ml-explore/mlx-lm Discussion #890,
  branch `kmsalah/mlx:eagle3-speculative-decoding`.
- In-repo precedent: `src/dspark.zig`, `src/mtp.zig`, `src/drafter.zig`,
  `src/generate.zig` (`pldAcceptWalk`), `src/transformer.zig`
  (`capture_layers`/`KVCache.truncate`), `docs/dspark-port-spec.md`,
  `docs/dspark-acceptance-investigation.md`.
