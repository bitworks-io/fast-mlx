# DSpark → Zig Port Specification

Target deliverable: `src/dspark.zig`, a new sidecar alongside `src/drafter.zig`
and `src/mtp.zig`, implementing DeepSeek-AI's DSpark speculative-decoding
drafter for Qwen3 targets in mlx-serve, using the mlx-c FFI already bound in
`src/mlx.zig`.

Status: COMPLETE. All 7 requested items are answered from primary source
(real Python source + real checkpoint metadata), not from memory or
reconstruction. See "Source provenance" for exactly what was fetched and how.

---

## 0. Source provenance

The task described `~/dspark-ref` as already cloned. **It was not present**
at task start (confirmed via `ls`, `find ~ -iname "*dspark*"` — no hits
anywhere on the machine). Both sources were fetched fresh over the network
during this session:

- **Checkpoint metadata**: `deepseek-ai/dspark_qwen3_8b_block7` on
  HuggingFace (public, not gated). `hf` CLI was not installed, so:
  - `config.json` fetched via `curl` to `~/dspark-ref/ckpt_q8b/config.json`
    (full contents reproduced in section 6 below).
  - The checkpoint ships a single unsharded `model.safetensors` (no
    `model.safetensors.index.json` — that fetch 404'd, confirmed via the HF
    API's `siblings` listing: `.gitattributes`, `config.json`,
    `model.safetensors` only). Its safetensors header (tensor names +
    dtypes + shapes) was fetched WITHOUT downloading weight bytes, via an
    HTTP Range request for the 8-byte header-length prefix + the header
    JSON itself (`curl -r 0-7` then `curl -r 8-<8+len-1>`), saved to
    `~/dspark-ref/ckpt_q8b/safetensors_header.json`. Confirmed via the HF
    API (`api/models/...`): 64 tensors, all BF16, `{"BF16": 2371081729}`
    parameters (~2.37B), `usedStorage` ~4.74 GB, `__metadata__: {"format":
    "pt"}`.
  - No README on the checkpoint repo (404).
- **Reference code**: the checkpoint's linked HF Space
  (`Mike0021/deepspec-decoding-lab`) is a Gradio *simulator* built on
  published Table-1 accept-length numbers — its `requirements.txt` is just
  `gradio==6.10.0` + `plotly==5.24.1` (no torch/transformers), and its
  `README.md` names the real sources:
  - DeepSpec repository: `https://github.com/deepseek-ai/DeepSpec`
  - DSpark paper: `DeepSpec/DSpark_paper.pdf` (in-repo, not separately
    fetched — the code is authoritative and sufficient for the port)
  This confirmed the actual code repo, which was `git clone --depth 1`'d to
  `~/dspark-ref/DeepSpec/`. All citations below (`file:line`) refer to this
  clone.
- Every claim in this document that isn't explicitly marked "inferred" or
  "unconfirmed" is backed by a full read of the cited source file(s), not a
  search-result summary or paraphrase from training data. DSpark postdates
  this assistant's knowledge cutoff, so nothing here was recalled from
  memory.

Files read in full (all under `~/dspark-ref/DeepSpec/`):
- `deepspec/modeling/dspark/qwen3/config.py` (61 lines)
- `deepspec/modeling/dspark/qwen3/modeling.py` (532 lines)
- `deepspec/modeling/dspark/common.py` (309 lines)
- `deepspec/modeling/dspark/markov_head.py` (319 lines)
- `deepspec/eval/dspark/draft_ops.py` (154 lines)
- `deepspec/eval/dspark/evaluator.py` (224 lines)
- `deepspec/eval/base_evaluator.py` (728 lines)
- `deepspec/eval/dspark/confidence_head.py` (skimmed in full — 605 lines;
  confirmed to be ECE/AUROC/Brier calibration + reliability-diagram
  plumbing only, no additional model math beyond what's in
  `draft_ops.py`/`common.py`)
- `deepspec/utils/sampling.py` (53 lines)
- `deepspec/trainer/base_trainer.py` (partial — the
  `initialize_embeddings_and_head` call site, lines ~245-278)
- `config/dspark/dspark_qwen3_8b.py` (68 lines, training config — confirms
  target = `Qwen/Qwen3-8B`)
- Structural diff of `deepspec/modeling/dspark/gemma4/modeling.py` against
  the qwen3 variant (confirms the same abstraction — differences are
  attention-layer specifics only: gemma4's `_repeat_kv`/sliding-window
  handling. Qwen3 variant is authoritative for this port.)
- `deepspec/modeling/dspark/loss.py` — skimmed (training-loss-only, no
  inference-relevant content; not needed for a serving-side port).

Existing mlx-serve sidecars read for pattern precedent:
- `src/mtp.zig` (644 lines, read via a scout sub-agent + direct citations
  confirmed against `src/generate.zig`'s `Generator.nextMtp` at
  `generate.zig:2221-2583` approx., read directly in this session)
- `src/drafter.zig` (skimmed for the embed-sharing precedent, `drafter.zig`
  lines ~251-294 and ~181-212)
- `src/mlx.zig` (grepped for available FFI primitives — confirms
  `mlx_fast_rms_norm`, `mlx_fast_rope`, `mlx_fast_scaled_dot_product_attention`,
  `mlx_take_axis`, `mlx_matmul`/`mlx_quantized_matmul`, `mlx_sigmoid`,
  `mlx_softmax_axis` are all already bound and reusable)
- `src/transformer.zig` (grepped for `mask_mode` usage patterns —
  `"causal"`, `"array"`, and empty-string/no-mask conventions)

---

## 1. Drafter network architecture

**5-layer, non-autoregressive, cross-attention-fused backbone.** Not an
autoregressive stack — one forward pass processes an entire draft block.

### Config (from `~/dspark-ref/ckpt_q8b/config.json`, reproduced in full)

```json
{
  "architectures": ["Qwen3DSparkModel"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "block_size": 7,
  "bos_token_id": 151643,
  "confidence_head_with_markov": true,
  "dtype": "bfloat16",
  "enable_confidence_head": true,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 12288,
  "layer_types": ["full_attention", "full_attention", "full_attention", "full_attention", "full_attention"],
  "markov_head_type": "vanilla",
  "markov_rank": 256,
  "mask_token_id": 151669,
  "max_position_embeddings": 40960,
  "max_window_layers": 36,
  "model_type": "qwen3",
  "num_anchors": 512,
  "num_attention_heads": 32,
  "num_hidden_layers": 5,
  "num_key_value_heads": 8,
  "num_target_layers": 36,
  "pad_token_id": null,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
  "sliding_window": null,
  "target_layer_ids": [1, 9, 17, 25, 33],
  "tie_word_embeddings": false,
  "transformers_version": "5.10.2",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}
```

`deepspec/modeling/dspark/qwen3/config.py:9-56` (`build_draft_config`)
confirms the draft config is a **deep copy of the target's own HF config**,
then overridden with draft-specific fields. So every field NOT listed above
as draft-specific (hidden_size, vocab_size, rope_theta, rms_norm_eps,
attention_bias, hidden_act, head_dim) is inherited verbatim from
`Qwen/Qwen3-8B`'s config — this checkpoint's config.json happens to show
identical values because Qwen3-8B and the draft share hidden_size=4096,
which is a coincidence of this particular target/draft pairing, not a port
requirement (the draft's hidden_size always equals the target's).

### Exact layer count and hidden dim

**5 decoder layers** (`num_hidden_layers: 5`, all `layer_types:
"full_attention"` — no sliding-window layers). Hidden dim **4096**, same as
Qwen3-8B (32 attention heads × head_dim 128 = 4096; 8 KV heads × 128 = 1024,
i.e. standard 4:1 GQA ratio). MLP intermediate 12288 (SwiGLU, `hidden_act:
"silu"`). RMS norm eps 1e-6. No biases anywhere (`attention_bias: false`,
and the MLP/attn linears have no bias tensors in the checkpoint either).

### Target hidden-state layers tapped (EAGLE-3-style multi-layer tap)

**`target_layer_ids: [1, 9, 17, 25, 33]`** — 5 taps into the 36-layer Qwen3-8B
target (`num_target_layers: 36`), evenly spaced every 8 layers. This is
confirmed to be the multi-layer feature-fusion EAGLE-3 popularized.

Exact indexing semantics, `deepspec/modeling/dspark/common.py:52-56`
(`extract_context_feature`):

```python
def extract_context_feature(hidden_states, layer_ids):
    return torch.cat(
        [hidden_states[0 if layer_id == -1 else layer_id + 1] for layer_id in layer_ids],
        dim=-1,
    )
```

HF's `output_hidden_states=True` tuple has index 0 = embedding output,
index `i+1` = output of decoder layer `i` (0-indexed). So `target_layer_ids:
[1, 9, 17, 25, 33]` reads tuple indices `[2, 10, 18, 26, 34]` — the raw
(pre-final-norm) OUTPUTS of Qwen3-8B decoder layers 1, 9, 17, 25, 33
(0-indexed). `-1` in a `target_layer_ids` list would mean "the embedding
output" (tuple index 0) but is not used by this checkpoint.

**Hard invariant** (`deepspec/eval/base_evaluator.py:100-112`,
`assert_no_final_target_layer`): `target_layer_ids` must NEVER include the
target's LAST decoder layer (35 for a 36-layer target), because HF's
`output_hidden_states` tuple stores the FINAL NORMALIZED hidden state at
that position while every earlier entry is a raw (pre-norm) decoder-layer
output. Concatenating a normalized tensor with 4 raw ones would be a
silent train/inference mismatch. **A Zig port capturing target hidden
states via `forwardCaptureHidden`-style hooks must capture the RAW
per-layer output (before any final `model.norm`), matching this
convention** — mlx-serve's own `forwardCaptureHidden` already captures
post-final-norm hidden at the LAST position for drafter/PLD seeding
(per project CLAUDE.md), so this is a NEW capture point: raw per-layer
outputs at 5 specific layer indices, not the final normed hidden.

Concatenation order matches `target_layer_ids` list order:
`[batch, ctx_len, 5*4096=20480]` — matches `fc.weight [4096, 20480]` exactly
(see tensor inventory, section 6).

### Input construction

Two inputs feed the backbone (`_forward_backbone`,
`qwen3/modeling.py:361-386`):

1. **`noise_embedding`** — token embeddings of the draft block's input ids
   (`[anchor_token, MASK, MASK, MASK, MASK, MASK, MASK]` — see section 2),
   looked up via the draft's OWN `embed_tokens` table (see section 7 for why
   this is a full separate copy, byte-identical to the target's).
2. **`target_hidden_states`** — the RAW concat of the 5 tapped layers
   (`[batch, ctx_len, 20480]`), immediately projected down and normed
   INSIDE `_forward_backbone` before use as attention K/V source:
   ```python
   # qwen3/modeling.py:372-373
   target_hidden_states = self.hidden_norm(self.fc(target_hidden_states))
   ```
   `fc` is a bias-free `Linear(20480, 4096)`; `hidden_norm` is a
   `Qwen3RMSNorm(4096)`. This projection+norm happens FRESH every round
   from the raw `target_hidden_states` argument — nothing about it is
   cached across rounds (only the raw `ctx_len`-length hidden-state
   tensor persists between rounds, sliced down after each verify — see
   section 5).

There is NO separate positional embedding added to `noise_embedding` beyond
what RoPE supplies inside attention — the "noise" name refers to the masked
(non-anchor) positions carrying only the `mask_token_id` embedding plus
positional signal, not literal noise vectors.

### The `Qwen3DSparkAttention` block — verbatim Python (this IS the module to port)

`deepspec/modeling/dspark/qwen3/modeling.py:43-151`:

```python
class Qwen3DSparkAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            self.num_attention_heads // self.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        )

    def forward(self, hidden_states, target_hidden_states, position_embeddings, attention_mask, past_key_values=None, cache_position=None, **kwargs):
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden_states.shape[1]
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_attention_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden_states)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden_states)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        # ... GQA repeat_interleave for flex_attention path (training only) ...
        attn_output, attn_weights = attn_fn(self, q, k, v, attention_mask, scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)
        attn_output = attn_output.reshape(bsz, q_len, self.num_attention_heads * self.head_dim)
        return self.o_proj(attn_output), attn_weights
```

**Load-bearing details for the Zig port:**

- **Q is projected from `hidden_states` (the draft/noise stream) ONLY.**
  Q is NEVER computed from `target_hidden_states`.
- **K and V are each computed TWICE per layer**: once from
  `target_hidden_states` (the fused, projected+normed target context —
  `k_ctx`/`v_ctx`) and once from `hidden_states` (the draft block itself —
  `k_noise`/`v_noise`), then **concatenated along the sequence axis**
  (`dim=1`) BEFORE the RoPE/norm step: `k = cat([k_ctx, k_noise])`. So the
  attention keys/values span `ctx_len + q_len` positions: the (growing)
  target context first, then the current block's own tokens.
- **QK-norm is applied AFTER the K-concat**, i.e. `k_norm` runs over the
  FULL concatenated `[ctx_len+q_len, head_dim]` tensor (same RMSNorm
  weights `[head_dim]=128` for all positions — no per-source distinction).
- **RoPE is applied to the concatenated K** using `position_embeddings`
  built from `rotary_emb(noise_embedding, draft_position_ids)`
  (`qwen3/modeling.py:374`), where `draft_position_ids` comes from
  `forward_dspark_draft_block` (`draft_ops.py:32-34`):
  ```python
  draft_position_ids = position_ids[:, past_key_values_draft.get_seq_length() : start + block_size]
  ```
  **This resolves cleanly — `cos`/`sin` DO span exactly `ctx_len + q_len`
  positions, matching K's length, with no mismatch.** The key fact (easy to
  get wrong, and initially mis-traced during this investigation): the
  draft's own KV cache PERSISTS across rounds — `get_seq_length()` at round
  entry equals the PREVIOUS round's `start`, not the current one (the
  `past_key_values_draft.crop(start)` at the end of `forward_dspark_draft_block`,
  `draft_ops.py:44`, crops using THAT round's `start` value, which becomes
  next round's floor). So the slice lower bound is `start_prev`, the upper
  bound is `start_curr + block_size`, and the length is
  `(start_curr + block_size) − start_prev = ctx_len_curr + block_size`
  (since `ctx_len_curr` — the current round's `target_hidden_states.shape[1]`,
  set by `_update`'s `[:, :accepted_prev+1, :]` slice, `evaluator.py:143-147`
  — is exactly `start_curr − start_prev`, the number of tokens committed
  last round). Verified by direct arithmetic simulation of two consecutive
  rounds (prompt len L=10, round-1 accept=4): round 1 gives
  `draft_position_ids` length 17 = ctx_len(10)+block_size(7); round 2 gives
  length 12 = ctx_len(5)+block_size(7) — both match K's length exactly in
  every round. **So the target context K/V IS rotated using its own true
  absolute sequence positions** (the low end of the `position_ids` slice),
  not a reset/local offset — consistent with `position_ids` being one
  single `torch.arange` built ONCE for the whole generation
  (`base_evaluator.py:342`) and never re-based per round. No ambiguity
  remains here; see the corrected KV-cache-persistence discussion in
  section 2 and section 5 for the full implication (the draft's own cache
  is NOT ephemeral/scratch — it accumulates the full committed history,
  which is WHY only `accepted+1)`-sized fresh `target_hidden_states` need
  to be fed to `_update` each round: the older committed positions'
  projected-context K/V already live in the persisted draft cache from
  earlier rounds).
- **Attention itself**: standard scaled dot-product,
  `scaling = head_dim**-0.5`, GQA repeat handled by the attention backend
  (SDPA repeats KV heads internally when `num_key_value_groups>1`, EXCEPT
  the manual `repeat_interleave` shown is gated on
  `_attn_implementation == "flex_attention"`, i.e. training only — the
  eval path uses `"sdpa"` per `evaluator.py:33`,
  `EVAL_ATTN_IMPLEMENTATION = "sdpa"`, which handles GQA head-repeat
  internally without the manual code path).
- **Output**: `o_proj(attn_output)` — standard.

### Decoder layer (verbatim structure, `modeling.py:154-198`)

Standard pre-norm residual block:
```python
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)
hidden_states = self.self_attn(hidden_states=hidden_states, target_hidden_states=target_hidden_states, ...)[0]
hidden_states = residual + hidden_states
residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)  # Qwen3MLP: down(silu(gate(x)) * up(x))
return residual + hidden_states
```
Note `target_hidden_states` is passed to EVERY layer unchanged (not
re-projected per-layer — the `fc`+`hidden_norm` projection happens ONCE in
`_forward_backbone` before the layer loop, and the SAME projected
`target_hidden_states` tensor is reused by all 5 layers' K/V).

### Final norm + output head

After all 5 layers: `hidden_states = self.norm(hidden_states)` (final
`Qwen3RMSNorm(4096)`) — this is `output_hidden` in
`_forward_backbone`/`forward_dspark_draft_block`. Logits are computed by a
SEPARATE call to `compute_logits` (`modeling.py:289-290`):
```python
def compute_logits(self, hidden_states):
    return self.lm_head(hidden_states)
```
`lm_head` is a bias-free `Linear(4096, 151936)` — a full-vocab dense
projection (unlike mlx-serve's Gemma-4 assistant drafter, which uses a
sparse `MaskedEmbedding` LM head over ~4096 of 262144 tokens; DSpark's
draft LM head is a full dense matmul over the full 151936-token vocab,
matching the target's own vocab size exactly since it's byte-copied from
the target — see section 7).

---

## 2. Block prediction — parallel, not autoregressive

**One backbone forward per round proposes ALL `block_size` (7) draft
tokens in parallel** via mask-token infill, NOT 7 sequential AR steps. This
is the single most important architectural fact distinguishing DSpark from
mlx-serve's existing PLD/drafter/MTP sidecars (all of which are
autoregressive chains).

### Construction (`deepspec/eval/dspark/evaluator.py:99-132`, `_propose`)

```python
draft_input_ids = torch.full((1, block_size), mask_token_id, dtype=torch.long)
draft_input_ids[:, 0] = output_ids[:, start]  # the current accepted/anchor token
block_hidden = forward_dspark_draft_block(model, draft_input_ids=draft_input_ids, ...)
```

So the draft block's INPUT is `[t1, MASK, MASK, MASK, MASK, MASK, MASK]`
(length 7 = block_size), where `t1` is the currently-accepted token (the
"anchor" — same role as PLD/drafter/MTP's `t1`) and the remaining 6
positions are the literal `mask_token_id` (151669) embedding, carrying no
token information — only their sequence position (via RoPE) and their
non-causal attention to the target context.

### The single forward (`deepspec/eval/dspark/draft_ops.py:22-45`)

```python
def forward_dspark_draft_block(model, *, draft_input_ids, position_ids, past_key_values_draft, target_hidden_states, start, block_size):
    draft_position_ids = position_ids[:, past_key_values_draft.get_seq_length() : start + block_size]
    block_hidden = model._forward_backbone(
        target_hidden_states=target_hidden_states,
        noise_embedding=model.embed_tokens(draft_input_ids),
        position_ids=draft_position_ids,
        attention_mask=None,
        past_key_values=past_key_values_draft,
        use_cache=True,
        is_causal=False,
    )
    past_key_values_draft.crop(start)
    return block_hidden
```

This is ONE call through all 5 backbone layers — no per-token loop. The
output `block_hidden` has shape `[1, block_size, 4096]`: one hidden vector
per draft position, computed simultaneously with full bidirectional
attention among the 7 positions (see section 1's attention-mask analysis)
and cross-attention into the (growing) target context K/V.

**`attention_mask=None` + `is_causal=False`** means the SDPA call applies
**NO masking at all** — every draft position (including position 0, the
anchor) attends to every other draft position (past AND future within the
block) plus every position in the fused target context. This is a
non-causal, BERT/MaskGIT-style parallel decoder, fundamentally unlike
drafter.zig's causal cross-attention chain or mtp.zig's autoregressive
draft loop.

### From hidden states to sampled tokens

`draft_ops.py:96-153` (`build_dspark_proposal`):
```python
proposal_hidden_states = block_hidden[:, :block_size, :]
base_draft_logits = model.compute_logits(proposal_hidden_states)   # [1, 7, 151936]
sampled_tokens, draft_logits = model.sample_draft_tokens(
    base_draft_logits, first_prev_token_ids=draft_input_ids[:, 0], temperature=..., hidden_states=proposal_hidden_states,
)
```
`model.sample_draft_tokens` (`modeling.py:309-333`) dispatches to the
Markov head's `sample_block_tokens` when a Markov head is present (this
checkpoint's case — see section 3). **This is where the seemingly-parallel
block prediction gets an AUTOREGRESSIVE correction layer bolted on**: the
BASE logits for all 7 positions come from ONE parallel forward, but the
Markov head then walks positions 0..6 SEQUENTIALLY, feeding each step's
sampled token as `prev_token_ids` into the next step's Markov bias (see
section 3) — this sequential walk operates purely on the already-computed
per-position base logits + a rank-256 embedding lookup (cheap), NOT another
transformer forward. So block prediction is: **backbone = 1 parallel
forward; Markov correction + sampling = 7 cheap sequential steps over
already-computed logits.**

### Confidence-gated adaptive length

After the block's tokens are sampled, an OPTIONAL confidence head (present
in this checkpoint — see section 4) can TRUNCATE the proposal to fewer than
`block_size` tokens before it's ever sent to the target for verification —
saving verify-forward cost on rounds where the drafter is unconfident past
some prefix. This happens entirely draft-side, before section 5's
verify/accept step even runs.

---

## 3. Markov head — rank-256 previous-token correction

**Exact math (VanillaMarkov — this checkpoint's `markov_head_type`)**,
`deepspec/modeling/dspark/markov_head.py:8-32`:

```python
class VanillaMarkov(nn.Module):
    def __init__(self, *, vocab_size, markov_rank):
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)   # [151936, 256]
        self.markov_w2 = nn.Linear(markov_rank, vocab_size, bias=False)  # [151936, 256] (out,in) — no bias

    def get_prev_embeddings(self, token_ids):
        return self.markov_w1(token_ids.long())         # lookup: [*, 256]

    def project_bias(self, latent_states):
        return self.markov_w2(latent_states)             # [*, 256] -> [*, 151936]

    def compute_step_bias(self, token_ids, hidden_states):
        del hidden_states   # UNUSED for vanilla — no hidden-state dependency at all
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_step_logits(self, logits, *, token_ids, hidden_states):
        return logits + self.compute_step_bias(token_ids, hidden_states)
```

**In one line: `bias = markov_w2 @ markov_w1[prev_token_id]`, and
`corrected_logits = base_logits + bias`.** This is a rank-256 factored
"bigram correction" — purely a function of the single previous token id
(NOT the current position's hidden state, NOT any earlier tokens in the
block beyond the immediately-preceding one). The two safetensors tensors
`markov_head.markov_w1.weight [151936, 256]` (embedding table) and
`markov_head.markov_w2.weight [151936, 256]` (linear weight, PyTorch
`(out_features, in_features)` layout — this IS `markov_w2`'s weight, applied
as `latent @ markov_w2.weight.T` to go from 256→151936) implement exactly
this.

**Confirmed as the active variant**: the checkpoint's config has
`markov_head_type: "vanilla"`. Two other variants exist in the code
(`GatedMarkovHead` — adds a sigmoid gate over the bias, needs a
`gate_proj: Linear(hidden_size+markov_rank, markov_rank)`; `RNNHead` — a
GRU-like recurrent state across block positions, needs a
`joint_proj: Linear(2*markov_rank+hidden_size, 3*markov_rank)`) but **this
checkpoint's safetensors header has NEITHER `gate_proj` NOR `joint_proj`
tensors** — confirming only `VanillaMarkov`'s two tensors are present and
the Zig port only needs to implement the vanilla math for THIS checkpoint.
(A future checkpoint could ship `markov_head_type: "gated"` or `"rnn"` —
the Zig loader should assert on `markov_head_type` and fail loudly rather
than silently mis-load if it's not `"vanilla"`.)

### How the sequential block walk combines with parallel base logits

`markov_head.py:55-90` (`sample_block_tokens`, called by
`Qwen3DSparkModel.sample_draft_tokens` when a Markov head exists):

```python
def sample_block_tokens(self, base_logits, *, first_prev_token_ids, hidden_states, temperature=0.0):
    # base_logits: [1, block_size, vocab] — from the ONE parallel backbone forward
    sampled_tokens, corrected_logits = [], []
    prev_token_ids = first_prev_token_ids.long()   # the anchor token t1
    for step_idx in range(block_size):             # 7 iterations, CHEAP (no transformer fwd)
        step_logits = self.apply_step_logits(base_logits[:, step_idx, :], token_ids=prev_token_ids, hidden_states=None)
        next_token_ids = sample_tokens(step_logits.unsqueeze(1), temperature=temperature).squeeze(1)
        sampled_tokens.append(next_token_ids)
        prev_token_ids = next_token_ids   # THIS step's sampled token becomes NEXT step's Markov input
    return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)
```

So position `k`'s Markov bias depends on position `k-1`'s SAMPLED token
(which itself already includes ITS OWN Markov correction) — a genuine
sequential dependency chain, but over cheap [256]-dim lookups + a
[256→151936] matmul, not over transformer layers. **`draft_logits`
returned here (the corrected, post-Markov logits) are what get used for
`draft_probs` in the verify step (section 5) — the target model's verify
step must be compared against these CORRECTED probabilities, not the raw
backbone logits.**

### Where else the Markov head is applied

- **Confidence-head feature construction** (section 4): when
  `confidence_head_with_markov=true` (this checkpoint), the confidence
  head's input concatenates the backbone hidden state with
  `markov_w1[prev_token_id]` (the SAME 256-dim embedding, reused as a
  feature — NOT re-run through `markov_w2`).
- `sample_draft_token_step` (`modeling.py:335-359`) is a single-position
  variant of the same math, used elsewhere (not on the eval hot path
  described above, which uses the block variant) — same
  `apply_step_logits` call.

---

## 4. Confidence head — architecture + adaptive block length

**Architecture**: a single linear layer, `AcceptRatePredictor`
(`deepspec/modeling/dspark/common.py:43-49`):
```python
class AcceptRatePredictor(nn.Module):
    def __init__(self, input_dim):
        self.proj = nn.Linear(int(input_dim), 1)   # [1, input_dim] weight + [1] bias
    def forward(self, features):
        return self.proj(features).squeeze(-1)
```

**Input dim** (`qwen3/modeling.py:262-267`): `hidden_size` (4096) **PLUS**
`markov_rank` (256) when `confidence_head_with_markov=true` (this
checkpoint's setting) → **4352**, matching `confidence_head.proj.weight
[1, 4352]` exactly. The extra 256 dims come from concatenating the
PREVIOUS token's Markov embedding (`markov_w1[prev_token_id]`, the SAME
lookup table used for logit correction, section 3) onto the backbone
hidden state — NOT re-deriving anything from `markov_w2`.

Feature construction (`draft_ops.py:57-79`, `_predict_confidence_logits`):
```python
prev_token_ids = torch.cat([draft_input_ids[:, :1], sampled_tokens[:, :-1]], dim=1)  # shift-by-1: anchor, then each position's own sampled predecessor
confidence_pred = model.predict_confidence_step(proposal_hidden_states, prev_token_ids=prev_token_ids)
```
And `predict_confidence_step` (`qwen3/modeling.py:292-307`):
```python
def predict_confidence_step(self, hidden_states, prev_token_ids=None):
    if self.confidence_head_with_markov:
        prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids)
        features = torch.cat([hidden_states, prev_embeddings], dim=-1)   # [*, 4096+256]
        return self.confidence_head(features).float()
    return self.confidence_head(hidden_states).float()
```

So for each of the 7 draft positions, the confidence-head feature is
`concat(backbone_hidden[pos], markov_w1[prev_token_id_at_pos])`, where
`prev_token_id_at_pos` is the SAMPLED token immediately preceding this
position in the block (the anchor for position 0, else the previous
position's own Markov-corrected sample). Output is a scalar logit per
position; `sigmoid(logit)` is trained (via the cumulative-product target
described in `confidence_head.py:363-374`) to estimate **P(this position
AND everything before it in the block gets accepted by the target)**.

### Adaptive block length (`draft_ops.py:82-93`, `_confident_prefix_length`)

```python
def _confident_prefix_length(confidence_logits, *, block_size, threshold):
    if threshold <= 0.0:
        return int(block_size)              # threshold disabled -> always propose full block
    below_threshold = confidence_logits.sigmoid() < threshold
    if not bool(below_threshold[0].any().item()):
        return int(block_size)              # every position confident enough -> full block
    return int(torch.nonzero(below_threshold[0], as_tuple=False)[0].item())  # truncate at FIRST low-confidence position
```

This is the mechanism: find the first position (0-indexed within the
block) whose sigmoid-confidence drops below `confidence_threshold`
(a runtime/request parameter, NOT in the checkpoint config — see section 9
for where this default lives), and propose ONLY that many draft tokens
(0..that position exclusive) to the target for verification, discarding
the rest of the block's (already-computed, essentially free) tokens. A
proposal can legitimately shrink to **0 draft tokens** (`draft_ops.py:133-134`,
`_empty_dspark_proposal` — the round degenerates to sampling a single
fresh token from the target with no speculative gain, still correct, just
no speedup that round).

**Interaction with mlx-serve's runtime acceptance gate**: this is a
DIFFERENT mechanism from mlx-serve's existing `RUNTIME_GATE_MIN_PER_DRAFT_RATE`
sticky disable (per project CLAUDE.md) — DSpark's confidence gate operates
WITHIN a single round (shrinking THIS round's proposal length), while
mlx-serve's runtime gate operates ACROSS rounds (disabling spec-decode
entirely after sustained low acceptance). Both could coexist: DSpark's
gate as the per-round adaptive-length mechanism, mlx-serve's existing
sticky gate as the outer safety net if DSpark's own gate isn't tuned well
for a given prompt distribution.

---

## 5. Verify / accept — Leviathan-style stochastic rejection sampling

**This is architecture-agnostic in the reference repo** — the same
`verify_draft_tokens` + `generate_decoding_sample` functions in
`deepspec/eval/base_evaluator.py` serve DSpark, EAGLE-3, and DFlash
evaluators alike; only `_propose`/`_init_context`/`_update` differ per
architecture (confirmed: `Gemma4DSparkEvaluator(Qwen3DSparkEvaluator)` in
`evaluator.py:224-225` overrides NOTHING but `draft_model_cls` — the entire
propose/verify/update flow is shared even across target architectures).

### Verify step (`base_evaluator.py:186-304`, `verify_draft_tokens`)

```python
verify_length = draft_token_count + 1        # +1 for the anchor token itself
target_output = target_model(input_ids=proposal.verify_input_ids, position_ids=..., past_key_values=..., use_cache=True, output_hidden_states=True)
target_probs = logits_to_probs(target_output.logits, temperature)   # softmax or one-hot-argmax if temp<1e-5

# Leviathan rejection sampling, per draft position:
proposed_tokens = proposal.verify_input_ids[:, 1:]
selected_target_probs = gather_token_probs(target_probs[:, :-1, :], proposed_tokens)
selected_draft_probs = gather_token_probs(proposal.draft_probs, proposed_tokens).clamp_min(1e-8)
accept_prob = torch.clamp(selected_target_probs / selected_draft_probs, max=1.0)
accept_mask = (torch.rand_like(accept_prob) < accept_prob).to(torch.int64)
accept_prefix_mask = accept_mask.cumprod(dim=1)     # stops at first rejection
accepted_draft_tokens = int(accept_prefix_mask.sum(dim=1)[0].item())

# Residual/bonus sampling for the position right after the accepted prefix:
if accepted_draft_tokens < draft_token_count:
    next_token = sample_residual(target_probs[:, accepted_draft_tokens, :], proposal.draft_probs[:, accepted_draft_tokens, :])
else:
    next_token = sample_from_probs(target_probs[:, -1:, :]).squeeze(1)   # full accept -> bonus token from the LAST verify-logits row
```

`sample_residual` (`deepspec/utils/sampling.py:34-44`):
```python
def sample_residual(target_probs, draft_probs):
    residual = torch.clamp(target_probs - draft_probs, min=0.0)
    residual_mass = residual.sum(dim=-1, keepdim=True)
    if torch.any(residual_mass <= 1e-8):   # degenerate case: draft_probs already covers all of target_probs
        residual = torch.where(residual_mass <= 1e-8, target_probs, residual)
        residual_mass = residual.sum(dim=-1, keepdim=True)
    residual = residual / residual_mass.clamp_min(1e-8)
    return sample_from_probs(residual.unsqueeze(1)).squeeze(1)
```

**This is the textbook Leviathan et al. speculative sampling algorithm** —
`accept_prob = min(1, p_target/p_draft)`, residual = `max(p_target -
p_draft, 0)` renormalized. **Directly matches mlx-serve's existing verify
invariant** (per project CLAUDE.md: "stochastic verify: draft as one-hot;
`accept_prob = min(1, target_p[draft[i]])`, residual `max(target_p −
one_hot, 0)` renormalized — preserves marginal per Leviathan" — same
formula family, generalized here from PLD's implicit one-hot draft
distribution to DSpark's REAL draft probability distribution
`proposal.draft_probs`, which is the actual softmax the drafter sampled
from — a strictly more general case of the same math, not a different
algorithm).

### Mapping onto mlx-serve's existing verify contract

Per project CLAUDE.md ("Speculative decoding (PLD + drafter) — overview"):
> `cache.step = prompt_len + tokens_emitted`, t1 NOT in cache on entry, no
> pending state. Verify input is `[t1, draft[0..m-1]]` length `1+m`; full
> accept samples `new_t1` from `verify_logits[m]` (bonus prediction);
> partial accept rolls back via `KVCache.snapshot/restore` ... Pending
> correction sampled from *original* `verify_logits[accepted]` (NOT
> re-forward); index is `accepted` not `accepted-1`.

DSpark's `verify_draft_tokens` maps onto this EXACTLY:
- `proposal.verify_input_ids = [t1, draft[0..draft_token_count-1]]`,
  length `1+draft_token_count` — identical shape/semantics to mlx-serve's
  `[t1, draft[0..m-1]]`.
- `accepted_draft_tokens` ≡ mlx-serve's `accepted`.
- Full accept (`accepted_draft_tokens == draft_token_count`): bonus token
  sampled from `target_probs[:, -1:, :]` — the LAST row of verify logits,
  i.e. row index `draft_token_count` (0-indexed, since `verify_length =
  draft_token_count+1` rows exist) — this is exactly mlx-serve's
  `verify_logits[m]` convention.
- Partial accept: residual sampled from `target_probs[:,
  accepted_draft_tokens, :]` — row index `accepted_draft_tokens`, i.e.
  `verify_logits[accepted]`, EXACTLY matching mlx-serve's documented
  "index is `accepted` not `accepted-1`" invariant. **No off-by-one
  divergence between the two codebases on this point** — good, this is
  the single detail the project's own docs flag as most likely to
  silently corrupt output, and DSpark's reference implementation agrees
  with mlx-serve's existing convention.
- **KV rollback**: DSpark's evaluator doesn't need an explicit
  snapshot/restore for the TARGET's KV cache the way mlx-serve does,
  because it re-forwards the target model FRESH each verify call with
  `past_key_values=past_key_values_target` (a `DynamicCache` that only
  ever grows — HF's cache never speculatively writes ahead) and then
  crops it explicitly (`base_evaluator.py:418,425`,
  `past_key_values_target.crop(start)`, where `start` only ever advances
  by `accepted_draft_tokens+1` — i.e. the cache is TRIMMED to exactly what
  was ACCEPTED, matching mlx-serve's own snapshot/restore semantics but
  implemented via HF's `Cache.crop` API instead of mlx-serve's
  `KVCache.snapshot/restore`. **A Zig port uses mlx-serve's own existing
  `KVCache.snapshot/restore` (as PLD/drafter/MTP already do) — this is a
  pure implementation-detail difference (crop-in-place vs.
  snapshot-then-restore), not an algorithmic one; the OUTCOME (only
  accepted+bonus tokens persist in cache) is identical.**
- **The DRAFT's own KV cache** (`past_key_values_draft`) is a SINGLE
  `DynamicCache()` object created once in `_init_context`
  (`evaluator.py:92`) and threaded by reference through every subsequent
  round — it is **NOT** reset or rebuilt between rounds. `crop(start)`
  (`draft_ops.py:44`) removes only THIS round's `block_size` speculative
  (noise/mask) positions, leaving every previously-COMMITTED position's
  K/V resident (see the corrected trace in section 1's RoPE discussion:
  round-entry `get_seq_length()` equals the PREVIOUS round's `start`, and
  the cache therefore accumulates the full committed history across
  rounds, growing by `accepted+1` positions per round — same growth rate
  as the target's own cache). **This means the draft cache DOES need
  persistence across rounds** — but, unlike the target's cache, it does
  **NOT** need accept-conditional snapshot/restore. The `crop(start)` call
  (`draft_ops.py:44`) runs INSIDE `forward_dspark_draft_block`, i.e.
  strictly BEFORE `verify_draft_tokens` is ever called for that round
  (`base_evaluator.py:385-393`'s loop calls `propose` then
  `verify_draft_tokens`, in that order) — and the `start` value it crops to
  is the loop's CURRENT-iteration `start`, which was already fully fixed at
  the END of the PREVIOUS iteration (`start_prev + accepted_prev + 1`,
  `base_evaluator.py:424`). So the crop target is deterministic and
  independent of THIS round's (not-yet-known) accept outcome — it
  UNCONDITIONALLY removes exactly this round's `block_size` transient
  noise/speculative positions and nothing else, every single round,
  regardless of what verify later decides. **Contrast the target's cache**,
  which genuinely needs the accept-conditional rollback dance, because the
  target's verify forward writes K/V for ALL `1+draft_token_count`
  positions (most of them still-unverified draft tokens) and a partial
  accept must discard the unverified tail — a fundamentally different
  situation from the draft cache, which only ever appends
  ALREADY-COMMITTED (post-verify) context via `_update`'s
  `[:, :accepted+1, :]` slice (section 1 fact 6) and its own transient,
  always-fully-discarded noise positions. **A Zig port should NOT tie the
  draft cache to the target cache's `KVCache.snapshot/restore` accept-branch
  logic** — the correct design is: monotonic append of one entry per
  COMMITTED token (fed only from verified, post-`_update` context — never
  speculative content), plus a per-round transient append-then-ALWAYS-crop
  of the block's own noise positions, with no accept-conditional branch at
  all. (Also worth being precise about WHAT persists: only `k_ctx/v_ctx`
  from the fused target-hidden taps — one committed entry per committed
  token — survives a crop; `k_noise/v_noise` from a round's own block
  positions is transient scratch that is discarded EVERY round regardless
  of outcome, never resident as "one k_noise/v_noise per committed
  position.") (This document went through two rounds of correction on the
  draft-cache question: first pass wrongly called it ephemeral/scratch
  entirely; second pass correctly identified persistence but wrongly
  assumed it needed the SAME accept-conditional rollback as the target's
  cache. Both errors were caught and independently re-verified before this
  final text; see section 8's design plan, updated to match.)

### Outer loop (`base_evaluator.py:307-441`, `generate_decoding_sample`)

Standard propose → verify → commit → update cycle:
1. Prefill target, sample first token via ordinary sampling (not
   speculative).
2. Loop: `propose` (drafter forward, section 2) → `verify_draft_tokens`
   (above) → write `accepted_draft_tokens+1` tokens into `output_ids` →
   `update` (re-slice `target_hidden_states`, section 1/6's fact 6) →
   advance `start` by `accepted_draft_tokens+1` → crop target KV cache to
   `start`.
3. Stop-token handling: checked BOTH mid-proposal (truncates
   `accepted_draft_tokens` at the first stop token within the accepted
   prefix, `base_evaluator.py:264-276`) and post-commit
   (`has_stop_token(new_token_ids, ...)`, line 428) — two separate checks,
   the first cheaper (avoids emitting past a stop token that was ALREADY
   accepted), the second a safety net for the bonus/correction token.

**Batch size**: hardcoded to 1 throughout (`assert input_ids.size(0) == 1`,
multiple call sites) — the reference implementation is NOT batched. This
matches mlx-serve's own documented constraint that MoE/complex archs
aren't always batchable, but note mlx-serve's PLD/drafter/MTP DO support
concurrent-request batching on batchable archs — **whether DSpark's
non-causal, full-context-refetch-per-round design batches cleanly across
concurrent requests is UNCONFIRMED by the reference code** (it never
attempts it) — see section 9.

---

## 6. Checkpoint tensor inventory — `dspark_qwen3_8b_block7/model.safetensors`

Fetched via HTTP Range request against the safetensors header only (no
weight bytes downloaded). Single unsharded file. `__metadata__: {"format":
"pt"}`. All 64 tensors are **BF16**. Total ~2.371B parameters, ~4.74 GB.

### Embedding / output (3 tensors)
| Tensor | Dtype | Shape | Notes |
|---|---|---|---|
| `embed_tokens.weight` | BF16 | `[151936, 4096]` | Byte-copy of target's input embeddings at training start, then frozen (section 7) |
| `lm_head.weight` | BF16 | `[151936, 4096]` | Byte-copy of target's output embeddings, then frozen (section 7). PyTorch `Linear` convention: apply as `hidden @ lm_head.weight.T` |
| `norm.weight` | BF16 | `[4096]` | Final RMSNorm after the 5-layer backbone |

### Fusion / context projection (2 tensors)
| Tensor | Dtype | Shape | Notes |
|---|---|---|---|
| `fc.weight` | BF16 | `[4096, 20480]` | Bias-free `Linear(20480, 4096)`. 20480 = 5 × 4096 (concat of 5 tapped target layers). PyTorch layout `(out,in)` → apply as `x @ fc.weight.T` |
| `hidden_norm.weight` | BF16 | `[4096]` | RMSNorm applied AFTER `fc`, before use as attention context K/V source |

### Backbone decoder layers (5×, `layers.0` … `layers.4` — 11 tensors each, 55 total)
| Tensor pattern | Dtype | Shape | Notes |
|---|---|---|---|
| `layers.N.input_layernorm.weight` | BF16 | `[4096]` | Pre-attention RMSNorm |
| `layers.N.self_attn.q_proj.weight` | BF16 | `[4096, 4096]` | 32 heads × 128 head_dim, no bias |
| `layers.N.self_attn.k_proj.weight` | BF16 | `[1024, 4096]` | 8 KV heads × 128 head_dim, no bias |
| `layers.N.self_attn.v_proj.weight` | BF16 | `[1024, 4096]` | 8 KV heads × 128 head_dim, no bias |
| `layers.N.self_attn.o_proj.weight` | BF16 | `[4096, 4096]` | No bias |
| `layers.N.self_attn.q_norm.weight` | BF16 | `[128]` | Per-head-dim QK-norm (Qwen3 convention), applied AFTER K-concat with context (section 1) |
| `layers.N.self_attn.k_norm.weight` | BF16 | `[128]` | Same |
| `layers.N.post_attention_layernorm.weight` | BF16 | `[4096]` | Pre-MLP RMSNorm |
| `layers.N.mlp.gate_proj.weight` | BF16 | `[12288, 4096]` | SwiGLU gate, no bias |
| `layers.N.mlp.up_proj.weight` | BF16 | `[12288, 4096]` | SwiGLU up, no bias |
| `layers.N.mlp.down_proj.weight` | BF16 | `[4096, 12288]` | SwiGLU down, no bias |

`N` ranges 0..4 (5 layers total, matching `num_hidden_layers: 5`). Standard
Qwen3 decoder-layer shape throughout: GQA 32:8 heads @ head_dim 128,
QK-norm present, SwiGLU MLP, zero biases anywhere (matches
`attention_bias: false` and the absence of any `.bias` tensor in
attention/MLP).

### Markov head (2 tensors)
| Tensor | Dtype | Shape | Notes |
|---|---|---|---|
| `markov_head.markov_w1.weight` | BF16 | `[151936, 256]` | `nn.Embedding(vocab_size, markov_rank)` — lookup table, row-indexed by token id |
| `markov_head.markov_w2.weight` | BF16 | `[151936, 256]` | `nn.Linear(256, 151936, bias=False)` weight, PyTorch `(out,in)` layout → apply as `latent @ markov_w2.weight.T` |

No `gate_proj`/`joint_proj` tensors present → confirms `VanillaMarkov` is
the only variant instantiated in this checkpoint (section 3).

### Confidence head (2 tensors)
| Tensor | Dtype | Shape | Notes |
|---|---|---|---|
| `confidence_head.proj.weight` | BF16 | `[1, 4352]` | `nn.Linear(4352, 1)` weight. 4352 = 4096 (hidden) + 256 (markov_rank), confirming `confidence_head_with_markov: true` |
| `confidence_head.proj.bias` | BF16 | `[1]` | Scalar bias |

**Total: 3 + 2 + 55 + 2 + 2 = 64 tensors** — matches the safetensors header
count exactly.

**No quantization** — every tensor is plain BF16 (the checkpoint's
`dtype: "bfloat16"` config field and the safetensors header agree). A Zig
loader does NOT need `computeQuantParams`/`QLinear`-style quant-geometry
inference the way `mtp.zig` does for its (often-quantized) sidecars — plain
`mlx_matmul` against dense bf16 weights suffices for every linear in this
checkpoint. (A future quantized DSpark release would need the
`QLinear`/`inferBits` treatment mtp.zig already has, but this specific
checkpoint doesn't require it.)

---

## 7. Target compatibility

**Target**: `Qwen/Qwen3-8B` (confirmed directly —
`config/dspark/dspark_qwen3_8b.py:10`,
`target_model_name_or_path="Qwen/Qwen3-8B"`). Filename `block7` refers to
`block_size=7`, not a model-size suffix.

### What must match between drafter and target

- **`hidden_size`**: draft config is a deep-copy of the target's HF config
  (`qwen3/config.py:37`, `copy.deepcopy(target_config)`) — so hidden_size,
  `rope_theta`, `rms_norm_eps`, `attention_bias`, `hidden_act`, `head_dim`
  are ALWAYS inherited from whatever target this draft was built against.
  A Zig loader should assert the drafter's `hidden_size` equals the
  currently-loaded target's `hidden_size` at bind time (mirrors
  `drafter.zig`'s `error.DrafterTargetMismatch` pattern per project
  CLAUDE.md) rather than trusting the draft checkpoint's config.json blindly.
- **`vocab_size`**: must match the target's tokenizer/vocab exactly — the
  draft's `embed_tokens`/`lm_head`/`markov_head.markov_w1`/`markov_w2` are
  ALL shaped `[151936, ...]`, i.e. sized to Qwen3-8B's vocab. A mismatched
  target vocab would silently corrupt token embeddings/logits.
  `num_target_layers: 36` must equal the bound target's actual decoder
  layer count (needed to validate `target_layer_ids` are in range and to
  size the hidden-state capture points).
- **Tokenizer**: implicitly, since token ids flow directly between draft
  and target (the anchor token id, the mask token id, sampled draft token
  ids) — these must be the SAME tokenizer/vocab space. `mask_token_id:
  151669` is a DSpark-specific reserved id that must exist in — or at
  least not collide destructively with — the target's tokenizer vocabulary
  (Qwen3-8B's tokenizer has 151936 total ids; 151669 is a normal
  in-range id, likely one of Qwen3's reserved/unused special-token slots —
  UNCONFIRMED without inspecting the tokenizer's special-tokens map;
  flagged in section 9).
- **`target_layer_ids` range**: `[1, 9, 17, 25, 33]` must all be valid
  0-indexed decoder-layer indices for the bound target, and must NOT
  include the target's LAST layer index (hard assert in the reference
  code, section 1).

### Does the drafter ship its own embedding/lm_head, or share the target's?

**Ships its own — but they are BYTE-IDENTICAL to the target's by
construction, not independently learned.**

`deepspec/trainer/base_trainer.py` (the `initialize_embeddings_and_head`
call site, ~lines 260-274):
```python
# Training only uses the target checkpoint to initialize frozen draft
# embeddings and lm_head weights.
target_model = AutoModelForCausalLM.from_pretrained(model_args.target_model_name_or_path, ...).to("cpu").eval()
target_embed_tokens = target_model.get_input_embeddings()
target_lm_head = target_model.get_output_embeddings()
draft_model.initialize_embeddings_and_head(embed_tokens=target_embed_tokens, lm_head=target_lm_head, freeze=True)
```
And `Qwen3DSparkModel.initialize_embeddings_and_head`
(`qwen3/modeling.py:270-283`):
```python
def initialize_embeddings_and_head(self, *, embed_tokens, lm_head, freeze=True):
    assert self.embed_tokens.weight.shape == embed_tokens.weight.shape
    assert self.lm_head.weight.shape == lm_head.weight.shape
    with torch.no_grad():
        self.embed_tokens.weight.copy_(embed_tokens.weight.detach())
        self.lm_head.weight.copy_(lm_head.weight.detach())
    if freeze:
        self.set_embedding_head_trainable(False)   # requires_grad_(False)
```

So: at the START of training, the draft's `embed_tokens`/`lm_head` are
copied VERBATIM from the target's own input/output embeddings, then FROZEN
(`requires_grad_(False)`) for the ENTIRE training run — never updated
again. The checkpoint's `embed_tokens.weight`/`lm_head.weight` tensors are
therefore separate storage in the safetensors file, but their VALUES are
(and forever remain) identical to `Qwen/Qwen3-8B`'s own embedding/lm_head
weights.

**Design implication for the Zig port** (a genuine design decision, not
dictated by the reference code, which always runs draft and target as
fully separate HF model objects with no sharing):

- mlx-serve's existing `drafter.zig` precedent (Gemma-4 assistant drafter)
  refcount-SHARES the target's `embed_tokens` rather than loading a
  duplicate copy (`drafter.zig` ~lines 251-294, "Refcount-share the
  target's embed_tokens... The drafter therefore re-runs the target's
  embedding lookup path through its own copy of `embed_w` (== target's
  embed weight)"). mlx-serve's `mtp.zig` sidecar shares the target's
  `lm_head` outright (no separate lm_head tensor in the MTP checkpoint at
  all — `mtp.zig:364-381`, "Project the MTP post-norm hidden through the
  TARGET's lm_head").
- **Recommendation**: mirror this precedent — at DSpark load time, SKIP
  loading `embed_tokens.weight`/`lm_head.weight` from the drafter's own
  safetensors file, and instead refcount-share the ALREADY-LOADED target
  transformer's `embed_tokens`/`lm_head` arrays (saves ~2×
  151936×4096×2 bytes ≈ 2.5 GB of duplicate bf16 weight residency, and
  guarantees byte-identity by construction rather than by checkpoint
  convention). **Caveat**: this is only safe if the values ARE in fact
  identical for the specific target/draft pairing being loaded — the
  reference code's freeze-after-copy convention makes this true for any
  checkpoint produced by this training pipeline, but a Zig loader should
  not assume it silently. Recommend either (a) trusting the convention
  (simplest, matches existing sidecar precedent, fastest to implement), or
  (b) an opt-in debug-mode value-equality assertion at load time (hash or
  spot-check a few rows) if paranoia about a mismatched/hand-edited
  checkpoint is warranted. Loading the draft's own copies unconditionally
  (i.e. NOT sharing) is also valid and simpler to implement correctly on
  a first pass — it costs the extra ~2.5 GB residency but sidesteps the
  "is this really the same target" question entirely. **This document
  recommends starting with the simple/safe option (load the draft's own
  copies) and revisiting sharing as a follow-up optimization once
  correctness is established** — do not let the optimization block the
  initial correctness-focused port.
- **`markov_head.markov_w1`/`markov_w2` and `confidence_head.proj` have NO
  target-side analog** — these must always be loaded from the drafter's
  own checkpoint; there is nothing to share.

---

## 8. Zig port plan

### File layout

New file `src/dspark.zig`, sibling to `src/drafter.zig` and `src/mtp.zig`,
following the SAME conventions (per project CLAUDE.md: "each new
model-specific engine... gets a thin Zig wrapper"; sidecars are
self-contained enough to delete cleanly — "delete this + `Generator.nextMtp`
to remove the feature" is the mtp.zig precedent to match).

### Top-level struct (mirrors `MtpModel` in `mtp.zig:65-128`)

```zig
pub const DsparkModel = struct {
    // Config (validated against the bound target at bind time)
    hidden_size: u32,        // 4096, must == target.hidden_size
    vocab_size: u32,         // 151936, must == target.vocab_size
    num_layers: u32,         // 5 (num_hidden_layers)
    num_heads: u32,          // 32
    num_kv_heads: u32,       // 8
    head_dim: u32,           // 128
    intermediate_size: u32,  // 12288
    rms_norm_eps: f32,       // 1e-6
    rope_theta: f32,         // 1_000_000
    block_size: u32,         // 7 (max draft tokens per round)
    mask_token_id: u32,      // 151669
    target_layer_ids: []const u32,  // [1, 9, 17, 25, 33] (0-indexed target decoder layers)
    markov_rank: u32,        // 256
    // markov_head_type MUST be "vanilla" for this checkpoint — assert at load,
    // do not silently mis-load a future gated/rnn checkpoint.
    confidence_head_with_markov: bool,  // true

    // Weights — bf16 dense (no quant in this checkpoint; see section 6)
    embed_tokens_w: mlx.mlx_array,   // [vocab, hidden] — or refcount-shared from target, see section 7
    lm_head_w: mlx.mlx_array,        // [vocab, hidden] — or refcount-shared from target
    norm_w: mlx.mlx_array,           // [hidden]
    fc_w: mlx.mlx_array,             // [hidden, num_taps*hidden]  (num_taps = target_layer_ids.len = 5)
    hidden_norm_w: mlx.mlx_array,    // [hidden]
    layers: []DsparkLayer,           // 5 entries

    markov_w1: mlx.mlx_array,        // [vocab, markov_rank] embedding table
    markov_w2: mlx.mlx_array,        // [vocab, markov_rank] linear weight ((out,in) layout)
    confidence_proj_w: mlx.mlx_array,  // [1, hidden+markov_rank]
    confidence_proj_b: mlx.mlx_array,  // [1]
};

pub const DsparkLayer = struct {
    input_ln_w: mlx.mlx_array,
    q_proj_w: mlx.mlx_array,   // [hidden, hidden]
    k_proj_w: mlx.mlx_array,   // [kv_dim, hidden]
    v_proj_w: mlx.mlx_array,   // [kv_dim, hidden]
    o_proj_w: mlx.mlx_array,   // [hidden, hidden]
    q_norm_w: mlx.mlx_array,   // [head_dim]
    k_norm_w: mlx.mlx_array,   // [head_dim]
    post_attn_ln_w: mlx.mlx_array,
    gate_proj_w: mlx.mlx_array,   // [intermediate, hidden]
    up_proj_w: mlx.mlx_array,    // [intermediate, hidden]
    down_proj_w: mlx.mlx_array,  // [hidden, intermediate]
};
```

### Weight loading (mirrors `loadMtp`, `mtp.zig:216-278`)

Direct tensor-name binding from `<dspark-dir>/model.safetensors` (or
`dspark/weights.safetensors` if following mtp.zig's sidecar-subdirectory
convention — this is a naming-convention choice, not dictated by the
reference; recommend matching mtp.zig's `mtp/weights.safetensors`
subdirectory pattern for consistency, e.g. `dspark/model.safetensors`).
Since every tensor is plain bf16 (no quant), the loader is simpler than
`mtp.zig`'s (`inferBits`/`inferGroupSize`/`QLinear` are unnecessary here —
plain `mlx_array` handles + `mlx_matmul` suffice). Tensor name → field
binding follows the exact names in section 6's table (e.g.
`"layers.0.self_attn.q_proj.weight"` → `layers[0].q_proj_w`, transposed at
load time the same way `mtp.zig`'s `ownAndTranspose2D` pre-transposes
PyTorch `(out,in)` weights to whatever contraction layout `mlx_matmul`
expects — follow that existing helper's convention exactly for consistency
with the rest of the codebase).

### Forward — the per-round proposal (mirrors the shape of `MtpModel.forward`, `mtp.zig:407-425`, but the BODY differs substantially per sections 1-2 above)

```zig
pub fn proposeBlock(
    self: *const DsparkModel,
    target: *Transformer,
    draft_cache: *DsparkKVCache,        // PERSISTENT per-request, 5-layer, monotonically grows (committed context only) — NOT rebuilt per round, and NOT accept-conditionally rolled back (see section 1/5/8: this is asymmetric with the target's KVCache.snapshot/restore — the draft cache only ever discards its OWN transient per-round noise, unconditionally, never speculative content that needs undoing)
    anchor_token_id: u32,
    target_hidden_taps: mlx.mlx_array,  // [1, committed_since_last_update, num_taps*hidden] — RAW per-layer target hidden concat for ONLY the newly-committed tokens (accepted+1 from last round), see section 1 fact 6
    round_position_ids: []const u32,    // length = ctx_len_this_round + block_size = (draft_cache's current length) + block_size; TRUE global sequence positions, section 1 (resolved)
    s: mlx.mlx_stream,
) !ProposalOut {
    // 1. noise_embedding = embed_tokens([anchor_token_id, mask_id x (block_size-1)])
    // 2. target_ctx = hidden_norm(fc(target_hidden_taps))   -- fresh projection every round from
    //    the FRESH (small) per-round taps; but this round's K/V CONCATENATES with draft_cache's
    //    already-persisted committed-context K/V from prior rounds -- it does not replace it.
    // 3. For each of 5 layers: Qwen3DSparkAttention-style block
    //    (Q from noise stream only; K/V this round = concat(ctx-derived from step 2, noise-derived);
    //     these get APPENDED to draft_cache's persistent per-layer K/V, not swapped in;
    //     q_norm/k_norm AFTER concat; RoPE using round_position_ids (section 1, resolved -- no
    //     ambiguity, cos/sin span exactly ctx_len+block_size and align with K's length);
    //     full non-causal attention over the FULL persisted+fresh K/V -- mask_mode "" / no mask,
    //     NOT "causal"); then SwiGLU MLP, standard pre-norm residual (section 1).
    // 4. final norm -> block_hidden [1, block_size, hidden]
    // 5. base_logits = block_hidden @ lm_head_w.T  -> [1, block_size, vocab]
    // 6. Markov-corrected SEQUENTIAL sample walk (section 3): 7 cheap steps,
    //    NOT a transformer forward -- markov_w2 @ markov_w1[prev_tok], add to
    //    base_logits[pos], sample, feed forward as next prev_tok.
    // 7. Confidence head (section 4): sigmoid(Linear(concat(hidden[pos], markov_w1[prev_tok_at_pos])))
    //    per position; find first position below threshold; truncate proposal there.
    // 8. IMMEDIATELY after this forward completes (BEFORE verify runs, and regardless of what
    //    verify later decides -- draft_ops.py's crop(start) is called unconditionally inside
    //    forward_dspark_draft_block, using a `start` value fixed by the PREVIOUS round's outcome,
    //    never this round's): crop draft_cache back to its PRE-round length, discarding exactly
    //    this round's transient block_size noise/speculative positions. This is NOT an
    //    accept-conditional operation and does NOT need target-cache-style snapshot/restore --
    //    there is no "abandon this round" branch for the draft cache, because it never wrote
    //    anything except transient scratch that gets discarded every round unconditionally.
    //    The NEXT round's committed-context append (one k_ctx/v_ctx entry per newly-accepted
    //    token) happens separately, fed from the VERIFY forward's own hidden-state output
    //    (section 1 fact 6) -- not from anything written by this draft forward.
    ...
}
```

### Target-side hidden-state capture (NEW capture point, not an existing one)

Per section 1's analysis: mlx-serve's existing `forwardCaptureHidden`
captures the POST-FINAL-NORM hidden at the LAST position only (used by
drafter.zig/PLD). DSpark needs something DIFFERENT: **RAW (pre-final-norm)
per-layer decoder outputs at 5 SPECIFIC layer indices, for ALL positions in
the newly-committed range** (not just the last position — section 1's
fact 6 shows the whole `[:, :accepted+1, :]` range is kept). This is a NEW
capture mechanism to add to `transformer.zig`'s forward pass — likely a
`capture_hidden_layers: ?[]const u32` option on `ForwardCtx` that stashes a
copy of each targeted layer's output tensor as it's produced, analogous to
existing `capture_hidden`/`capture_hidden_all` flags (per project CLAUDE.md,
`ForwardCtx.capture_hidden_all` already exists for MTP's committed-history
mechanism — study that flag's exact implementation in transformer.zig as
the nearest precedent, since it already solves "capture multiple positions'
hidden states from one forward pass", just not "multiple LAYERS' hidden
states").

### Verify/accept — reuse mlx-serve's EXISTING machinery, do not reimplement

Per section 5's analysis, DSpark's verify math is the same Leviathan
rejection-sampling family mlx-serve already implements for
PLD/drafter/MTP. The Zig port should:
- Build `verify_input = [t1, draft[0..proposal_len-1]]` (length
  `1+proposal_len`, `proposal_len` possibly `< block_size` per section 4's
  confidence gate) exactly as `nextMtp`/`nextDrafter`/`nextPld` already do.
- Run ONE target forward over `verify_input`, using mlx-serve's EXISTING
  `KVCache.snapshot()`/`.restore()` for rollback (per section 5, this is
  functionally equivalent to the reference's HF-`Cache.crop`-based
  approach — same outcome, different mechanism, and mlx-serve's own
  mechanism is what the rest of the codebase already expects).
  **Additionally capture this SAME verify forward's per-tapped-layer raw
  hidden states** (needed for section 1 fact 6's `_update` step — the next
  round's `target_hidden_taps` comes from THIS verify forward's own
  hidden-state outputs, sliced to `[:accepted+1]`) — this is a genuine
  DUAL-PURPOSE requirement on the verify forward (produce both `verify_logits`
  AND the 5-layer hidden-state taps in one pass), similar in spirit to how
  `nextMtp`'s verify forward already produces both `verify_logits` and
  `new_hidden`/`verify_hidden_all` in one `forwardWithCaptureAll` call
  (`generate.zig:2331-2336`) — that function is the closest existing
  precedent for "one forward, multiple captured outputs," though its
  capture is single-layer (post-final-norm) rather than DSpark's
  multi-layer raw capture.
- Compute `accept_prob = min(1, target_p[draft_i]/draft_p[draft_i])` per
  position using the ACTUAL Markov-corrected `draft_probs` (section 3 — NOT
  PLD's implicit one-hot), residual-sample the correction position, exactly
  per section 5's formula (matches mlx-serve's existing "index is
  `accepted` not `accepted-1`" invariant with no divergence — confirmed in
  section 5).
- **The DRAFTER's own internal KV state PERSISTS across rounds, but does
  NOT need accept-conditional `KVCache.snapshot/restore` treatment** —
  this is NOT symmetric with the target's cache, and getting this
  asymmetry right matters (see the fully-corrected section 5 discussion:
  this document went through two rounds of error here — first wrongly
  ephemeral, then wrongly assumed symmetric-with-target rollback — both
  independently re-verified and corrected). Concretely: the draft cache
  holds one `k_ctx/v_ctx` entry (from the fused target-hidden taps) per
  COMMITTED token across the whole generation — growing by `accepted+1`
  per round, same rate as the target's cache — but this content is fed
  ONLY from already-verified, post-`_update` context (section 1 fact 6),
  never from speculative/unverified content. The block's own
  `k_noise/v_noise` positions are pure per-round TRANSIENT scratch:
  appended during the draft forward, then UNCONDITIONALLY cropped away at
  the end of that same forward — before verify even runs, regardless of
  what verify later decides (`draft_ops.py:44`'s `crop(start)` uses a
  `start` value fixed by the PREVIOUS round's outcome, never the current
  one). So there is no "partial accept" branch to roll back to for the
  draft cache — it is a simple monotonic append (committed context) plus
  an unconditional append-then-crop (this round's noise), full stop. A Zig
  port should implement it that way — NOT wired into the target cache's
  `KVCache.snapshot/restore` accept/reject branch at all. This is 5
  layers' worth of persistent K/V (one structure per backbone layer, sized
  to the committed generation length so far) — materially LARGER than
  `mtp.zig`'s single-layer cache, proportional to DSpark's own layer count
  (5), but SIMPLER in its rollback semantics than the target cache (no
  conditional restore path needed at all).

### Dispatch integration (`generate.zig`)

Mirrors `nextMtp`'s shape (`generate.zig:2221` onward) as the closest
existing analog — a new `Generator.nextDspark` function following the same
phase structure (snapshot target KV → draft round → verify forward →
accept-decision → commit/rollback → update `target_hidden_taps` for next
round). Needs a new `spec_disabled_runtime` interaction identical to
`nextMtp`'s (same sticky-disable-on-low-acceptance contract, same
`next()`-transition-shim hand-off). Priority relative to existing
MTP/drafter/PLD: **UNSPECIFIED by anything in scope here** — this is a
product/config decision for whoever wires it in (does a Qwen3-8B target
with BOTH a native MTP sidecar AND a DSpark drafter available prefer one
over the other? The reference repo treats DSpark, EAGLE-3, and DFlash as
mutually exclusive standalone evaluators — it never runs two together —
so there's no reference answer to crib from). Needs its own CLI flag
(e.g. `--dspark <dir>`, mirroring `--drafter <dir>`) and per-request
`enable_dspark`/`confidence_threshold` override fields on the four HTTP
surfaces, per the project's stated convention that "PLD, drafter, and MTP
dispatch on ALL FOUR HTTP surfaces... in both streaming and non-streaming
modes" and the explicit warning that "two non-streaming call sites shipped
with a hardcoded `use_drafter=false` for a month because output-equality
tests can't see a silent fallback to regular decode" — a DSpark port MUST
extend the SAME engagement-count testing pattern
(`tests/test_drafter_tools.sh`-style) rather than trusting equivalence
tests alone.

### Testing (per project's mandatory-TDD convention)

Following the `tests/test_mtp_equivalence.sh` precedent (closest existing
analog — a checkpoint-loaded sidecar with a non-trivial verify contract):
- Unit tests at the bottom of `dspark.zig` (Zig convention) for: Markov
  bias math (a hand-computed small-rank case), confidence-gate truncation
  logic (`_confident_prefix_length`'s boundary cases — all-confident,
  first-position-low, threshold<=0), tensor-shape validation against a
  mismatched target (the `error.DsparkTargetMismatch`-style assert).
- An integration equivalence test mirroring
  `tests/test_mtp_equivalence.sh`: load `dspark_qwen3_8b_block7` against
  `Qwen/Qwen3-8B` (or an MLX-converted equivalent), diff first-N-token
  temp-0 output against `--no-dspark` baseline (per-token EXACT equality
  is not guaranteed at temp=0 across float-reduction-order differences —
  see the project's own PLD/drafter/MTP note on this — so this test should
  follow the same "first ~30-80 tokens byte-identical, engagement count via
  `[spec-stats] mode=dspark` log line" pattern as the existing sidecars,
  NOT expect indefinite exact-match).
- An ACCEPTANCE-FLOOR check analogous to MTP's (per project CLAUDE.md:
  "a structurally broken head... engages, accepts ~0%... passes both
  equivalence and engagement" without a floor check) — this is
  ESPECIALLY important for DSpark given how easy it would be to
  mis-implement the draft-cache persistence / RoPE-offset interaction
  described in sections 1, 5, and 8 (this document's author initially
  mis-traced it and concluded the draft cache was ephemeral before
  independent re-verification caught the error): a Zig port that
  incorrectly rebuilds the draft cache fresh each round instead of
  persisting it (or gets the resulting position-id/RoPE-offset arithmetic
  subtly wrong) would produce a head that "works" (doesn't crash, produces
  valid token ids, may even pass a SHORT equivalence test if early rounds
  happen to still align) but accepts near-0% of its proposals once the
  committed context grows past the first round — which only an
  acceptance-floor test run over a reasonably long generation would catch.
  **Concretely: round 1 alone cannot exercise the subtlest part of this
  design** — round 1's `target_hidden_states` comes straight from the
  initial prefill (`_init_context`, full prompt, no accumulated-cache
  interaction yet), so a cache-append bug specifically in the "append
  round r's newly-committed context, sourced from round r-1's verify
  hiddens, rotated with the `[start_{r-1} : start_r+block_size]` position
  slice, concatenated with round r's fresh noise" step (the single
  subtlest spot in the whole port — see section 1/8) can only manifest
  starting at round 2. **The acceptance-floor test MUST run long enough to
  force at least 2-3 full accept/verify rounds** (i.e. more than
  `block_size` newly-generated tokens with at least one round NOT fully
  accepted, so the committed-context append path actually executes) — a
  test that stops after round 1 would pass even with this exact bug
  present, giving false confidence.

---

## 9. Ambiguities / unconfirmable items

**(a) RoPE application to the concatenated K at EVAL time — RESOLVED, not
an ambiguity.** (This item was originally written up as "the single most
consequential open question, needs a live trace to resolve" — that framing
was WRONG, caused by mis-tracing the `get_seq_length()`/`crop(start)`
interaction; the arithmetic was worked out fully from code already read,
verified by direct simulation, and is now documented in section 1.
Restating the resolution here for completeness: `draft_position_ids`
(`draft_ops.py:32-34`) is `position_ids[:, past_key_values_draft.get_seq_length()
: start + block_size]`, and because the draft's own KV cache PERSISTS
across rounds (it is one `DynamicCache()` object threaded through every
round via `context.past_key_values_draft`, `evaluator.py:92`, cropped but
never recreated), `get_seq_length()` at round entry equals the PREVIOUS
round's `start`, not the current one. This makes the slice length exactly
`ctx_len + block_size` in every round — matching K's length precisely, no
mismatch, no broadcast trick needed. The target context K/V therefore DOES
get correctly-shaped fresh RoPE at eval time, using each committed
position's own true absolute sequence position (the low end of the
`position_ids` slice) — consistent with the single global `torch.arange`
built once for the whole generation (`base_evaluator.py:342`). No live
trace or further verification against the reference is needed for this
specific question — it is resolved by the code as read. The DOWNSTREAM
consequence — the draft cache must persist (monotonically append)
committed context across rounds, though WITHOUT the target cache's
accept-conditional snapshot/restore semantics (a genuine asymmetry, not a
mirrored mechanism) — is the part that takes real implementation care; see
the corrected sections 1, 5, and 8.**

**(b) `mask_token_id: 151669`'s relationship to Qwen3-8B's tokenizer.**
Not verified against the actual Qwen3-8B tokenizer's special-tokens map
(would require fetching `tokenizer_config.json`/`special_tokens_map.json`
from `Qwen/Qwen3-8B`'s HF repo, not fetched in this session — out of the
task's explicit scope, which asked only for the drafter checkpoint's
config + safetensors index). Likely a normal in-vocab id repurposed for
this training pipeline (151936 total ids; Qwen3 models typically reserve a
block of ids past the "real" vocabulary for exactly this kind of use) but
UNCONFIRMED.

**(c) `num_anchors: 512`'s role at INFERENCE time.** Confirmed as a
TRAINING-only hyperparameter (`sample_anchor_positions`,
`common.py:123-169` — governs how many random block-anchor positions get
sampled per training sequence) via `qwen3/modeling.py:398-403`'s `forward`
method (the TRAINING forward). The EVAL path
(`evaluator.py`/`draft_ops.py`) never references `num_anchors` or
`sample_anchor_positions` at all. **Confirmed NOT needed for the Zig
inference-only port** — included in `DsparkModel`'s config struct in
section 8 only for completeness/config-validation parity, not because any
inference code path reads it.

**(d) Batching / concurrent-request behavior.** The reference
implementation is hardcoded to `batch_size=1` throughout (section 5) and
never exercises concurrent requests. Whether DSpark's non-causal,
K/V-refetch-per-round design (fusing a per-request `target_hidden_taps`
tensor into every layer's K/V) batches cleanly across concurrent slots the
way mlx-serve's PLD/drafter/MTP already do on batchable archs is
UNCONFIRMED — this needs original design work during implementation, not
just a port of the reference. Note this is a BIGGER lift than it might
first appear: per the corrected section 8 design, each request needs its
OWN persistent 5-layer draft KV cache (not a stateless per-round scratch
buffer — see the corrected sections 1/5/8), so batching DSpark across
concurrent slots means managing N per-request 5-layer caches of growing
size in lockstep with N per-request target caches, not just N independent
stateless forward calls. A conservative first port should probably force
`max_concurrent=1`-equivalent serialization when DSpark is active
(mirroring how MoE+drafter is handled per project CLAUDE.md: "MoE default-off"
as a precedent for "when in doubt, don't enable speculative-decode
batching on a newly-ported architecture until it's specifically verified"),
then relax that constraint as a follow-up once batching correctness is
established.

**(e) `confidence_threshold`'s default/runtime source.** The reference
evaluator reads it from `self.args.confidence_threshold` (a CLI/eval-script
argument, not part of the checkpoint's `config.json` at all) — so there is
NO "correct" default baked into the checkpoint; it's a serving-time policy
choice. Section 4 confirms `threshold<=0` disables the adaptive-length gate
entirely (always proposes the full `block_size`). A Zig port needs to pick
its own default (e.g. mirroring mlx-serve's own
`RUNTIME_GATE_MIN_PER_DRAFT_RATE=0.50` precedent of "start conservative,
tune from measurement" rather than inventing a number) and expose it as a
per-request override, consistent with how `spec_gate_threshold` and
`RUNTIME_GATE_MIN_PER_DRAFT_RATE` are already runtime-tunable constants
elsewhere in the codebase.

**(f) Whether `target_layer_ids` semantics generalize or are checkpoint-specific.**
Section 1 confirms the exact indexing rule (`hidden_states[layer_id+1]`,
never the final layer) is a property of the REFERENCE CODE (i.e. any
DSpark checkpoint, not just this one) — so this is NOT flagged as an
ambiguity, just noting for completeness that a Zig loader should read
`target_layer_ids` from the DRAFT checkpoint's own config.json at load
time (not hardcode `[1,9,17,25,33]`), since a different DSpark checkpoint
(e.g. `dspark_qwen3_4b_block7` or `dspark_qwen3_14b_block7`, both present
in the sibling collection per the HF Space's `models` list) would have
different values sized to its own target's layer count.
