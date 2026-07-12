"""Faithful MLX implementation of the pinned RedHat Qwen3-32B EAGLE-3 head."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache
from mlx_lm.models.rope_utils import initialize_rope

from preflight_core import CheckpointSpec


class Eagle3Attention(nn.Module):
    def __init__(self, spec: CheckpointSpec):
        super().__init__()
        self.num_heads = spec.num_attention_heads
        self.num_kv_heads = spec.num_key_value_heads
        self.head_dim = spec.head_dim
        self.scale = spec.head_dim**-0.5
        input_width = 2 * spec.hidden_size
        self.q_proj = nn.Linear(input_width, spec.query_width, bias=False)
        self.k_proj = nn.Linear(input_width, spec.kv_width, bias=False)
        self.v_proj = nn.Linear(input_width, spec.kv_width, bias=False)
        self.o_proj = nn.Linear(spec.query_width, spec.hidden_size, bias=False)
        self.rope = initialize_rope(
            spec.head_dim,
            base=spec.rope_theta,
            traditional=False,
            scaling_config=None,
            max_position_embeddings=40960,
        )

    def __call__(self, x, mask=None, cache=None, position_offset=1):
        batch, length, _ = x.shape
        queries = self.q_proj(x).reshape(
            batch, length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(
            batch, length, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(
            batch, length, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        offset = position_offset + (cache.offset if cache is not None else 0)
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(
            batch, length, self.num_heads * self.head_dim)
        return self.o_proj(output)


class Eagle3MLP(nn.Module):
    def __init__(self, spec: CheckpointSpec):
        super().__init__()
        self.gate_proj = nn.Linear(spec.hidden_size, spec.intermediate_size, bias=False)
        self.up_proj = nn.Linear(spec.hidden_size, spec.intermediate_size, bias=False)
        self.down_proj = nn.Linear(spec.intermediate_size, spec.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Eagle3DecoderLayer(nn.Module):
    def __init__(self, spec: CheckpointSpec):
        super().__init__()
        self.hidden_size = spec.hidden_size
        self.input_layernorm = nn.RMSNorm(spec.hidden_size, eps=spec.rms_norm_eps)
        self.hidden_norm = nn.RMSNorm(spec.hidden_size, eps=spec.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            spec.hidden_size, eps=spec.rms_norm_eps)
        self.self_attn = Eagle3Attention(spec)
        self.mlp = Eagle3MLP(spec)

    def __call__(self, layer_input, mask=None, cache=None, position_offset=1):
        embeds = layer_input[..., : self.hidden_size]
        hidden = layer_input[..., self.hidden_size :]
        hidden = self.hidden_norm(hidden)
        # The checkpoint pins norm_before_residual=true.
        residual = hidden
        embeds = self.input_layernorm(embeds)
        attention_input = mx.concatenate([embeds, hidden], axis=-1)
        hidden = residual + self.self_attn(
            attention_input,
            mask=mask,
            cache=cache,
            position_offset=position_offset,
        )
        residual = hidden
        hidden = self.mlp(self.post_attention_layernorm(hidden))
        return residual + hidden


class Eagle3Head(nn.Module):
    def __init__(self, spec: CheckpointSpec):
        super().__init__()
        self.spec = spec
        self.embed_tokens = nn.Embedding(spec.target_vocab_size, spec.hidden_size)
        self.fc = nn.Linear(3 * spec.hidden_size, spec.hidden_size, bias=False)
        self.layers = [Eagle3DecoderLayer(spec)]
        self.norm = nn.RMSNorm(spec.hidden_size, eps=spec.rms_norm_eps)
        self.lm_head = nn.Linear(spec.hidden_size, spec.draft_vocab_size, bias=False)

    @classmethod
    def load(cls, head_directory: Path) -> "Eagle3Head":
        raw_config = json.loads((head_directory / "config.json").read_text(encoding="utf-8"))
        model = cls(CheckpointSpec.from_config(raw_config))
        weights = mx.load(str(head_directory / "model.safetensors"))
        d2t = weights.pop("d2t")
        t2d = weights.pop("t2d")
        model.load_weights(list(weights.items()))
        model.d2t = d2t
        model.t2d = t2d
        mx.eval(model.parameters(), model.d2t, model.t2d)
        return model

    @staticmethod
    def new_cache():
        return KVCache()

    def fuse_target_hidden(self, target_hidden_states):
        return self.fc(target_hidden_states)

    def forward_fused(self, input_ids, fused_hidden, cache=None, position_offset=1):
        embeddings = self.embed_tokens(input_ids)
        layer_input = mx.concatenate([embeddings, fused_hidden], axis=-1)
        mask = create_attention_mask(layer_input, cache)
        hidden = self.layers[0](
            layer_input,
            mask=mask,
            cache=cache,
            position_offset=position_offset,
        )
        logits = self.lm_head(self.norm(hidden))
        return logits, hidden

    def __call__(self, input_ids, target_hidden_states, cache=None, position_offset=1):
        return self.forward_fused(
            input_ids,
            self.fuse_target_hidden(target_hidden_states),
            cache=cache,
            position_offset=position_offset,
        )

    def map_draft_to_target(self, draft_token_ids):
        return draft_token_ids + self.d2t[draft_token_ids]
