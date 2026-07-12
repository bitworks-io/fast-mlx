#!/usr/bin/env python3
"""Generate the authoritative PyTorch/speculators head-math parity fixture."""

import argparse
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

# The reference must never attach or download the 32B verifier.
if sys.version_info < (3, 10):
    raise RuntimeError("the pinned reference environment requires Python 3.10 or newer")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import torch
from safetensors.torch import load_file, save_file
from torch.nn.attention.flex_attention import create_block_mask
from transformers import DynamicCache

from inspect_checkpoint import inspect
from preflight_core import CheckpointSpec, read_harness_git_sha
from speculators.models.eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
from speculators.models.eagle3.attention import create_combined_mask_mod


PINNED_VERSIONS = {
    "speculators": "0.6.0",
    "torch": "2.12.0",
    "transformers": "5.10.4",
    "safetensors": "0.8.0",
}


def require_pinned_versions() -> None:
    mismatches = []
    for package, expected in PINNED_VERSIONS.items():
        actual = version(package)
        if actual != expected:
            mismatches.append(f"{package}: expected {expected}, got {actual}")
    if mismatches:
        raise RuntimeError("reference environment drift: " + "; ".join(mismatches))


def build_model(head_directory: Path):
    raw = json.loads((head_directory / "config.json").read_text(encoding="utf-8"))
    CheckpointSpec.from_config(raw)
    config = Eagle3SpeculatorConfig.model_validate(raw)
    config.eagle_aux_hidden_state_layer_ids = [2, 32, 61]
    model = Eagle3DraftModel(config)
    missing, unexpected = model.load_state_dict(
        load_file(head_directory / "model.safetensors"), strict=False)
    allowed_missing = {"verifier_lm_head.weight", "verifier_norm.weight"}
    if set(missing) != allowed_missing or unexpected:
        raise RuntimeError(
            f"authoritative head load mismatch: missing={missing}, unexpected={unexpected}")
    return model.to(torch.bfloat16).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--harness-sha-file", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.sequence_length < 2:
        raise ValueError("sequence length must be at least two so RoPE affects relative attention")

    harness_git_sha = read_harness_git_sha(
        args.harness_sha_file, allow_dirty=args.allow_dirty)
    require_pinned_versions()
    identity = inspect(args.head)
    model = build_model(args.head)
    spec = CheckpointSpec.from_config(
        json.loads((args.head / "config.json").read_text(encoding="utf-8")))

    generator = torch.Generator().manual_seed(args.seed)
    input_ids = torch.randint(
        0,
        spec.target_vocab_size,
        (1, args.sequence_length),
        generator=generator,
        dtype=torch.long,
    )
    hidden_states = torch.randn(
        (1, args.sequence_length, 3 * spec.hidden_size),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16) * 0.01
    lengths = torch.tensor([args.sequence_length], dtype=torch.long)
    position_ids = 1 + torch.arange(args.sequence_length, dtype=torch.long).unsqueeze(0)
    mask = create_block_mask(
        create_combined_mask_mod(lengths, args.sequence_length),
        B=None,
        H=None,
        Q_LEN=args.sequence_length,
        KV_LEN=args.sequence_length,
        device=hidden_states.device,
    )

    with torch.no_grad():
        fused_hidden = model.fc(hidden_states)
        layer_input = torch.cat([model.embed_tokens(input_ids), fused_hidden], dim=-1)
        position_embeddings = model.rotary_emb(layer_input, position_ids)
        cache = DynamicCache(config=model.config.transformer_layer_config)
        hidden = model.layers[0](
            layer_input,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=cache,
            cache_position=torch.arange(args.sequence_length),
            position_embeddings=position_embeddings,
        )
        draft_logits = model.lm_head(model.norm(hidden))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint_revision": identity["revision"],
        "checkpoint_blob_id": identity["model_blob_id"],
        "checkpoint_model_sha256": identity["model_sha256"],
        "checkpoint_config_sha256": identity["config_sha256"],
        "harness_git_sha": harness_git_sha,
        "position_offset": "1",
        "seed": str(args.seed),
        "sequence_length": str(args.sequence_length),
        **{f"version_{key}": value for key, value in PINNED_VERSIONS.items()},
    }
    save_file(
        {
            "input_ids": input_ids.cpu().contiguous(),
            "hidden_states": hidden_states.cpu().contiguous(),
            "draft_logits": draft_logits.cpu().contiguous(),
        },
        args.output,
        metadata=metadata,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "output": args.output.name,
                "shape": list(draft_logits.shape),
                "argmax": draft_logits.argmax(dim=-1).tolist(),
                "metadata": metadata,
            },
            indent=2,
            sort_keys=True,
        ))


if __name__ == "__main__":
    main()
