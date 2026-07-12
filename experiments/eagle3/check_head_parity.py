#!/usr/bin/env python3
"""Fidelity gate: compare MLX head logits with the pinned speculators fixture."""

import argparse
import json
import math
from pathlib import Path

import mlx.core as mx

from eagle_head_mlx import Eagle3Head
from inspect_checkpoint import PINNED_BLOB_ID, PINNED_REVISION, inspect
from preflight_core import read_harness_git_sha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--harness-sha-file", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-cosine", type=float, default=0.99)
    args = parser.parse_args()
    if not math.isfinite(args.minimum_cosine) or not 0.99 <= args.minimum_cosine <= 1:
        parser.error("--minimum-cosine must be finite and in 0.99...1.0")

    # llmbench has an explicitly raised wired-memory ceiling; never inherit MLX's cache default.
    mx.set_cache_limit(8 << 30)
    harness_git_sha = read_harness_git_sha(
        args.harness_sha_file, allow_dirty=args.allow_dirty)
    identity = inspect(args.head)
    fixture, metadata = mx.load(str(args.fixture), return_metadata=True)
    if metadata.get("checkpoint_revision") != PINNED_REVISION:
        raise RuntimeError(
            f"fixture revision mismatch: {metadata.get('checkpoint_revision')!r}")
    if metadata.get("checkpoint_blob_id") != PINNED_BLOB_ID:
        raise RuntimeError(f"fixture blob mismatch: {metadata.get('checkpoint_blob_id')!r}")
    if metadata.get("checkpoint_model_sha256") != identity["model_sha256"]:
        raise RuntimeError(
            "fixture weight hash mismatch: "
            f"{metadata.get('checkpoint_model_sha256')!r}")
    if metadata.get("checkpoint_config_sha256") != identity["config_sha256"]:
        raise RuntimeError(
            "fixture config hash mismatch: "
            f"{metadata.get('checkpoint_config_sha256')!r}")
    if metadata.get("harness_git_sha") != harness_git_sha:
        raise RuntimeError(
            "fixture was generated from a different tree: "
            f"fixture={metadata.get('harness_git_sha')!r}, current={harness_git_sha!r}")

    model = Eagle3Head.load(args.head)
    logits, _ = model(
        fixture["input_ids"],
        fixture["hidden_states"],
        position_offset=int(metadata["position_offset"]),
    )
    logits = logits.astype(mx.float32)
    reference = fixture["draft_logits"].astype(mx.float32)
    mx.eval(logits, reference)

    observed = logits.reshape(-1)
    expected = reference.reshape(-1)
    numerator = (observed * expected).sum()
    denominator = mx.sqrt((observed * observed).sum()) * mx.sqrt(
        (expected * expected).sum())
    cosine = float(numerator / (denominator + 1e-12))
    observed_argmax = mx.argmax(logits, axis=-1)
    expected_argmax = mx.argmax(reference, axis=-1)
    argmax_match = float(
        (observed_argmax == expected_argmax).astype(mx.float32).mean())
    max_absolute_error = float(mx.max(mx.abs(logits - reference)))
    passed = cosine > args.minimum_cosine
    result = {
        "status": "pass" if passed else "fail",
        "minimum_cosine": args.minimum_cosine,
        "cosine_similarity": cosine,
        "argmax_match_rate": argmax_match,
        "max_absolute_error": max_absolute_error,
        "observed_argmax": observed_argmax.tolist(),
        "expected_argmax": expected_argmax.tolist(),
        "fixture_metadata": metadata,
        "checkpoint_identity": identity,
        "harness_git_sha": harness_git_sha,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not passed:
        raise SystemExit(
            f"PARITY FAIL: cosine {cosine:.6f} <= {args.minimum_cosine:.6f}")


if __name__ == "__main__":
    main()
