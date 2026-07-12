#!/usr/bin/env python3
"""Validate the pinned RedHat Qwen3-32B EAGLE-3 checkpoint without loading weights."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from preflight_core import (
    CheckpointSpec,
    PreflightValidationError,
    read_harness_git_sha,
    read_safetensors_layout,
    resolve_huggingface_revision,
    sha256_file,
    validate_tensor_manifest,
)


PINNED_REPOSITORY = "RedHatAI/Qwen3-32B-speculator.eagle3"
PINNED_REVISION = "dc84fe7ff1db31efa824776f49c141fc8195eb47"
PINNED_BLOB_ID = "e6343761b2ca1ac528c6eb09b13f1e1f880b8a2cde20c5f42105338f15ed26b8"
PINNED_FILE_SIZE = 3_121_274_856
PINNED_CONFIG_SHA256 = "eaeecf9f0c630187039d9d423214cb42bb3ae7013ed2a6ab93a3ef69c0912fbd"


def inspect(head_directory: Path) -> dict:
    config_path = head_directory / "config.json"
    weights_path = head_directory / "model.safetensors"
    try:
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
    except OSError as error:
        raise PreflightValidationError(f"cannot read checkpoint config: {config_path}") from error
    except json.JSONDecodeError as error:
        raise PreflightValidationError(f"checkpoint config is invalid JSON: {config_path}") from error

    spec = CheckpointSpec.from_config(config)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if config_sha256 != PINNED_CONFIG_SHA256:
        raise PreflightValidationError(
            f"checkpoint config SHA-256 changed: expected {PINNED_CONFIG_SHA256}, "
            f"got {config_sha256}")
    header, data_size = read_safetensors_layout(weights_path)
    validate_tensor_manifest(header, data_size=data_size)
    revision, blob_id = resolve_huggingface_revision(head_directory)
    if revision != PINNED_REVISION:
        raise PreflightValidationError(
            f"checkpoint revision changed: expected {PINNED_REVISION}, got {revision}")
    if blob_id != PINNED_BLOB_ID:
        raise PreflightValidationError(
            f"checkpoint blob changed: expected {PINNED_BLOB_ID}, got {blob_id}")
    file_size = weights_path.stat().st_size
    if file_size != PINNED_FILE_SIZE:
        raise PreflightValidationError(
            f"checkpoint size changed: expected {PINNED_FILE_SIZE}, got {file_size}")
    model_sha256 = sha256_file(weights_path)
    if model_sha256 != PINNED_BLOB_ID:
        raise PreflightValidationError(
            f"checkpoint weight SHA-256 changed: expected {PINNED_BLOB_ID}, "
            f"got {model_sha256}")

    return {
        "status": "pass",
        "repository": PINNED_REPOSITORY,
        "revision": revision,
        "model_blob_id": blob_id,
        "model_sha256": model_sha256,
        "model_file_bytes": file_size,
        "config_sha256": config_sha256,
        "tensor_count": len(CheckpointSpec.expected_tensor_manifest()),
        "config": asdict(spec),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=Path, required=True, help="Downloaded checkpoint directory")
    parser.add_argument("--harness-sha-file", type=Path, required=True)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow diagnostic evidence from an explicitly dirty tree",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON evidence path")
    args = parser.parse_args()

    harness_git_sha = read_harness_git_sha(
        args.harness_sha_file, allow_dirty=args.allow_dirty)
    result = inspect(args.head)
    result["harness_git_sha"] = harness_git_sha
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
