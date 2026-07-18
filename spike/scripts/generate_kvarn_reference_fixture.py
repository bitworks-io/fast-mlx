#!/usr/bin/env python3
"""Generate the fast-mlx KVarN oracle fixture from a pinned official checkout.

This script deliberately imports the official pure-PyTorch reference files from a caller-supplied
checkout. It does not import or vendor the vLLM runtime. The resulting JSON records the source
commit, source-file hashes, Torch version, input tile, packed bytes, fp16 metadata bit patterns,
and dequantized output used by the pure Swift conformance test.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


EXPECTED_COMMIT = "7586257f1c632e63187bfacbbe21ccb51540f7b3"
SINKHORN_MODULE = "vllm.model_executor.layers.quantization.kvarn.sinkhorn"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalized_hadamard(dimension: int) -> torch.Tensor:
    matrix = torch.ones(1, 1, dtype=torch.float32)
    while matrix.shape[0] < dimension:
        matrix = torch.cat(
            [torch.cat([matrix, matrix], dim=1), torch.cat([matrix, -matrix], dim=1)],
            dim=0,
        )
    return matrix / dimension**0.5


def fp16_bits(values: torch.Tensor) -> list[int]:
    return values.detach().cpu().contiguous().view(torch.uint16).reshape(-1).tolist()


def floats(values: torch.Tensor) -> list[float]:
    return values.detach().cpu().float().reshape(-1).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    upstream = args.upstream.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_COMMIT:
        raise SystemExit(f"expected KVarN {EXPECTED_COMMIT}, found {commit}")
    dirty = subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if dirty:
        raise SystemExit("pinned KVarN checkout has modified tracked files")

    # The batch RTN helper reads this ablation from the process environment at call time. Pin the
    # source-default min/max path so a developer shell cannot silently change the checked fixture.
    os.environ["KVARN_RTN_QUANTILE"] = "0"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    sinkhorn_path = (
        upstream / "vllm/model_executor/layers/quantization/kvarn/sinkhorn.py"
    )
    store_path = upstream / "vllm/v1/attention/ops/kvarn_store.py"
    decode_path = upstream / "vllm/v1/attention/ops/kvarn_decode.py"
    sinkhorn = load_module(SINKHORN_MODULE, sinkhorn_path)
    store = load_module("fast_mlx_oracle.kvarn_store", store_path)
    decode = load_module("fast_mlx_oracle.kvarn_decode", decode_path)

    # D=G=4 keeps the Hadamard factor exactly representable (0.5), while exercising both
    # orientations, alternating variance normalization, 4-bit pairs, and 2-bit quads.
    dimension = 4
    group = 4
    iterations = 16
    keys = torch.tensor(
        [
            [-1.50, -0.50, 0.50, 1.50],
            [0.25, -1.25, 1.75, -0.75],
            [2.00, 0.50, -1.00, -0.25],
            [-0.50, 1.25, 0.25, -2.00],
        ],
        dtype=torch.float32,
    )
    values = torch.tensor(
        [
            [1.25, -0.25, -1.50, 0.50],
            [-1.75, 0.75, 0.25, 1.00],
            [0.50, 1.50, -0.75, -1.25],
            [2.25, -1.00, 0.75, -0.50],
        ],
        dtype=torch.float32,
    )

    hadamard = normalized_hadamard(dimension)
    key_rotated = (keys @ hadamard).transpose(0, 1).contiguous()
    value_rotated = (values @ hadamard).contiguous()

    k_balanced, k_col, k_row = sinkhorn.variance_normalize(
        key_rotated, iterations=iterations
    )
    v_balanced, v_col, v_row = sinkhorn.variance_normalize(
        value_rotated, iterations=iterations
    )
    k_record = store.kvarn_store_tile_k_batch_from_sinkhorn(
        k_balanced.unsqueeze(0),
        k_col.squeeze(0).unsqueeze(0),
        k_row.squeeze(-1).unsqueeze(0),
        bits=4,
    )
    v_record = store.kvarn_store_tile_v_batch_from_sinkhorn(
        v_balanced.unsqueeze(0),
        v_col.squeeze(0).unsqueeze(0),
        v_row.squeeze(-1).unsqueeze(0),
        bits=2,
    )

    key_dequant_rotated = decode.kvarn_dequant_tile_k(
        k_record["q_packed_uint8"][0],
        k_record["s_col_K"][0],
        k_record["zp_K"][0],
        k_record["s_row_K"][0],
        group=group,
        bits=4,
    )
    value_dequant_rotated = decode.kvarn_dequant_tile_v(
        v_record["q_packed_uint8"][0],
        v_record["s_col_V"][0],
        v_record["s_row_V"][0],
        v_record["zp_V"][0],
        head_dim=dimension,
        bits=2,
    )
    key_dequant = key_dequant_rotated.transpose(0, 1) @ hadamard
    value_dequant = value_dequant_rotated @ hadamard

    payload = {
        "schemaVersion": 1,
        "source": {
            "repository": "https://github.com/huawei-csl/KVarN",
            "commit": commit,
            "torchVersion": torch.__version__,
            "generator": {
                "path": "spike/scripts/generate_kvarn_reference_fixture.py",
                "sha256": sha256(Path(__file__).resolve()),
            },
            "files": {
                "vllm/model_executor/layers/quantization/kvarn/sinkhorn.py": sha256(
                    sinkhorn_path
                ),
                "vllm/v1/attention/ops/kvarn_store.py": sha256(store_path),
                "vllm/v1/attention/ops/kvarn_decode.py": sha256(decode_path),
            },
        },
        "config": {
            "headDimension": dimension,
            "groupSize": group,
            "keyBits": 4,
            "valueBits": 2,
            "iterations": iterations,
            "rtnQuantile": 0.0,
        },
        "input": {"keysTokenMajor": floats(keys), "valuesTokenMajor": floats(values)},
        "expected": {
            "keyPacked": k_record["q_packed_uint8"].reshape(-1).tolist(),
            "keyAbsorbedScaleFP16Bits": fp16_bits(k_record["s_col_K"]),
            "keyAbsorbedBiasFP16Bits": fp16_bits(k_record["zp_K"]),
            "keyTokenScaleFP16Bits": fp16_bits(k_record["s_row_K"]),
            "valuePacked": v_record["q_packed_uint8"].reshape(-1).tolist(),
            "valueChannelScaleFP16Bits": fp16_bits(v_record["s_col_V"]),
            "valueAbsorbedScaleFP16Bits": fp16_bits(v_record["s_row_V"]),
            "valueAbsorbedBiasFP16Bits": fp16_bits(v_record["zp_V"]),
            "keysDequantizedTokenMajor": floats(key_dequant),
            "valuesDequantizedTokenMajor": floats(value_dequant),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded)
    print(f"wrote {args.output} sha256={hashlib.sha256(encoded.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
