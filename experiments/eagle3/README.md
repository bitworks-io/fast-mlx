# Qwen3-32B EAGLE-3 Phase 0

This directory is the reproducible, non-shipping gate for the public
`RedHatAI/Qwen3-32B-speculator.eagle3` checkpoint. It answers three questions in order:

1. Is the downloaded checkpoint exactly the reviewed checkpoint?
2. Does the MLX head reproduce the authoritative PyTorch/speculators head math?
3. Does greedy chain speculation stay byte-identical and pay for this target/checkpoint pair on
   Apple Silicon?

No weights, fixtures, or result artifacts belong in Git. `pins.json` records the reviewed source
and checkpoint revisions; generated evidence goes under the ignored `artifacts/` directory.

## Environments

The reference fixture and MLX check intentionally use separate **Python 3.10+** environments
(the recorded bench uses Python 3.13). The reference environment is pinned by
`reference-requirements.txt` and never loads or downloads verifier weights. The MLX environment
needs MLX 0.32.0 and mlx-lm 0.29.1 for the current bench cycle.

Installing the reference requirements downloads third-party packages and executes their import
code; use a dedicated virtual environment and review the pins before installing. All gate scripts
force Hugging Face and Transformers offline mode and require existing local model directories, so
a misspelled path cannot start an implicit model download.

Promotion evidence requires a full 40-character clean harness SHA. `--allow-dirty` exists only for
explicitly non-promotable development diagnostics. The checkpoint inspector authenticates the
actual config and 3.1 GB weight bytes with SHA-256, in addition to validating revision metadata,
file size, tensor schema, and every tensor payload range. The two target pairings are also pinned
by config, revision, exact shard set, shard sizes, and full shard SHA-256; authentication therefore
adds a deliberate sequential read of 17 GB (4-bit) or 32 GB (8-bit) before a gate run.

## Gate sequence

Set portable paths first:

```bash
export HEAD_DIR=/path/to/RedHatAI-Qwen3-32B-speculator.eagle3
export MODEL_DIR=/path/to/Qwen3-32B-4bit
export HARNESS_SHA_FILE=/path/to/synced-fast-mlx/.harness-sha
```

Run the MLX-free checks and checkpoint validation:

```bash
python -m unittest test_preflight_core.py
python inspect_checkpoint.py \
  --head "$HEAD_DIR" \
  --harness-sha-file "$HARNESS_SHA_FILE" \
  --output artifacts/checkpoint.json
```

Generate the authoritative fixture in the pinned reference environment, then check MLX parity:

```bash
reference-python dump_reference_fixture.py \
  --head "$HEAD_DIR" \
  --harness-sha-file "$HARNESS_SHA_FILE" \
  --output artifacts/reference-fixture.safetensors

mlx-python check_head_parity.py \
  --head "$HEAD_DIR" \
  --fixture artifacts/reference-fixture.safetensors \
  --harness-sha-file "$HARNESS_SHA_FILE" \
  --output artifacts/parity.json
```

Parity must exceed cosine `0.99` before generation is meaningful. Then run exactness before the
bench:

```bash
mlx-python run_preflight.py verify \
  --model "$MODEL_DIR" \
  --head "$HEAD_DIR" \
  --harness-sha-file "$HARNESS_SHA_FILE" \
  --prompt "Write a Python function that reverses a linked list." \
  --max-tokens 64 \
  --num-draft 3 \
  --synchronize-phases \
  --output artifacts/verify.json

mlx-python run_preflight.py bench \
  --model "$MODEL_DIR" \
  --head "$HEAD_DIR" \
  --harness-sha-file "$HARNESS_SHA_FILE" \
  --max-tokens 256 \
  --num-draft 1 3 \
  --warmup 1 \
  --runs 3 \
  --output artifacts/bench.json \
  --csv-output artifacts/bench.csv
```

The bench alternates base/speculative run order, asserts token and byte identity on every pair,
and records separate synchronized draft, target-verify, and commit/rollback diagnostics. A failed
verify also replays the first mismatch with full-verify, retained-only, and all-sequential target
cache histories so rollback corruption can be distinguished from shape-dependent cached-state
drift. The unsynchronized end-to-end decode rate—not the diagnostic projection—is the promotion
authority. A failed fidelity or exactness gate stops the cycle before throughput benchmarking.

JSON and CSV benchmark views share a content-derived `evidence_id`; detached CSV rows also carry
the harness, runtime, target, draft, and parameter provenance needed to attribute them.

The runner sets `mx.set_cache_limit(8 << 30)` before model loading because the bench host uses an
explicitly raised `iogpu.wired_limit_mb` ceiling.
