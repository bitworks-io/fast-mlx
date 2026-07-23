# Llama-3.3-70B source lock — ACCEPT admission, no runtime claim

- **Date:** 2026-07-23
- **Lane:** source admission; non-promotable
- **Engine source:** `dcfbbe39c1ee3ee5e9119820ec67994e196968c0` (clean)
- **Model:** `mlx-community/Llama-3.3-70B-Instruct-4bit`
- **Revision:** `de2dfaf56839b7d0e834157d2401dee02726874d`
- **Decision:** **ACCEPT this local model/tokenizer snapshot as the exact second-family source
  boundary. Make no runtime, quality, speed, capacity, tier, or broad-support claim.**

## Operator story and acceptance

A long-context Apple-Silicon operator needs the materially different model-family qualification to
use a complete, immutable checkpoint and tokenizer rather than a mutable cache alias.

The one-shot admission at
`/Users/llmbench/perf-work/results/fused-compressed-kv-llama3-70b-source-dcfbbe3/source-lock-v1`
authenticated the pinned public repository response, cached revision/tree/per-file metadata, exact
file set, eight LFS shard hashes, Git blob identities, checkpoint index, tokenizer manifest, and
fast-mlx production provenance digests. Unsupported geometry, partial files, symlinks, mutation,
hash mismatch, interruption, or pre-existing output fails closed.

| Artifact | SHA-256 |
| --- | --- |
| Reviewed authenticator | `b14fa61b147b611c8abd76b02635a29f3d571462295c57a7a8125fe27813bd37` |
| Reviewed launcher | `5b8859cc8f1a95d17692d17f49e0ad5ba7ca82e38abcf668a87a07481adf21c6` |
| Official API response | `9242217f955acd490e882b04702b612f63a33fdbac43b071c2baf132b0394459` |
| Source-lock receipt | `145127546c6c9872e80512716494eed77905d6e3ddd398c47c8f34a5ec796a4f` |
| Completion | `f2e9082ac048f36157d1e5d05ca7c513fab9b8cad4dc1fe681e5d76b58eaf7af` |
| Snapshot manifest | `51d9a8e2c0514bac96781849482985d7cddc16a61c4720588043382378bb5cbb` |

The receipt binds 15 files, eight weight shards, 39,706,010,909 bytes, Llama Q64/KV8/D128 with 80
layers and maximum context 131,072, 4-bit group-64 weights, and Llama-3 RoPE. Checkpoint-content
SHA-256 `5083c6af…be1400` and tokenizer SHA-256 `da67fb22…be58ab` match the prior phase-0 identities.
Independent receipt/API/tree validation passed; stderr was empty; no lock or process survived.

## Runtime follow-through

The model-scoped runtime cycle is now recorded separately in
[`2026-07-23-llama3-70b-runtime.md`](2026-07-23-llama3-70b-runtime.md). The source admission remains
immutable: it does not itself authorize the retained runtime, quality, capacity, or task claims.
The runtime verdict binds this receipt, keeps the Qwen KVTuner schedule unavailable, and closes
without a Llama speed tier or broad/default claim.
