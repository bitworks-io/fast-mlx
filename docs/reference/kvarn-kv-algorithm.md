# KVarN / asymmetric KV-cache algorithm lock

**Reviewed:** 2026-07-14

**Purpose:** implementation and measurement input for the KVarN storage-quality gate.

**Primary sources:** [KVarN arXiv v1](https://arxiv.org/abs/2606.03458), the
[official KVarN repository](https://github.com/huawei-csl/KVarN) pinned at
[`7586257f1c632e63187bfacbbe21ccb51540f7b3`](https://github.com/huawei-csl/KVarN/tree/7586257f1c632e63187bfacbbe21ccb51540f7b3),
[KVTuner paper](https://arxiv.org/abs/2502.04420), and the
[official KVTuner repository](https://github.com/cmd2001/KVTuner) pinned at
[`96dd05eb2fe350c72c1a3dfdca04e878506f7c17`](https://github.com/cmd2001/KVTuner/tree/96dd05eb2fe350c72c1a3dfdca04e878506f7c17).

This file distinguishes the peer-reviewed/preprint method from later repository behavior. A
repository performance claim is useful prioritization evidence, but it is not a fast-mlx result
and is never copied into a dial tier without a clean-SHA Apple-Silicon measurement.

## Source-review findings

| Claim | Finding | Implementation consequence |
|---|---|---|
| KVarN rotates K and V in the channel dimension with an orthonormal Hadamard transform, then alternates row/column variance normalization before asymmetric round-to-nearest quantization. | **CONFIRMED** by arXiv v1 sections 2.3 and 3.3 plus Appendix H; the pinned repository contains the corresponding pure PyTorch reference in `sinkhorn.py` and `kvarn_store.py`. | The Swift reference must implement all three stages. Hadamard-only or affine-only controls are not KVarN. |
| K uses the channel-by-token orientation and V uses token-by-channel; the ordinary RTN scale is absorbed into one variance-normalization axis while the other scale remains explicit. | **CONFIRMED** by the paper's KIVI-oriented description and explicitly specified by the pinned repository. | K and V need different row orientations and metadata layouts. Treating both as per-token affine is a control, not KVarN. |
| The paper establishes KVarN quality at `K2V2`, about 2.3 effective bits per element including metadata, on Qwen3-4B, Llama-3.1-8B, and Phi-4-family models. | **CONFIRMED** by arXiv v1 tables 1–3 and experimental details. | These results motivate the technique but do not predict Qwen3-32B K4V2 quality on Apple Silicon. |
| `kvarn_k4v2_g128` is the paper's evaluated 2.3-bit configuration. | **CONTRADICTED.** The pinned repository's later preset is 4-bit K, 2-bit V, 128-token tiles. Its dense D=128 layout is 3.375 effective bits per K/V element before fixed sink/tail/workspace costs. | Never label K4V2 as the paper's 2.3-bit row. Record raw K/V bits, effective packed bits, and fixed overhead separately. |
| KVarN K4V2 gives Qwen3-32B fp16-level AIME25 quality, about 4x capacity, and fp16-or-better throughput. | **UNVERIFIED as a reproducible result.** This is a post-paper official-repository claim. It is not in arXiv v1, and the repository presentation does not provide the raw logs, full command/provenance packet, seeds, or an Apple result needed by this project. | It justifies running the gate. It cannot appear as fast-mlx evidence or a promotion premise. |
| Eight variance-normalization iterations are equivalent to sixteen for fast-mlx's target. | **UNVERIFIED for fast-mlx.** The pinned repository defaults its runnable preset to 8 and describes it as converged, while the paper/reference default is 16. | The pure fixture pins explicit iteration counts. The first matrix measures 8 and retains a 16-iteration audit cell; no silent default. |
| A 128-token fp16 sink and one incomplete fp16 tail are free or negligible. | **CONTRADICTED as an accounting assumption.** The pinned backend keeps scheduler block zero unquantized and maintains a fixed tail pool/workspace for partial tiles and serving concurrency. At the pinned revision, `KVARN_SINK_TOKENS` is parsed but the backend always identifies the first block as the sink. | The fast-mlx gate pins one full-group sink. Capacity includes sink, tail, alignment, and measured workspace; an apparently configurable upstream token count is not copied into the product contract. |
| The upstream vLLM allocation formula is the KVarN format's portable byte cost. | **CONTRADICTED.** The pinned CUDA backend reserves a compressed record for every scheduler block (including blocks currently resident in its fp16 pool), sizes that pool with request, prefill, and headroom slots, and uses extra per-token power-of-two padding at `D>=256`. Those are backend allocator choices, not algorithm payload. | The pure accountant models fast-mlx's explicitly documented tight sequential arrays. It reports upstream-layout comparisons separately, accepts caller-supplied local workspace/alignment, and must reconcile every prediction with actual MLX-array `nbytes` before measurement. |
| KVTuner is a runtime quantizer. | **CONTRADICTED.** Its core contribution is an offline sensitivity search that selects per-layer K/V bit widths; the runtime consumes the frozen schedule using an underlying quantizer. | KVTuner belongs in the control plane. A schedule artifact must pin model/config, calibration corpus, seed, objective, and every layer's K/V bits. |
| KVTuner supports asymmetric per-token and KIVI-oriented controls with independent K/V widths. | **CONFIRMED** by the paper and pinned repository configuration surface (`nbits_key`, `nbits_value`, `axis_key`, `axis_value`, `asym`, `group_size`, and residual length). | The first matrix includes a simple per-token affine control; KIVI orientation is separately named when used. |

No external claim above is promoted without local measurement. The paper's ~50-GPU-day full
benchmark reproduction is explicitly out of scope; fast-mlx uses its hardened, bounded harness.

## Algorithm locked for the pure reference

For each full token tile and KV head:

1. Build the normalized Sylvester Hadamard matrix `H / sqrt(D)` and rotate every token along
   `head_dim`. The transform is self-inverse.
2. Orient K as `[D, G]` and V as `[G, D]`, where `G` is the token tile size.
3. In float32, initialize log column/row scales to zero. Alternately update the column and row
   log scales from the unbiased sample standard deviation, clamped to `[1e-3, 1e3]`; clamp log
   scales to `[-0.3, 10]`. Track and return the lowest-imbalance state, where imbalance is the
   sum of max/min column-std and row-std ratios.
4. Apply unsigned asymmetric RTN per row into `[0, 2^bits - 1]` using that row's minimum and
   range. K rows are channels; V rows are tokens.
5. Absorb the RTN scale and bias into K's channel scale or V's token scale, cast metadata to
   fp16, and pack low-bit values least-significant group first. A 4-bit pair stores the even
   value in the low nibble; 2-bit values pack four per byte.
6. Reconstruct in the rotated frame with
   `K=(q*absorbedChannelScale+absorbedChannelBias)*tokenScale` and
   `V=(q*absorbedTokenScale+absorbedTokenBias)*channelScale`, then apply the inverse Hadamard.

The compact scalar reference fixture is generated from a clean checkout of the pinned Apache-2.0
PyTorch implementation using a deterministic finite input, Torch `2.11.0`, one CPU thread,
deterministic algorithms, 16 balancing iterations, and the optional RTN-quantile ablation pinned
off. It records and tests the upstream file hashes plus the generator hash. The Swift gate requires
byte-identical packed codes and fp16 metadata, not merely similar reconstruction error. Non-finite
inputs, intermediate overflow, unsupported bits, non-power-of-two head dimensions, impossible
Hadamard allocations, and incomplete tiles fail closed. This small `D=G=4` fixture locks scalar
semantics; Phase 3 separately owns `D=G=128`, 8-versus-16 iterations, fp16 MLX rotation, and
runtime-array conformance.

## Dense D=128, G=128 K4V2 layout

The pinned repository defines one record per full tile and KV head:

| Component | Bytes |
|---|---:|
| K payload, `128 × 128 × 4 / 8` | 8,192 |
| K fp16 metadata, `2D + G` values | 768 |
| V payload, `128 × 128 × 2 / 8` | 4,096 |
| V fp16 metadata, `D + 2G` values | 768 |
| **Full tile/head** | **13,824** |
| **Amortized full-tile/head/token** | **108** |
| **Effective bits per K/V element** | **3.375** |

The payload-only average is 3 bits/element; metadata raises it to 3.375. Fully quantized fp16 K/V
would be 512 bytes per head/token, so the ideal full-tile ratio is 4.741x. That is not the
end-to-end capacity ratio: the first sink tile, the incomplete tail, local allocation alignment,
allocator residency, and decode workspace remain separate measured terms. The pinned CUDA
backend's additional block records, pool headroom, and `D>=256` padding are not silently attributed
to the tight Apple layout; each backend reports its own actual bytes.

## Controls and evidence boundary

- **Same-weights reference:** the candidate checkpoint with fp16 KV. A bf16-weight checkpoint is
  a different weight-quantization comparison and cannot supply the affine/KVarN marginal loss.
- **Affine controls:** independent K/V widths, asymmetric per-token grouping at 64 and 128, with
  native packed arrays plus every scale/bias byte counted from actual MLX arrays. KIVI-oriented
  controls are labeled separately.
- **KVTuner controls:** schedules are selected on a calibration split and frozen before the
  measurement corpus is scored. Evaluation prompts may not leak into the search.
- **Quality:** KL, perplexity, top-1 agreement, and 24K tail-p95 are teacher-forced. Free-running
  math/code/tool/retrieval checks are task/coherence evidence, not substitutes for context-locked
  loss.
- **Runtime:** the correctness-first path may materialize full K/V before attention. Its measured
  speed is reported honestly, but fused compressed-domain attention remains a separate gate.
