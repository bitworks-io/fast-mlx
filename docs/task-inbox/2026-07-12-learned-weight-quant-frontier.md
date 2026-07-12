---
status: captured
type: quantization-benchmark
priority: medium-high
created: 2026-07-12
source: sol-audit
planning_ready: true
implementation_ready: false
---

# Learned and mixed-precision weight-quant frontier

## Raw Capture

Replace the vague oQ4e lead with a reproducible offline checkpoint sweep: standard affine,
official MLX-LM dynamic quantization, optional DWQ refinement, and oMLX oQ4e. Keep the Swift
decode loop unchanged by producing ordinary MLX safetensors and let the hardened harness rank
quality, size, and speed.

## Planning Seed

User/operator: a dial user choosing model footprint and decode speed with measured quality
loss, especially on low-memory Macs.

Desired outcome: a provenance-stamped per-model weight-quant Pareto frontier rather than a
vendor preset name.

Acceptance signals:

- calibration source, sample IDs/hashes, tool versions, seeds, bits/group sizes, and per-layer
  policy are recorded without committing private data;
- output checkpoints load through the unmodified Swift engine;
- size, peak memory, prefill/decode, teacher-forced KL/ppl/tail-p95, and task deltas are
  measured against the same bf16 reference;
- the first bounded order is affine → dynamic mixed precision → dynamic+DWQ; oQ4e is a
  comparison arm, not the source of truth;
- a checkpoint below the coherence/garbage floor is never offered, however small.

Known failure cases: calibration leakage, MoE experts never exercised by calibration,
conversion memory exceeding the bench box, nominal bits hiding mixed layer precision, and a
checkpoint whose storage shrinks without improving Apple decode.

## Sources

- [Official MLX-LM learned quantization guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md)
- [oQ quantization documentation](https://github.com/jundot/omlx/blob/main/docs/oQ_Quantization.md)

## Next Step

Choose one staged production-size checkpoint and a trusted calibration corpus, then write the
conversion/evidence manifest before generating any weights.
