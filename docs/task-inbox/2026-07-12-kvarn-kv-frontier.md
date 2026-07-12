---
status: captured
type: quantization-spike
priority: high
created: 2026-07-12
source: sol-audit
planning_ready: true
implementation_ready: false
---

# KVarN K4V2 and asymmetric mixed-precision KV frontier

## Raw Capture

Evaluate KVarN's `K4V2-g128` configuration as the strongest new KV candidate, but make it
earn the Apple path against ordinary affine K4V2/K8V2, per-layer KVTuner schedules, 4-bit
affine, and any matching clean-SHA evidence produced by the separately owned TurboQuant-B
closure, all at equal effective bytes.

## Planning Seed

User/operator: a dial user choosing more context/concurrency for a quantified quality cost.

Desired outcome: a measured Apple-Silicon Pareto frontier for asymmetric key/value precision,
with one storage winner selected before custom fused-kernel investment.

Acceptance signals:

- exact KVarN tile transform reproduces the paper/reference implementation on fixtures;
- actual bytes/token include all scale, codebook, alignment, recent-window, and workspace data;
- 24K teacher-forced KL, perplexity, and tail-p95 remain the precision-loss contract;
- math/code/tool and long-retrieval task scores test autoregressive error accumulation without
  replacing teacher-forced precision metrics with free-running drift;
- batch-1 prefill/decode, peak unified memory, and capacity are reported;
- the first matrix includes affine K4V2/K8V2 at group 64/128, KVTuner schedules, and KVarN
  K4V2-g128; the separate TurboQuant-B task owns outlier, boundary, and QJL ablations, whose
  clean-SHA rows may be imported only when conditions match.

Known failure cases: CUDA/Triton wins do not survive Metal; partial tiles or recent tokens
erase capacity; key precision is too low for Qwen; normalization overhead dominates; a nominal
bit label hides metadata; task gains fail to generalize beyond calibration prompts.

## Sources

- [KVarN paper](https://arxiv.org/abs/2606.03458) and [official implementation](https://github.com/huawei-csl/KVarN)
- [KVTuner paper](https://arxiv.org/abs/2502.04420) and [official implementation](https://github.com/cmd2001/KVTuner)
- [KIVI baseline](https://arxiv.org/abs/2402.02750)

## Next Step

Implement the pure/reference tile transform and offline per-layer configuration artifact;
do not begin Metal work until the quality/size matrix selects a winner.
