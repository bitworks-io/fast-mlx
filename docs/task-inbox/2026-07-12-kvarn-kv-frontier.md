---
status: complete
type: quantization-spike
priority: high
created: 2026-07-12
source: sol-audit
planning_ready: false
implementation_ready: false
completed: 2026-07-18
---

# KVarN K4V2 and asymmetric mixed-precision KV frontier

## Completion

Completed 2026-07-18. Dated verdict:
[`docs/superpowers/verdicts/2026-07-18-kvarn-kv-frontier.md`](../superpowers/verdicts/2026-07-18-kvarn-kv-frontier.md);
content piece:
[`docs/content/2026-07-18-when-smaller-kv-is-not-faster.md`](../content/2026-07-18-when-smaller-kv-is-not-faster.md).

Decision: promote fp16 KV as the Transparent baseline, affine K4V2-g64 as Balanced capacity,
frozen KVTuner as explicit Max-fit capacity, and KVarN i8 as capacity-only Max-fit plus fused
compressed-domain attention candidate. Shelve affine K8 controls and KVarN i16; reject the g128
4-bit affine cells that crossed the hard floor.

Boundary: this is Qwen3-32B-4bit, same-weights, M5 Max evidence only. It selected useful
capacity tiers and storage primitives; it did not prove a speed win or broad model support.

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
bit label hides metadata; task gains fail to generalize beyond calibration prompts; later batch
compaction overwrites surviving quantized KV state. That last case is a correctness failure, not
loss that may be exposed through the dial.

## Sources

- [KVarN paper](https://arxiv.org/abs/2606.03458) and [official implementation](https://github.com/huawei-csl/KVarN)
- [KVTuner paper](https://arxiv.org/abs/2502.04420) and [official implementation](https://github.com/cmd2001/KVTuner)
- [KIVI baseline](https://arxiv.org/abs/2402.02750)

## Next Step

Closed. Continue with the selected follow-on gate:
[`2026-07-12-fused-compressed-kv-attention.md`](2026-07-12-fused-compressed-kv-attention.md).
Carry forward KVarN i8 plus shared affine/KVTuner storage primitives and require direct 32K/128K
end-to-end proof before any speed-tier claim.
