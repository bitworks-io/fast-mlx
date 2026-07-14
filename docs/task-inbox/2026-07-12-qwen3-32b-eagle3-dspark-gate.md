---
status: partial-eagle-red-controls-blocked
type: performance-spike
priority: highest
created: 2026-07-12
source: sol-audit
planning_ready: false
implementation_ready: false
---

# Qwen3-32B EAGLE-3/DSpark gate, with DFlash and native-MTP controls

## Raw Capture

Execute the committed trained-speculator cycle on Qwen3-32B. Use the existing PLD
propose/verify/accept/rollback framework, but make the bench—not a model-card acceptance
headline—decide whether EAGLE-3/DSpark pays on Apple Silicon. Include DFlash and the pinned
Swift dependency's native-MTP path as explicit reference/control implementations where
matching checkpoints make a valid comparison possible.

## Planning Seed

User/operator: the fast-mlx owner choosing the first trained speculative decoder to port.

Desired outcome: a clean-SHA, same-baseline economic verdict for the Qwen3-32B EAGLE-3
checkpoint. DFlash may displace a full port only after an apples-to-apples result; native MTP
is a lower-port-cost control, not an “already-have” feature.

Acceptance signals:

- temperature-0 output is byte-identical to base decode for every implemented verify path;
- engagement and accepted tokens/round are measured with the correct denominator;
- evidence splits draft cost, target verify cost, rollback/commit cost, and net tok/s;
- the target workload clears its base-loop throughput, while an economic yield gate keeps
  losing pairs neutral;
- same-target Qwen3-8B DSpark/DFlash can compare method economics, but product-scale 27B/32B
  results are each normalized to their own target baseline—raw cross-model tok/s is invalid.

Known failure cases: accepted drafts below the **pairing-specific** measured break-even (the
old ~2.3 value applies only to Qwen3-8B/DSpark); confusing inclusive `acceptance_length` with
accepted draft tokens; a verify kernel that changes the greedy byte stream;
hidden-state/checkpoint mismatch; recurrent-state rollback drift; draft memory that violates
capacity policy; long-output acceptance collapse.

## Sources / existing foundation

- [Qwen3-32B EAGLE-3 checkpoint](https://huggingface.co/RedHatAI/Qwen3-32B-speculator.eagle3)
- [Official EAGLE implementation](https://github.com/SafeAILab/EAGLE)
- [DFlash paper](https://arxiv.org/abs/2602.06036) and [Apple/MLX port](https://github.com/bstnxbt/dflash-mlx)
- [`HarnessCore/SpecDecode/`](../../spike/Sources/HarnessCore/SpecDecode/)
- [Preserved DSpark investigation](../reference/mlx-serve-archive/dspark-acceptance-investigation.md)

## Next Step

**Resolved 2026-07-12:** the authenticated EAGLE-3 Phase 0 passed checkpoint and head parity,
then failed byte identity on both Qwen3-32B-4bit (first mismatch index 17) and 8-bit (index 7).
The 8-bit target selects different greedy tokens for one-token and multi-token probes from the
same sequential prefix; the 4-bit replay isolates drift to full verify+rollback histories.
Verdict: [`2026-07-12-qwen3-32b-eagle3-preflight.md`](../superpowers/verdicts/2026-07-12-qwen3-32b-eagle3-preflight.md).

EAGLE-3 is shelved and no Swift port begins. DSpark, DFlash, and native MTP were not executable
same-target Qwen3-32B controls: no compatible product-size checkpoint is staged or available
in the pinned paths. This records **no negative verdict** for those untested methods. Reopen
trained speculation only for a deterministic target-verify repair that still pays, or a
compatible product-size checkpoint. The next actionable flywheel item is continuous batching
plus decode-first chunked prefill.

**Subsequent status (2026-07-14):** continuous batching completed its exact engine/policy gate;
the live queue in `docs/agent-handoff.md` now advances to KVarN/asymmetric affine.
