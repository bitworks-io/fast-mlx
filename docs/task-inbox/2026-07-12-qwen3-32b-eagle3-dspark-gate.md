---
status: captured
type: performance-spike
priority: highest
created: 2026-07-12
source: sol-audit
planning_ready: true
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

Write the architecture/measurement plan, inventory staged target/draft checkpoints on the
bench Mac, and run the external Python preflight before touching the Swift compiled path.
