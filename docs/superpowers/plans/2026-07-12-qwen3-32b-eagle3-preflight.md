# Qwen3-32B EAGLE-3 trained-speculator gate

- **Status:** COMPLETE — Phase 0 RED; compiled Swift work shelved
- **Date:** 2026-07-12
- **Owner:** Codex
- **Queue seed:** [`docs/task-inbox/2026-07-12-qwen3-32b-eagle3-dspark-gate.md`](../../task-inbox/2026-07-12-qwen3-32b-eagle3-dspark-gate.md)

## User story

As the fast-mlx owner choosing the first trained speculative decoder to port, I need a
clean-SHA, same-target Apple-Silicon verdict for the public Qwen3-32B EAGLE-3 checkpoint so
that I can invest in a production Swift path only when it is byte-exact and economically
positive for the actual target/checkpoint pair.

## Acceptance criteria and proof

1. **Accounting is unambiguous.** Evidence reports proposed drafts, accepted drafts, verify
   rounds, accepted drafts per verify round, proposal acceptance rate, and inclusive acceptance
   length as separate fields. A pure HarnessCore test pins
   `inclusiveAcceptanceLength = 1 + acceptedDraftsPerRound` and pairing-specific break-even.
2. **The ported head is faithful.** The preflight validates the checkpoint SHA, config, tensor
   names, shapes, and dtypes, then compares fixed-input MLX logits with the current authoritative
   PyTorch/speculators implementation. Cosine similarity must exceed `0.99` before generation
   results are interpreted.
3. **Greedy output is exact and engaged.** For each implemented verify shape, EAGLE-on emits the
   same token IDs and decoded bytes as the same target's base loop at temperature zero. At least
   one verify round and one accepted draft are required; a silent fallback is a failure.
4. **Economics are measured per pairing.** The runner records draft, target verify,
   rollback/commit, and total decode time. Qwen3-32B-4bit and Qwen3-32B-8bit are each compared
   with their own same-process base loop; no archived 8B threshold is reused. A BF16 target is
   a conditional hidden-state-precision diagnostic because inventory found it is not staged.
5. **A losing arm fails closed.** Checkpoint/target mismatch, missing tap state, zero engagement,
   long-output state drift, or negative net throughput must not produce a promotion claim. A
   production yield gate is designed from the measured break-even only after Phase 0 clears.

Happy path: the verified head produces exact tokens, sustains healthy accepted drafts per round,
and beats the same Qwen3-32B target baseline on at least one realistic workload. Failure/recovery
path: the preflight stops at the first failed fidelity/exactness gate, or records an honest
pairing-specific negative economic verdict without modifying the compiled engine.

## Source-confirmed architecture

The checkpoint is
[`RedHatAI/Qwen3-32B-speculator.eagle3`](https://huggingface.co/RedHatAI/Qwen3-32B-speculator.eagle3)
at repository revision `dc84fe7ff1db31efa824776f49c141fc8195eb47`. The 2026-07-12 manifest
inspection resolves the archived plan's unknowns:

- one BF16 draft decoder layer and 16 tensors total;
- `hidden_size=5120`, `intermediate_size=25600`, `head_dim=128`, 64 query heads, 8 KV heads;
- `fc.weight [5120,15360]`, Q/K/V input width 10240, draft vocabulary 32000, target
  vocabulary 151936;
- `d2t [32000]` is additive (`target = draft + d2t[draft]`); `t2d [151936]` is a target-token
  availability mask;
- the serialized draft layer is deliberately **Llama-style** (`model_type=llama`) and has no
  Q/K-normalization tensors, even though the verifier is Qwen3-32B. Do not synthesize Q/K norm.

Current `speculators` resolves omitted auxiliary tap IDs as `{2, N/2, N-3}`. For a 64-layer
Qwen3-32B target that is `{2,32,61}`. Current vLLM records index 0 at the embedding and index
`i+1` after target layer `i`; mlx-lm's pre-layer activation at index `i` is therefore the same
state. The preflight captures pre-layer activations 2, 32, and 61, before the final norm.

One draft step is:

1. concatenate the three target activations and project `3H -> H` through `fc`;
2. embed the current target token with the head's full-vocabulary embedding;
3. separately normalize the embedding and fused hidden state, concatenate to `2H`, then run
   the one-layer Llama-style attention/MLP block with `rope_theta=1_000_000`;
4. normalize, project to the 32K draft vocabulary, greedily select, and apply `d2t`;
5. verify the chain against the target, accept only the longest matching prefix, emit the
   target correction/bonus, and roll both target and draft state back to committed history.

Tree verification remains out of scope for this gate. Chain depths `k=1` and `k=3` isolate head
fidelity from the higher-sync prototype path.

## Metric contract

For counters `A = accepted draft tokens`, `P = proposed draft tokens`, and `R = target verify
rounds`:

- proposal acceptance rate = `A / P` (nil when `P=0`);
- accepted drafts per round = `A / R` (nil when `R=0`);
- inclusive acceptance length = `1 + A / R`;
- emitted tokens per speculative round = `1 + A / R` for greedy chain verification.

For measured mean baseline token time `T_base` and speculative phase times
`T_draft + T_verify + T_commit`:

- round-cost ratio = `(T_draft + T_verify + T_commit) / T_base`;
- predicted speedup = `(1 + A/R) / round-cost-ratio`;
- pairing-specific break-even accepted drafts per round = `round-cost-ratio - 1`.

The observed end-to-end decode rate is the promotion authority. The phase model is diagnostic,
not permission to substitute a projection for a measured win.

## Controls and scope

- **DFlash:** current public/MLX checkpoints cover Qwen3-8B and newer Qwen3.5/3.6 targets, not
  Qwen3-32B. It remains a same-target 8B method control in a separate cycle; cross-model raw
  tok/s is invalid.
- **Native MTP:** the pinned `mlx-swift-lm` revision contains generic speculation plus a Gemma4
  shared-KV MTP iterator. fast-mlx does not route it, and there is no compatible Qwen3-32B MTP
  checkpoint. It is an architecture/control reference, not an executable arm here.
- **Out:** temperature above zero, tree attention, serving API wiring, continuous batching, and
  quantizing the draft head before the BF16 fidelity baseline is established.

## Execution plan

### Phase 0A — accounting TDD

1. Add pure `SpeculativeAcceptanceSummary` and `SpeculativeEconomics` types to HarnessCore.
2. Write failing tests for the inclusive model-card conversion, zero-denominator behavior,
   per-pair break-even, and the preserved Qwen3-8B/DSpark example.
3. Implement the smallest pure logic and run `swift test --filter HarnessCoreTests` off-box.

### Phase 0B — reproducible external preflight

1. Add `experiments/eagle3/` with a pinned source/checkpoint manifest and no model artifacts.
2. Implement a checkpoint inspector/validator that can read the safetensors header without
   materializing tensors and rejects schema drift.
3. Implement the MLX Llama-style EAGLE head with direct safetensors loading. Unit-test pure
   config, token remapping, acceptance, counter, and evidence serialization behavior.
4. Generate a fixed-seed reference fixture with current `vllm-project/speculators`, then run
   the MLX parity checker. Persist only compact fixture metadata/results, not weights or large
   arrays.
5. Implement the same-process base and EAGLE chain loops with exact token comparison, explicit
   cache rollback, EOS/budget trimming, and phase timers. Emit JSONL/CSV evidence with model and
   source SHAs. Set MLX's allocator cache limit explicitly (`mx.set_cache_limit(8 << 30)`) before
   model loading because the bench host's wired-memory ceiling is raised.

### Phase 0C — bench gate

1. Push the tree with `spike/scripts/sync_llmbench.sh`; do not use `swift test` for any target
   importing MLX.
2. Run checkpoint validation, pure Python tests, PyTorch-to-MLX parity, a short exactness smoke,
   then a multi-round/long-output exactness run.
3. Measure coding, low-repetition prose, and reasoning/agentic shapes at `k=1` and `k=3`, first
   on Qwen3-32B-4bit, then on the staged 8-bit target. Downloading/staging BF16 is justified only
   if parity passes and quantized-target acceptance is low enough that precision is the leading
   remaining hypothesis. Use warmups and at least three measured repetitions for a promotion
   claim.
4. Apply the gate:
   - **GREEN:** exact, engaged, parity-passing, and at least `1.30x` measured on one realistic
     workload without a material losing-shape regression after an economic gate;
   - **AMBER:** exact and economically positive (`1.00x..1.30x`) with healthy acceptance;
   - **RED:** parity/exactness failure, persistently low acceptance after fidelity is proven, or
     throughput `<=1.00x`.

### Phase 1 — conditional compiled path

Only a Green result (or an explicit owner decision after Amber) makes the Swift engine work
implementation-ready. Re-plan from the measured constants. The production design must preserve
actor confinement, reuse `SpecAccept`/`SpecEmit` and target-cache truncation, give the head its
own rollback-safe cache, expose engagement plus phase-cost evidence, and pass byte-identity in
every compiled/uncompiled verify variant before benchmarking.

## Completion artifacts

- dated Phase 0 verdict with manifest, parity, exactness, acceptance, phase costs, and same-target
  throughput tables;
- compact evidence in `docs/superpowers/verdicts/` and the durable verification index;
- a `docs/content/` piece explaining the inclusive-acceptance trap and Apple round economics;
- updated task inbox and handoff with the next safe action, whether promoted or shelved.

## Execution outcome — 2026-07-12

Phase 0A and 0B completed. The authenticated checkpoint passed, and the MLX head matched the
pinned PyTorch/speculators fixture at cosine `0.9999815822` with 100% argmax agreement. Phase
0C stopped at its first hard gate: Qwen3-32B-4bit changed the greedy stream at generated index
17 and Qwen3-32B-8bit changed it at index 7, both at `k=1` with non-vacuous engagement. The 8-bit
diagnostic shows the same sequential prefix selecting token `279` with a one-token target
probe and `264` with a multi-token probe. The 4-bit diagnostic isolates drift to histories
that processed and rolled back rejected future tokens.

Per the planned RED rule, the multi-shape throughput table, `k=3`, BF16 download, and Phase 1
Swift port were not run. Apparent rates from the failing verifies are explicitly
non-authoritative because the outputs differ. Final verdict:
[`2026-07-12-qwen3-32b-eagle3-preflight.md`](../verdicts/2026-07-12-qwen3-32b-eagle3-preflight.md).
