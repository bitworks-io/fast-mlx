# Prompt-Lookup Decoding (PLD) first run — the spec-decode framework's first drafter

**Date:** 2026-07-11 (plan 2026-07-09) · **Branch:** `feat/spec-decode` · **Box:** llmbench (M5 Max, 128GB)
**Model:** Qwen3-32B-4bit (dense; fp16 KV) · **Params:** ngram=3, K=8, gate window=32 / min-accept=0.25/step / cooldown=16, lookback=4096
**Evidence:** `harness-evidence.jsonl` (subcommands `verify-spec` + `bench` labels `pld-*`), `pld-shapes.csv`
**Evaluation lane:** [`EXACT`](README.md)—PLD changes the amount of target work, never the
temperature-zero greedy result. This byte-identity gate is specific to exact speculation; it
does not prohibit measured quality loss in `LOSSY_FRONTIER` techniques.

## What was built (Tasks 1–6, all verified)

- **The framework** (`HarnessCore/SpecDecode/`, pure/off-box): `SpecDrafter` protocol +
  `PromptLookupDrafter` (most-recent n-gram match), `SpecAccept.walk` (greedy accept-walk —
  the exactness property), `SpecEmit.trim` (replays the plain loop's budget/eos stopping rules
  over a batch), `PLDGate` (windowed-mean yield gate with cooldown probe). 21 dedicated tests.
- **Engine integration** (`SpikeCore`, on-box): `CompiledKVCache.truncate(to:)` KV rollback
  (in-graph offset rolled back in place, buffer identities preserved — compiled step stays
  valid), `CompiledMLXDecoder.generateSpec` (one batched verify forward over `[last] + draft`,
  accept-walk, rollback by the rejected count, gate fed every step), `SpecDecodeConfig`.
- **Harness wiring** (Task 6): `--spec pld [--ngram N] [--max-draft K] [--compiled-verify B]`
  through `RunConfig`/`SwiftEngineDriver`; `verify --spec pld` asserts the byte-identical
  PLD-on vs PLD-off equivalence + the drafting-engagement delta; `bench --spec pld` times the
  speculative path with drafted/accepted/gate telemetry in evidence.

**Phase-1 safety flags confirmed in the shipped loop:** the gate's cooldown clock advances on
EVERY decode step (`gate.record(accepted: 0)` on non-speculative steps), and the drafter's
backward scan is bounded (`lookback` = 4K trailing tokens), so it cannot go O(context²).

## The equivalence gate — byte-identical, both compile strategies (the headline property)

At temp 0, PLD-on must emit **exactly** the tokens PLD-off would — speculation changes how many
tokens a forward emits, never which. `verify --spec pld` on Qwen3-32B-4bit:

| probe | verify forward | result | engagement |
|---|---|---|---|
| echo-shape prompt, n=80 | uncompiled | **byte-identical 80/80** | drafted 80, accepted 61 (76.2%) |
| echo-shape prompt, n=80 | compiled fixed-K | **byte-identical 80/80** | drafted 80, accepted 61 (76.2%) |
| low-repetition prose, n=120 | uncompiled | **byte-identical 120/120** | drafted 48, accepted 48; gate disabled 16 steps mid-run and recovered |

PASS. The fixed-K-padding trick on the compiled path (pad short drafts with a repeat of the
last draft token; a padded token can only be emitted if it EQUALS the model's own argmax) is
exact in practice, not just by argument.

## Perf — decode tok/s, three shapes (bench, 256 max tokens, 3 runs post-warmup, cold-prefix salted)

| shape | PLD off | PLD on | Δ | acceptance | mean accepted/verify | gate behavior |
|---|---|---|---|---|---|---|
| **(a) preamble-then-echo** | 28.25 | **55.80** | **+97.5%** | 684/696 (98.3%) | 7.86 of K=8 | never disabled |
| (a) echo, compiled fixed-K verify | 28.25 | 55.31 | +95.8% | 677/696 (97.3%) | 7.78 | never disabled |
| **(b) code (repeated structure)** | 28.41 | 27.36 | **−3.7%** | 102/279 (36.6%) | 2.83 | disabled 192/663 steps (29%) |
| **(c) low-repetition prose** | 28.63 | 28.04 | **−2.1%** | 0/0 (no drafts proposed) | — | disabled 64/271 steps (24%) |

(a) is the product's target workload — agent/RAG loops that restate, quote, and echo context.
**+97.5% ≈ 2×**, far past the promote bar (≥ +20%, mlx-serve's PLD ballpark was +25%).

**Compile strategy (locked-decision follow-through):** the batched verify forward **traces
cleanly under `MLX.compile`** (fixed-K compiled step, drafts padded to K) — no fallback was
needed — but it is **not faster** than the uncompiled verify at this model size (55.31 vs
55.80, −0.9%, within run noise): a 32B forward is weight-bandwidth-bound, so the per-call
graph-construction cost the compile removes is already amortized over 9 positions. **Default:
uncompiled verify forward** (`compiledVerify=false`), one less traced graph to invalidate.

## The non-target shapes, honestly

Code and prose came in **slightly negative (−3.7% / −2.1%), not neutral**. The gate's job is
≈0% there; it only got us most of the way:

- **(c) prose:** the drafter never found a single 3-gram match (`drafted=0`), so the entire
  −2.1% is **spec-loop overhead, not wasted verifies**: the spec path drops the plain loop's
  submit-first pipelining (the next compiled forward is no longer asyncEval'd before the
  current token's `.item()` readback — a host-side drafting decision sits in between) and adds
  a per-step drafter scan. **Named fix:** restore the submit-first overlap on the empty-draft
  fallback steps (draft from the *predicted* context before the readback, or overlap the scan
  with the in-flight forward), and skip drafting entirely while the gate is disabled-cooling.
- **(b) code:** acceptance 36.6% — the shape has repeated identifiers but the model
  interleaves novel logic, so ~2/3 of drafted tokens are wasted verify positions. The gate
  disabled for 29% of steps but its `window=32` fill requirement means it tolerates ~32
  low-yield steps before flipping, and the cooldown re-probe (16 steps) re-enables into the
  same low-yield regime. **Named fix (gate tuning):** judge on a partial window (or shorten
  it), lengthen/back-off the cooldown, and count a verify's cost (K+1 positions) rather than
  raw accepted-per-step so the threshold tracks actual overhead.

Both regressions are bounded, understood, and sit behind a **per-request toggle** on an exact
transform — acceptable to ship for the workloads PLD targets while the two fixes land.

## Verdict — PROMOTE

**PROMOTE** PLD to the dial's spec-decode toggle, default-on for agentic/echo-shaped workloads:

- **Byte-identical at temp 0** — proven on-box, both verify strategies, repetitive and
  non-repetitive shapes. Zero quality loss is a measured property, not a claim.
- **Material win on the target shape:** +97.5% (≈2×) on preamble-then-echo at 98.3% acceptance
  — mlx-serve's PLD edge (178→222.5, +25%) is not just matched but beaten on this shape.
- **(c) is near-flat but not flat:** −2.1% prose / −3.7% code, root-caused (lost pipelining on
  fallback steps; gate window/cooldown tuning) with named follow-ups. PLD stays a toggle —
  off for known-low-repetition serving — until those land.

**Framework dividend:** DSpark/EAGLE-3 and DFlash implement `SpecDrafter` and reuse the
accept-walk, KV rollback, gate, and the harness's spec triad unchanged — this run validated
the whole on-ramp, not just PLD. Named invariant carried forward: PLD assumes a single
in-flight KV (disabled under continuous batching until fused).

## Reproduce

```sh
bash spike/scripts/sync_llmbench.sh
ssh llmbench@192.168.1.252   # then, in ~/fast-mlx-spike:
BIN=$(ls ~/Library/Developer/Xcode/DerivedData/fast-mlx-spike-*/Build/Products/Release/fastmlx-harness)
$BIN verify --model ~/perf-work/models/Qwen3-32B-4bit --spec pld --n 80                          # equivalence gate
$BIN verify --model ~/perf-work/models/Qwen3-32B-4bit --spec pld --n 80 --compiled-verify true   # compiled-verify variant
bash scripts/bench_pld_shapes.sh ~/perf-work/models/Qwen3-32B-4bit 3 256                          # three-shape table
```

On-box tests: `xcodebuild test … -only-testing:SpikeCoreTests` — 20/20. Off-box:
`swift test --filter HarnessCoreTests` — 108 XCTest + 17 swift-testing, 0 failures.

## RESOLVED 2026-07-11: performance gate cleared for a default-on product policy

**Feature SHA:** `bb5b06f22dc62e258c5ee1bdaadd6e53e1f8019d` · **Box:** M5 Max,
128GB · **Model:** Qwen3-32B-4bit (fp16 KV) · **Runs:** 3 post-warmup, 256 tokens

The first-run verdict named two reasons PLD could not yet be left on indiscriminately: the
spec loop abandoned the base decoder's submit-first pipeline whenever it had no useful draft,
and a low-yield request could spend 32 enabled steps proving it was cold. Both causes are now
fixed and re-measured. This addendum preserves the original result above as the historical
first run; it records the follow-up decision.

### What changed

- `generateSpec` now starts in the base loop's pipelined state. Empty-draft and gate-disabled
  rounds use the same `prefill` / `step` lookahead as PLD-off. A one-time two-deep transition
  restores that pipeline after a speculative round.
- The accept walk can combine an already-prefetched target pick with the verify rows produced
  after the draft. This retains the exact K+1 target sequence while avoiding a redundant
  synchronous forward when entering speculation from the base pipeline.
- The gate moved from `window=32, threshold=0.25, cooldown=16` to `window=8,
  minimumSamples=4, threshold=0.5, cooldown=32`. It can step aside after four clearly cold
  enabled steps, then waits longer before probing again.

The cache-state split is explicit: either the committed context is in KV with a target pick
pending, or the last emitted token is not yet in KV and no pick is pending. Rejected verify
rows are still rolled back in place. No unsafe concurrency escape was introduced; MLX state
remains actor-confined.

### Clean-SHA result

| shape | PLD off | PLD on | delta | acceptance | gate behavior |
|---|---:|---:|---:|---:|---|
| preamble-then-echo | 28.28 | **56.70** | **+100.50%** | 684/696 (98.3%) | never disabled |
| code / repeated structure | 28.39 | **29.31** | **+3.24%** | 92/151 (60.9%) | disabled 543 steps |
| low-repetition prose | 28.62 | **28.66** | **+0.14%** | 0/0 | disabled 205 steps |
| echo, compiled fixed-K verify | 28.28 | 52.63 | +86.10% | 684/696 (98.3%) | never disabled |

The task's ±1% neutrality band represented a no-regression safety bar. Prose is inside it at
+0.14%. Code finished outside it in the useful direction: accepted drafts produced a measured
+3.24% gain rather than overhead. The original target win is preserved and slightly improved,
from +97.5% to +100.5%. Compiled fixed-K verification remains slower than the uncompiled
default (52.63 versus 56.70 tok/s), despite still beating PLD-off by 86.1%, so the default
verify strategy does not change.

### Exactness and verification

Both verify strategies emitted **byte-identical 120/120-token streams** against PLD-off at
temperature 0. Each run drafted 72 tokens, accepted 53, engaged the speculative path, and
passed the exact triad. The nine evidence records (seven shape rows plus two exactness rows) all
carry the full feature SHA above. The external evidence bundle is
`final-bb5b06f.csv`, `final-bb5b06f.jsonl`, and `final-bb5b06f-verify.jsonl`; its hashes and
criterion mapping are recorded in [`docs/verification-evidence.md`](../../verification-evidence.md).

- Pure harness: `swift test --filter HarnessCoreTests` — 113 XCTest + 17 Swift Testing tests,
  0 failures.
- PLD unit coverage: `PLDGate.swift` 90.6%; `SpecAccept.swift` 90.5%.
- MLX-coupled suite: `xcodebuild test … -only-testing:SpikeCoreTests` — 20/20,
  `TEST SUCCEEDED` on the bench Mac.
- Focused cache-state and gate reviews: no correctness, security, or logic findings.

### Final verdict — PROMOTE / default-on performance gate cleared

PLD remains exact and now costs effectively nothing when it finds no work. It is **cleared for
a default-on product policy with an opt-out**. This patch deliberately does not claim that a
runtime default was flipped: the measurement harness still selects PLD explicitly with
`--spec pld`, and the current `RunConfig` default remains `nil`. That product/API wiring is a
separate surface decision, not an inference-kernel risk.

The named residual constraint is unchanged: PLD assumes one in-flight KV and must remain off
under continuous batching until that interaction is separately designed and measured.
Temperature greater than zero was not part of this exact-greedy acceptance gate.
