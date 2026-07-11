# Speculative Decoding — PLD first (the spec-decode framework) Implementation Plan

> **For agentic workers:** superpowers:subagent-driven-development. Phase 1 is pure/off-box (**builder** ok). Phase 2+ is engine/MLX-coupled on `llmbench` (**deep-reasoner**). Branch `feat/spec-decode` off `main`.

**Goal:** the first decode-speed multiplier beyond the GPU-bound base loop — port **Prompt-Lookup Decoding (PLD)** on a reusable **spec-decode framework**. Biggest single-stream win on the product's real workload (agentic/RAG/code repetition), **distribution-preserving (exact → zero quality loss)**, no draft model, and the foundation DSpark/DFlash reuse.

**Why PLD first (recon 2026-07-09):** the compiled decode core is at Zig parity (155.5 vs 153.65 tok/s, GPU-bound); mlx-serve's edge is the multipliers on top, all absent here. PLD is the cheapest (no draft model), highest-ROI-on-workload (mlx-serve: 178→222.5 tok/s on a realistic preamble-then-echo agent shape, self-managing via a yield-gate), and exact.

**Architecture:** a `SpecDrafter` proposes up to K continuation tokens from the context; the target does **one batched verify forward** over `[lastToken, draft₁…draftₖ]`; a **distribution-preserving accept-walk** takes the longest prefix the target agrees with + one bonus token; the KV cache is **rolled back** by the rejected count; a **self-managing yield-gate** disables PLD when acceptance is too low and re-enables on recovery. PLD is the first `SpecDrafter` (n-gram prompt lookup). DSpark/DFlash later implement the same protocol.

**Tech stack:** Swift 6, pure `HarnessCore` for the framework logic (off-box TDD), `SpikeCore` for the engine integration (on-box, `CompiledMLXDecoder`/`CompiledKVCache`), the `fastmlx-harness` bench/verify for measurement. On-box build: `xcodebuild -skipPackagePluginValidation` (as documented in the handoff).

---

## Locked design decisions (v1)

| Decision | v1 choice | Later |
|---|---|---|
| First drafter | **PLD** (n-gram prompt-lookup, no model) | DSpark/EAGLE-3, DFlash implement `SpecDrafter` |
| Sampling | **Greedy (temp 0)** exact accept first | Leviathan-Chen rejection accept for temp>0 (the carry-forward L1 batched accept) |
| Verify forward | fixed **K** positions; compile strategy chosen on-box (separate compiled verify vs uncompiled) — measure both | — |
| Self-management | **yield-gate** (disable when accept-rate low; re-enable mid-request on recovery) — proven in Zig | — |
| Batched arm | PLD **disabled under continuous batching** (assumes single in-flight KV) — a named invariant, matches spec §5 | fused later if measured |

**Distribution-preserving is the headline:** at temp 0, PLD-on output must be **byte-identical** to PLD-off. That is the equivalence gate and the "zero quality loss, pure speed" marketing claim — provable, not asserted.

---

## Phase 1 — the framework core (pure, off-box, TDD) · `HarnessCore/SpecDecode/`

Pure logic, no MLX — the algorithm, de-risked and fully tested before touching the engine.

**Task 1: `SpecDrafter` protocol + PLD drafter.** `propose(context:[Int], k:Int) -> [Int]` — PLD finds the most recent earlier occurrence of the context's last-`n` tokens (the match n-gram) and returns the up-to-`k` tokens that followed it; empty if no match. Params `ngram` (match length, e.g. 3) + `maxDraft` (K, e.g. 8). TDD: a repeated phrase yields the correct continuation; no-match yields empty; the *most recent* occurrence wins; near-the-end matches clamp to available tokens.

**Task 2: the greedy accept-walk (distribution-preserving).** `accept(draft:[Int], verifyArgmax:[Int]) -> (acceptedCount:Int, bonusToken:Int)` — accept `draft[i]` while `draft[i] == verifyArgmax[i]`; at the first mismatch (or after all K accepted) the bonus token is `verifyArgmax[acceptedCount]`. So one verify forward emits `acceptedCount + 1` tokens, **exactly what greedy decode would have produced**. TDD: full-accept (K+1 emitted), zero-accept (1 bonus = plain decode), partial-accept, and the invariant that the emitted sequence equals plain greedy on the same logits.

**Task 3: the yield-gate.** `PLDGate` — tracks accepted-tokens-per-step over a sliding window; `enabled` flips false when the rate drops below a threshold (e.g. < 0.25 over 32 steps) and re-enables after a cooldown to probe recovery. Pure state machine. TDD: a low-acceptance run disables it; a recovery re-enables it; a high-acceptance run keeps it on.

Each task: failing test → implement → green → commit. `swift test --filter SpecDecodeTests` on this host.

---

## Phase 2 — engine integration (on-box, deep-reasoner) · `SpikeCore`

**Task 4: KV rollback.** Add `truncate(to newLength:Int)` to `CompiledKVCache` (+ the `CompiledCache` protocol) — roll the in-graph offset + logical length back by the rejected count, keeping buffer identities the compiled step is bound to (mirror `resetInPlace`'s discipline). TDD: append N, truncate to M<N, verify the next read/decode matches having only appended M.

**Task 5: the batched verify forward + PLD decode loop.** In `CompiledMLXDecoder`, add a spec-decode path behind a flag: each step, if the gate is enabled, draft K via `SpecDrafter`, run one verify forward over the K+1 tokens, accept-walk, emit accepted+bonus, **truncate the KV by the rejected count**, feed the gate. Choose the verify-forward compile strategy on-box (separate fixed-K compiled step vs uncompiled) — **measure both**; document which. Engagement marker: count drafted/accepted tokens (for the equivalence + measurement gates).

**Task 6: harness wiring.** A `--spec pld` flag through `RunConfig`/the driver; the bench records tokens/s with spec on/off; `verify` asserts the distribution-preserving equivalence + engagement.

---

## Phase 3 — measure + promote/shelve (on-box)

**Task 7.** Bench PLD-on vs PLD-off tok/s on three shapes on Qwen3-32B-4bit (and the fast MoE): **(a) preamble-then-echo** (agent — high repetition, PLD's best case), **(b) code** (repeated identifiers/structure), **(c) low-repetition prose** (the yield-gate must keep this ~flat, not regressed). Equivalence: temp-0 **byte-identical** PLD-on vs off (the exactness proof) + engagement delta. 

**Promote** to a default-on-for-agentic-workloads spec-decode path iff: byte-identical at temp 0, **material win on (a)/(b)** (target ≥ +20% on the echo shape, matching mlx-serve's ballpark), and **no regression on (c)** (the yield-gate's job). Else shelve with a dated negative result + the measured accept-rates. Record the result in a verdict + a content piece. This is the same flywheel discipline as TurboQuant — the difference is PLD is *exact*, so the only question is speed, not quality.

**Reusability:** DSpark/EAGLE-3 (archive design ready) and the intake's **DFlash** (block-diffusion drafter, claimed 1.7–1.9× on quantized MoE) implement `SpecDrafter` + reuse the accept-walk, rollback, and gate — this framework is the on-ramp for all of them.
