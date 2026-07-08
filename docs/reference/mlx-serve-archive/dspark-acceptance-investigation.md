# DSpark acceptance investigation (milestone 2.2) — findings

Goal: lift DSpark per-draft acceptance from 11.4% (8-bit Qwen3-8B target) toward
reference levels, keeping byte-equivalence + a measured throughput win.

Reference source: `/Users/braymond/dspark-ref/DeepSpec/deepspec` (DeepSpec).

## Verified CORRECT in the Zig port (ruled out as the killer)

1. **Tap layer index.** Reference `extract_context_feature`
   (`modeling/dspark/common.py:52`): `hidden_states[layer_id + 1]` = raw output
   of decoder layer `layer_id`. Zig `forwardWithCaptureLayers` captures `h` (raw
   post-residual, pre-final-norm) at loop index `li == layer_id`
   (`transformer.zig:4068`). For `[1,9,17,25,33]` both give raw outputs of
   decoder layers 1/9/17/25/33. **Match.**
2. **Tap concat order.** Reference cats in `target_layer_ids` order; Zig
   `layer_outs[ti]` is filled in the same order and `concatHiddenAxis` preserves
   it. **Match.**
3. **Markov walk.** Reference `sample_block_tokens` (`markov_head.py:55`):
   `step_logits = base_logits[:,k,:] + w2 @ w1[prev]`, sample, chain prev. Zig
   `nextDspark` Phase 1b (`generate.zig:2767`) is identical. **Match.**
4. **RoPE convention.** Reference `apply_rotary_pos_emb` uses `rotate_half`
   (NEOX/non-interleaved), Q at `cos[-q_len:]` (draft positions), K over full
   span; k_norm applied to concatenated k *before* rope; θ=1e6, full head_dim=128.
   Zig `mlx_fast_rope(..., traditional=false, base=1e6, dims=128, ...)`, QK-norm
   before rope. **Match.** Positions self-consistent on a fresh request.
5. **Noise embedding / anchor / mask.** `[anchor] ++ [mask]*(block_size-1)` via
   the drafter's own `embed_tokens`. **Match.**
6. **Fusion.** `hidden_norm(fc(taps))` — Zig `projectTargetContext`
   (`dspark.zig:573`) matches reference `_forward_backbone:373`. fc is
   [5*4096,4096] (non-square → a wrong transpose would shape-fail at load; it
   binds).

Byte-equivalence at temp=0 (proven by `test_dspark_equivalence.sh`) confirms the
verify/accept/rollback machinery is sound. So 11.4% is a **proposal-quality /
metering** question, not a corrupted-output bug.

## The reframe (diverges from the handoff's framing)

`per_draft_pct = accepted_drafts / (attempts × block_size)` with block_size=7.
11.4% ⇒ ~0.8 accepted drafts/round ⇒ ~1.8 committed tokens per verify forward.
The reference measures acceptance over the **confidence-truncated proposal
length**, not the full block — and the Zig confidence gate is NOT wired
(milestone 2.1, always proposes the full 7). So a drafter accepting ~0.8/round
(potentially ~reference-level throughput) is metered ~6× low, and the runtime
gate (`RUNTIME_GATE_MIN_PER_DRAFT_RATE = 0.50`) then **disables** it → no
speedup, no regression.

Implication: the first question is not "why are proposals weak" but "**is the
throughput actually a win with the gate forced off?**" That distinguishes a
genuinely-weak drafter from a mis-metered OK one. The measurement is the arbiter.

## Remaining untested hypotheses (ranked)

- **H-metric:** full-block denominator + unwired confidence gate under-report
  acceptance and trip the 0.50 runtime gate. Test: force gate off, measure
  tokens/sec vs no-dspark baseline; wire confidence gate; re-meter over effective
  length.
- **H-bf16 (handoff #1):** 8-bit target hiddens vs the bf16 the drafter was
  trained on. Test: `DSPARK_TARGET_MODEL=Qwen3-8B-bf16` (downloading on M5), no
  code change.
- **H-acceptprob:** Zig verify uses one-hot `min(1, target_p)`; reference uses
  `min(1, target_p/draft_p)` (Leviathan). Identical at temp=0; only affects
  stochastic acceptance accounting/throughput at temp>0.

## MEASUREMENT RESULTS (M5, gate forced off via SPEC_GATE_MIN_RATE=0)

Story prompt, 256 tokens, `--log-level debug`. Drafter `dspark_qwen3_8b_block7`.

### 8-bit target (Qwen3-8B-8bit)
| config | temp | tok/s | vs baseline |
|---|---|---|---|
| baseline (no dspark) | 0 | 62.6 | — |
| baseline | 0.7 | 62.5 | — |
| dspark (gate off) | 0 | 41.1 | **−34%** |
| dspark (gate off) | 0.7 | 39.0 | **−38%** |

Histogram (197 rounds): mean **1.60** accepted/round, **P(≥1)=0.67**, P(≥2)=0.40,
max=7. per_draft_pct 23% (= 1.6/7).

### bf16 target (Qwen3-8B, full precision) — hypothesis 1
| config | temp | tok/s | vs baseline |
|---|---|---|---|
| baseline (no dspark) | 0 | 31.8 | — |
| dspark (gate off) | 0 | 24.1 | **−24%** |
| dspark (gate off) | 0.7 | 18.7 | **−42%** |

Histogram (219 rounds): mean **1.34** accepted/round, **P(≥1)=0.65**.

## Conclusions

1. **Reframe CONFIRMED.** The drafter is correct and healthy — ~1.5 accepted
   tokens/round, P(accepted≥1)≈66% on BOTH targets. The "11.4%/23% per_draft_pct"
   is the full-block (÷7) denominator on a genuinely-good drafter, not a broken
   head. Acceptance was never the blocker.
2. **Hypothesis 1 (bf16 target) REJECTED by data.** bf16 does not raise
   acceptance (1.34 vs 1.60 — if anything slightly lower), and bf16 halves the
   baseline (2× weight read) so throughput is worse. The drafter was NOT
   mis-served by the 8-bit target.
3. **Q-RoPE confirmed correct** (matches reference `rotate_half`/θ=1e6/full-dims,
   and the main Qwen3 path's `traditional=false` — no inherited flag flip).
4. **Real blocker = batch=1 round cost.** DSpark net-regresses 24–42% on both
   targets. With a≈1.6, speedup = (1+a)/r; observed 0.65× ⇒ r≈4. The fat round:
   5-layer backbone + **8× full-vocab (151936) projections** (base lm_head over 7
   + 7 sequential Markov matmuls) + a **partial-accept re-forward that fires on
   ~97.5% of rounds** (full-7-accept is only 2.5%) = a SECOND target forward.
   The runtime gate disabling it PREVENTS this regression — it is not leaving a
   speedup on the table.
5. **A lean-round MLX win exists — but is proven only at 4B, not 8B.**
   Provenance check (advisor-prompted): the "~1.6×" is `mlx-dspark`, a real
   Apple-Silicon MLX port — but the published number is **Qwen3-4B on an M4 Pro**
   (52.9→~73 tok/s). For **Qwen3-8B it is aspirational only** ("the same approach
   *should also* work," a tweet; "benchmarked primarily on smaller models"). The
   DeepSpec reference I read is CUDA (README: 8-GPU, `CUDA_VISIBLE_DEVICES`). So
   the handoff's "published mlx-dspark numbers (Qwen3-8B ~1.6×)" MISATTRIBUTED a
   4B number to 8B. **No one has demonstrated a DSpark win at 8B on Apple
   Silicon.** A win here would require a LEAN round — (a) eliminate the
   re-forward by cropping the target KV to accepted+1 (mlx-dspark/reference
   approach) instead of rollback+re-forward; (b) wire the confidence gate (m2.1)
   to propose the confident prefix (~2–3) not the full 7 — real engineering with
   UNPROVEN 8B payoff. Neither is "acceptance tuning."

### Caveats on the round-cost analysis (not over-claimed)
- **r≈4 composition is INFERRED, not measured.** Bandwidth predicts ~1.3–1.5;
  two target forwards (verify + the 97.5%-firing re-forward) get ~2.8; the
  remaining ~1.2× is overhead attributed to the 8× vocab projections / 7
  sequential Markov dispatches / the per-round KV snapshot+restore on a growing
  8B cache — by reasoning, not instrumentation. Sizing which lever closes the
  gap needs one instrumented run (wall-clock around proposeBlock / verify /
  re-forward / markov-walk / snapshot in `nextDspark`). Eliminating the
  re-forward alone moves r 4→~3 (≈0.9×, still a loss); it needs the confidence
  gate on top to *maybe* cross 1×.
- **one-hot→ratio accept is a temp0.7 accounting sag, NOT a win-lever.** The
  temp0.7 numbers (esp. bf16 18.7, avg_per_round 1.08) are worse than temp0 —
  consistent with the Zig verify's one-hot `min(1,target_p)` under-accepting vs
  the reference `min(1,target_p/draft_p)`. It preserves the target marginal (so
  temp0 stays byte-identical) but does NOT reduce r, so it cannot turn the
  regression into a win. Noted; closed as not-a-speedup-path.

## Byte-equivalence gate — RE-VERIFIED this session on the new binary
`test_dspark_equivalence.sh` (Qwen3-8B-8bit target, gate forced off): **7 passed,
0 failed** — chat non-stream, chat stream, /v1/messages non-stream all
byte-identical to the no-dspark baseline; engagement + `enable_dspark:false`
opt-out confirmed. The `SPEC_GATE_MIN_RATE` env override is default-preserving
(unset → shipped 0.50) and does not alter output.

## Milestone premise has shifted
The handoff scoped m2.2 as "lift 11.4% acceptance toward ~73%." The data shows
acceptance is already fine; the work that would unlock a win is round-cost
reduction, a different and larger effort. This is a decision point for the user.

## PURSUIT (user chose to build the lean round)

### Lever 1 — eliminate the partial-accept re-forward (committed 92f001e)
Replaced restore+re-forward with `KVCache.truncate(pre_verify_len+1+accepted)` in
the partial-accept branch (byte-identical by causal attention; DSpark targets are
dense Qwen3 so no SSM/MoE to reconcile). **Measured +24% on the DSpark path**
(8-bit temp0 41.1→50.9, temp0.7 39.0→46.7 tok/s), byte-equivalence 7/7 PASS,
acceptance unchanged. r ~3.95→~3.19. Still −19% vs baseline.

### Per-round timing breakdown (MLX_SERVE_DSPARK_TIMING, serialized upper bounds)
M5 8-bit temp0.7, 43 rounds: **verify=36.9ms (DOMINANT)**, propose=11.9,
markov=3.7, accept+append=1.2 → ~53.6ms/round, mean_accepted=1.33.

**The target verify forward dominates.** A baseline decode is ~16ms; the 8-position
verify is ~37ms ≈ 2.3× — the extra 7 query positions cost real compute (~3ms/pos:
fixed weight-read ~13ms + 3ms/pos). So the round is NOT drafter-dominated — the
earlier inference was wrong. Economics: round ≈ 29ms FIXED (propose 12 + verify
weight-read ~16 + accept 1) + ~3.5·m; to beat baseline needs (1+a)·16ms > round.
At m=7, a=1.33: 37.3 < 53.6 → lose. Even at m=1 (a≤1): 16·2 < 32.5 → still lose.
**To win, tokens/round (acceptance) must rise well above ~2.3, OR the confidence
gate must shrink the dominant verify enough — both uncertain at 8B batch=1.**

Lever candidates now, ranked by the timing:
- **Confidence gate (m2.1)** — shrinks the DOMINANT verify (1+P vs 1+m positions);
  byte-equivalence-trivial (proposal length only changes speed). But fewer
  proposed ⇒ fewer accepted; model says P≈2–3 lands ~break-even (~60–62 vs 62.5).
- **Ratio accept at temp>0** — `min(1,target_p/draft_p)` instead of one-hot
  `min(1,target_p)`; accepts MORE drafts (higher a) against a fixed round, raising
  tok/s where it matters (agent traffic). Distribution-preserving; temp0 unchanged
  (identical there → equivalence gate holds). Requires computing draft_probs.
  This is literally the handoff's "re-derive the acceptance."

### Lever 2 — confidence gate: KILLED as a win lever (advisor + economics)
Can't touch the 12ms propose (backbone runs the full block regardless), and
fewer-proposed ⇒ fewer-accepted. Floor argument: even a FREE verify at m=1 gives
round ≈ propose(12)+accept(1)+~19 ≈ 32ms for ~1.65 tokens = ~50 tok/s < 62.5.
No proposal length wins at current acceptance. Not built.

### Lever 3 — ratio-accept headroom probe (the only lever that raises `a`)
One-hot `min(1,target_p)` under-accepts vs ratio `min(1,target_p/draft_q)`;
ratio is temp>0-only (temp0 both = argmax, so temp0 stays −19% regardless — the
gate correctly disables it there). Measured the counterfactual (env-gated
`MLX_SERVE_DSPARK_RATIO_PROBE`, zero behavior change, 54 rounds temp0.7):

| metric | value |
|---|---|
| E[accepted] one-hot (shipped, ≈ actual 1.80) | 1.788 |
| E[accepted] ratio (counterfactual) | 2.057 |
| headroom | **1.15×** |
| mean draft_q | **0.751** |

The drafter is CONFIDENT (mean draft_q 0.75 → ratio ≈ one-hot), so ratio buys
only +15% acceptance → ~2.06/round, still below the ~2.35/round a m=7 win needs
(→ ~57–61 tok/s, break-even-to-just-under the 62.5 baseline). **Not a clear
measured win. Not built** (per the advisor's rule: headroom didn't jump toward
~2.3). Ratio genuinely helps only when the drafter is uncertain (draft_q ~0.3,
where ratio_E ≈ 4× one-hot) — rare on this pairing.

## FINAL VERDICT (pursuit complete)
- **Lever 1 (re-forward elimination): SHIPPED** — +24% on the DSpark path,
  byte-equivalent (7/7), committed `92f001e`. A strict, correct improvement.
- **DSpark on Qwen3-8B at batch=1 still does NOT beat the no-dspark baseline**
  (50.9 vs 62.5 tok/s temp0). The target verify (~37ms, 2.3× a decode) dominates
  the round; the drafter adds a fixed ~13ms tax; acceptance (~1.5–1.8/round) is
  healthy but below the ~2.35/round the economics require. The runtime gate
  correctly keeps DSpark OFF here → no regression.
- Both handoff levers closed (Q-RoPE correct; bf16 rejected); confidence gate
  killed by economics; ratio-accept measured as break-even (temp>0-only). The
  MLX-proven DSpark win is 4B (mlx-dspark, M4 Pro); 8B is unproven upstream.
- Instrumentation kept as env-gated diagnostics (`MLX_SERVE_DSPARK_TIMING`,
  `MLX_SERVE_DSPARK_RATIO_PROBE`, `SPEC_GATE_MIN_RATE`) — useful when re-testing
  DSpark on a bigger/slower target where the round economics shift favorably.

## Where a DSpark win likely IS (future, not this run)
A slower/bigger target (27B+/frontier MoE) where a single decode is much slower
than the ~13ms drafter tax, so the (1+a) speculative tokens dominate — the regime
the frontier scale-out plan targets. Not measurable here without those drafters.

## Gate (held throughout)
No change shipped without byte-equivalence preserved AND a measured tokens/sec
win. Lever 1 cleared it (byte-equivalent + measured +24%); confidence gate and
ratio-accept did not clear the win bar and were NOT shipped.
