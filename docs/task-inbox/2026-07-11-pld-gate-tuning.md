# PLD yield-gate tuning — keep non-target shapes neutral (before global default-on)

- **Captured:** 2026-07-11
- **Status:** backlog (follow-up from the PLD PROMOTE verdict)
- **Task type:** perf tuning
- **Priority:** medium — the gate before PLD can be *default-on* globally (it's already shippable as a per-request `--spec pld` toggle)
- **Owner project:** fast-mlx engine (`main`, PLD merged)

## Why (from the PLD first-run verdict)

PLD is exact (byte-identical) and **+97.5% (~2×) on high-repetition/echo workloads**, but the measured non-target shapes are slightly **negative**, not the ~neutral the yield-gate should deliver:
- **prose −2.1%** — pure spec-loop overhead: 0 drafts were even proposed, yet the spec path lost the base loop's submit-first pipelining on those fallback steps.
- **code −3.7%** — low-yield verifies (36.6% accept) that the gate's full-window / short-cooldown tuning tolerated too long before disabling (disabled 192/663 steps — too slowly).

A per-request toggle makes this a non-issue today (don't enable PLD for non-repetitive work). But for PLD to be **safely default-on**, non-target shapes must be ≈ 0%, not −2/−4%.

## The two named fixes

1. **Preserve submit-first pipelining on fallback (non-speculative) steps** — the empty-draft / gate-disabled path should decode exactly like the base loop (no lost overlap). This removes the prose −2.1%.
2. **Make the gate disable faster on low yield** — raise `minAcceptPerStep`, shorten the window, and/or judge partial windows; re-measure code/prose accept-rates and the disable latency. Target: non-target Δ within ±1%.

Re-measure all three shapes (echo/code/prose) through `spike/scripts/bench_pld_shapes.sh` after the fix; the echo win must be preserved. Verdict: `docs/superpowers/verdicts/2026-07-09-pld-firstrun.md`. Route engine work to deep-reasoner (fable).
