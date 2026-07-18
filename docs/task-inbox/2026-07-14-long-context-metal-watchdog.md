---
status: captured
type: reliability-performance-gate
priority: high
created: 2026-07-14
source: user-supplied-omlx-digest-plus-primary-source-review
planning_ready: true
implementation_ready: false
---

# Long-context Metal watchdog and dispatch qualification

## Raw Capture

Qualify stock fast-mlx attention at 65K/128K on both the M5 Max and the offered pre-NAX M3 Ultra
before investing in a custom long-context or compressed-attention kernel. Distinguish thermal or
power throttling from an IOGPU command-buffer watchdog failure; do not treat either as a generic
"long context is slow" result.

## Light Triage

The original oMLX FA-256 report was later corrected: its observed M3 Max throughput cliff was power
throttling, not proof of command-buffer preemption. Separately, upstream MLX issue #3302 documents
real process-killing watchdog failures at large key lengths. The proposed upstream chunked-SDPA PR
#3307 was closed unmerged, so it must not be described as part of MLX 0.32. oMLX retained its own
shorter-dispatch implementation as a custom-kernel safeguard.

User/operator: a long-context user who needs the engine to remain responsive and predictable near
the advertised context ceiling, across Apple GPU generations.

Acceptance signals:

- staircase 8K/32K/65K/128K with stable power/thermal preconditioning and per-chunk throughput;
- cover available D=128 and D=256 GQA shapes, causal boundaries, and the actual production model
  path rather than only a synthetic Metal kernel;
- external liveness/progress heartbeat, captured process exit, OS/GPU error evidence, and an
  explicit watchdog-artifact check;
- report command/step duration, peak active/cache/RSS memory, workspace, and an explicit
  `Memory.cacheLimit` for any raised wired-memory limit;
- compare M5 Max with M3 Ultra so a generation-specific dispatch hazard is visible;
- only if the stock path reproduces a failure or material cliff, compare application-level
  chunking or a logsumexp-correct chunked kernel; prove logits/output correctness and no
  short-context regression before promotion.

## Official Sources

- [MLX watchdog failure report #3302](https://github.com/ml-explore/mlx/issues/3302)
- [Closed, unmerged chunked-SDPA PR #3307](https://github.com/ml-explore/mlx/pull/3307)
- [oMLX FA-256 investigation #2225](https://github.com/jundot/omlx/issues/2225)
- [oMLX shorter-dispatch implementation](https://github.com/jundot/omlx/commit/6d1617495c754873bdad3942e63af5dfd876d56e)

## Next Planning Step

Bring the M3 Ultra online, choose one D=128 and one D=256 model/fixture that fit both boxes, and
freeze the thermal, liveness, memory, and OS-log capture protocol before the first 65K run.
