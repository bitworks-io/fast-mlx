---
status: planning-ready
type: product-ui-platform
priority: high
created: 2026-07-18
source: user
planning_ready: true
implementation_ready: false
---

# Adaptive macOS dial UI for fast-mlx

## Raw Capture

Capture the requirement for a rich but extremely lightweight/performance-first macOS UI that
adapts across low-end and high-end Apple Silicon; prioritizes ease of use; provides guided,
quantified performance-vs-quality tuning; exposes the full measured frontier while bounded by a
non-garbage floor; supports model and hardware profiles; includes chat, server control, model
management, and live metrics; supports menu bar behavior, auto-update, and accessibility; preserves
CLI/API parity; and adds no UI overhead to a headless engine.

## Agent Notes

Requirements and acceptance signals are captured and ready for a dedicated product plan. No UI
implementation exists yet.

This aligns with the platform design spec's v1 app surface and optimization dial:

- Section 4: the dial must show measured speed/quality points, default to
  fastest-with-unnoticeable-loss, and allow informed opt-in across the measured frontier while
  refusing incoherent or degenerate output points.
- Section 8: v1 includes a SwiftUI macOS app for chat, server control, and dial UI alongside the
  OpenAI-compatible API and CLI.
- Section 9: the model catalog is curated and profile-driven, with license and hardware-fit details
  treated as load-bearing.
- Section 10: the macOS app dial UI follows the engine, harness, quant, catalog, and measurement
  pipeline work rather than leading it.
- Section 12: v1 is done only when the app, CLI, and API serve the curated catalog with measured
  per-model and per-hardware frontiers and reproducible `fast-mlx verify` evidence.

Public-facing requirements should avoid competitor callouts. Any competitive positioning or
comparison context belongs in private planning notes unless explicitly approved for publication.

## Light Triage

User/operator: a macOS user running fast-mlx locally who needs a simple, trustworthy way to chat,
serve models, manage models, observe performance, and tune speed versus quality without losing
control over resource use or exact CLI/API workflows.

Desired outcome: a polished native macOS control surface that makes the optimization dial
understandable and safe, scales its UI and defaults to the user's Apple Silicon hardware profile,
and remains strictly optional for headless engine use.

Task type: product requirement capture for future macOS app, engine operability, catalog, and
measurement-plumbing planning.

Priority: high, because the UI is the main product surface for the measured dial and hardware-fit
promise, but implementation should wait for the underlying engine/harness/catalog frontiers to be
ready enough to drive it with real measurements.

Dependencies:

- Engine serving path with no UI dependency or runtime overhead for headless CLI/API operation.
- Measured dial frontiers per model, hardware class, and quality tier.
- Hardware/profile detection and capacity advisor data.
- Curated model catalog with license, memory-fit, context, and default-policy metadata.
- Live metrics stream for TTFT, TPOT, throughput, memory pressure, queueing, cancellation, and
  active server state.
- Update distribution, signing/notarization, and accessibility validation decisions for the macOS
  app.

Acceptance signals:

- A user can launch the macOS app, select or install a supported model, start/stop the local server,
  chat with the model, and see live health/performance metrics without using the terminal.
- The dial presents measured speed, latency, memory, and quality-loss numbers for the selected
  model and detected hardware profile, including the default fastest-unnoticeable-loss point.
- The UI exposes more aggressive measured points only with clear quantified tradeoffs and refuses
  points below the non-garbage floor.
- Low-memory or lower-end Apple Silicon profiles receive safer defaults, fit guidance, and
  degraded-but-usable UI behavior; high-memory profiles surface larger-model and long-context
  options when measurements support them.
- CLI and API users can perform equivalent model selection, server control, verification, and dial
  inspection without launching the UI.
- A headless serve/run/bench/verify workflow has no linked, initialized, or resident UI subsystem
  overhead.
- Menu bar status/control, auto-update behavior, and accessibility checks are verified before the
  app is treated as production-ready.

Known failure cases:

- The UI displays unmeasured estimates as if they were signed frontier points.
- A quality setting that fails the coherence, non-crash, or non-NaN floor is selectable.
- App defaults are tuned for large-memory machines and cause poor first-run behavior on smaller
  Apple Silicon systems.
- The app becomes required for server operation or adds runtime work to headless CLI/API serving.
- Live metrics are too coarse, delayed, or confusing to support guided tuning.
- Accessibility, menu bar lifecycle, background server lifecycle, or auto-update failure paths are
  treated as polish rather than product requirements.

Open questions:

- Which Apple Silicon hardware classes are first-class profiles for v1 beyond the current
  256GB/512GB big-memory references?
- What minimum supported macOS version and SwiftUI/AppKit mix should the app target?
- Which update channel, signing, notarization, and rollback mechanism should be used?
- What exact metric set and sampling cadence should the live metrics API expose to avoid UI-driven
  overhead?
- How should model installation, disk pressure, license acknowledgement, and failed download states
  be represented?
- What is the acceptable UI idle CPU, memory, and wakeup budget on low-end machines?

## Next Planning Step

Start the macOS app and operability planning lane in parallel with engine profiling. First define
the signed/measured dial-data contracts and no-overhead headless boundary; then map first-run model
install, chat, server, tuning, failure/recovery, and live-metrics flows to CLI/API equivalents and
verification checks. Implementation remains gated on stable serving, catalog, update, and metrics
contracts rather than on completion of every engine optimization.
