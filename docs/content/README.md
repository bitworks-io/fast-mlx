# fast-mlx Content Library

Short, self-contained technical write-ups — **one per significant spike, investigation, or optimization** — captured as source material for blog posts and larger whitepapers.

## The practice (standing)

After each notable spike / investigation / optimization, write one piece here **while the context is fresh**. Each should:
- **Stand alone** for a technical reader who doesn't know our codebase.
- Be **honest** — include the wrong hypotheses and negative results; the "we assumed X, measured, and X was wrong" arc makes the strongest, most credible content.
- Follow the arc: **problem/context → investigation → fix → result (real numbers) → generalizable lesson.**
- Treat this directory as source material, not automatic publication approval. Public pieces report
  fast-mlx's own reproducible results; private competitor performance/feature litmus stays out of
  public copy unless the owner explicitly approves a claim-review exception.

## Format

- File: `YYYY-MM-DD-<slug>.md`, blog-post length (~600–1200 words).
- Tag the candidate **whitepaper theme(s)** at the top so pieces aggregate cleanly.
- Draft quality — polish only when a piece is picked for publication.

## Index

| Date | Piece | Whitepaper theme(s) |
|---|---|---|
| 2026-07-09 | [The bottleneck wasn't the language: closing an 18% Swift-vs-Zig MLX inference gap](2026-07-09-closing-the-swift-mlx-gap.md) | Building a high-performance MLX engine in Swift |
| 2026-07-09 | [Who measures the measurer? Auditing a precision-loss harness that was quietly lying](2026-07-09-trusting-the-instrument.md) | The optimization dial — quantified precision-loss tuning |
| 2026-07-09 | [The 7K wall that wasn't: jetsam forensics, a quadratic allocator, and the statistic hiding in the tail](2026-07-09-the-wall-that-wasnt.md) | The optimization dial — quantified precision-loss tuning |
| 2026-07-09 | [One formula, wrong for a third of the catalog: a KV-memory model that refuses to lie](2026-07-09-one-formula-wrong-for-a-third-of-the-catalog.md) | Serving big models on Apple Silicon |
| 2026-07-09 | [We implemented Google's TurboQuant exactly, matched the paper's error tables — and it still lost to plain 4-bit quantization](2026-07-09-turboquant-exact-math-still-lost.md) | The optimization dial — quantified precision-loss tuning; Building a high-performance MLX engine in Swift |
| 2026-07-11 | [2× for free, when the model repeats itself: prompt-lookup decoding with a byte-identical proof](2026-07-11-two-x-for-free-when-the-model-repeats-itself.md) | Building a high-performance MLX engine in Swift; Rapid research integration — the flywheel |
| 2026-07-11 | [When zero speculation costs 2%: making a 2× decoder safe to leave on](2026-07-11-when-zero-speculation-costs-two-percent.md) | Building a high-performance MLX engine in Swift; Rapid research integration — the flywheel |
| 2026-07-12 | [The backlog was lying: when “already have” meant “the old engine had it”](2026-07-12-the-backlog-was-lying.md) | Rapid research integration — the flywheel; Building a high-performance MLX engine in Swift; The optimization dial — quantified precision-loss tuning |
| 2026-07-12 | [“Lossless” wasn't byte-identical: the speculative decoder that failed at generated index seven](2026-07-12-lossless-wasnt-byte-identical.md) | Rapid research integration — the flywheel; Building a high-performance MLX engine in Swift |
| 2026-07-14 | [The fastest request wasn't the fastest service](2026-07-14-the-fastest-request-wasnt-the-fastest-service.md) | Building a high-performance MLX inference engine in Swift; Rapid research integration — the flywheel |
| 2026-07-18 | [When smaller KV is not faster](2026-07-18-when-smaller-kv-is-not-faster.md) | The optimization dial — quantified precision-loss tuning; Serving big models on Apple Silicon; Rapid research integration — the flywheel |
| 2026-07-21 | [Fifteen times faster still was not fast](2026-07-21-fifteen-times-faster-still-not-fast.md) | Building a high-performance MLX inference engine in Swift; The optimization dial — quantified precision-loss tuning; Rapid research integration — the flywheel |
| 2026-07-23 | [Llama ran, but it did not earn a speed tier](2026-07-23-llama-ran-without-a-speed-tier.md) | Serving big models on Apple Silicon; The optimization dial — quantified precision-loss tuning; Rapid research integration — the flywheel |
| 2026-07-23 | [The third geometry said no](2026-07-23-the-third-geometry-said-no.md) | Serving big models on Apple Silicon; The optimization dial — quantified precision-loss tuning; Rapid research integration — the flywheel |

## Candidate whitepaper themes (aggregations)

- **Building a high-performance MLX inference engine in Swift** — safety *and* performance on Apple Silicon (this gap-closing piece; future: the single-owner-actor eval loop, the test harness, custom Metal fusion).
- **The optimization dial: quantified precision-loss tuning** — measuring and shipping the speed↔quality frontier (future: the metric stack, per-model Pareto measurement, default-selection policy).
- **Serving big models on Apple Silicon** — SSD streaming + aggressive quantizer builds on unified memory (future: the Max-fit tier, DeepSeek-V4-Flash streaming at 128GB).
- **Rapid research integration** — the flywheel: turning a firehose of new inference techniques into measured, shippable dial tiers.

_Precedent: the mlx-serve `perf-case-study.html` in `docs/reference/mlx-serve-archive/` — the launch-piece source being rewritten for fast-mlx._
