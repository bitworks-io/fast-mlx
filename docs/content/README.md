# fast-mlx Content Library

Short, self-contained technical write-ups — **one per significant spike, investigation, or optimization** — captured as source material for blog posts and larger whitepapers.

## The practice (standing)

After each notable spike / investigation / optimization, write one piece here **while the context is fresh**. Each should:
- **Stand alone** for a technical reader who doesn't know our codebase.
- Be **honest** — include the wrong hypotheses and negative results; the "we assumed X, measured, and X was wrong" arc makes the strongest, most credible content.
- Follow the arc: **problem/context → investigation → fix → result (real numbers) → generalizable lesson.**

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

## Candidate whitepaper themes (aggregations)

- **Building a high-performance MLX inference engine in Swift** — safety *and* performance on Apple Silicon (this gap-closing piece; future: the single-owner-actor eval loop, the test harness, custom Metal fusion).
- **The optimization dial: quantified precision-loss tuning** — measuring and shipping the speed↔quality frontier (future: the metric stack, per-model Pareto measurement, default-selection policy).
- **Serving big models on Apple Silicon** — SSD streaming + aggressive quantizer builds on unified memory (future: the Max-fit tier, DeepSeek-V4-Flash streaming at 128GB).
- **Rapid research integration** — the flywheel: turning a firehose of new inference techniques into measured, shippable dial tiers.

_Precedent: the mlx-serve `perf-case-study.html` in `docs/reference/mlx-serve-archive/` — the launch-piece source being rewritten for fast-mlx._
