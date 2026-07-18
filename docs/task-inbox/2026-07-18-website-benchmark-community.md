---
status: captured
type: website-community-automation
priority: unspecified
created: 2026-07-18
source: user-request
project: fast-mlx
planning_ready: false
implementation_ready: false
---

# fast-mlx website, benchmark publication, and community surfaces

## Raw Capture

Earlier user request, captured as a planning seed:

Have another agent team build the fast-mlx website, including automated benchmark-result
collection and publication, community engagement, update/news surfaces, and useful
agent-facing engagement for people working with or evaluating fast-mlx.

## Light Triage

Target audience: prospective users, technical evaluators, community contributors, and
agent/human maintainers who need an accurate public surface for fast-mlx's measured
capabilities and current project activity.

Obvious scope:

- Public website for fast-mlx as an Apple Silicon MLX inference platform.
- Automated collection and publication path for benchmark results produced by the fast-mlx
  harness, with enough provenance for readers to understand model, hardware, dial setting,
  commit, workload, and measurement method.
- Update/news surfaces fed by the existing `docs/content/` practice and dated verdicts.
- Community engagement surfaces for announcements, contribution pathways, issue/request
  intake, and evidence-backed discussion.
- Agent-facing engagement that helps future Codex/Claude/human agents find current docs,
  benchmark evidence, open tasks, publication rules, and safe contribution boundaries.

Dependencies:

- Platform positioning and public claims must align with
  `docs/superpowers/specs/2026-07-08-fast-mlx-platform-design.md`.
- Long-form technical material should draw from `docs/content/README.md` and the indexed
  content pieces, especially measured optimizations and negative results.
- Current implementation status and open-work ordering must align with `docs/agent-handoff.md`.
- Published benchmark data should come from durable harness/verdict/evidence artifacts, not
  ad hoc chat summaries or unsupported claims.
- Any automation requires a future decision on hosting, source of truth, publication cadence,
  review gates, and whether benchmark artifacts are generated locally, in CI, on `llmbench`,
  or by another trusted runner.

Trust and privacy notes:

- Public site claims must be fast-mlx-only and evidence-backed.
- Private competitor tracking, unpublished competitive research, private benchmark notes,
  machine-local paths, account details, credentials, and private overlay content must not leak
  into the public website, public JSON feeds, release notes, or community surfaces.
- Benchmark publication needs provenance and review controls so old, partial, failed,
  machine-specific, or incomparable runs are not presented as current product claims.
- Do not choose or configure a deployment provider as part of this capture; that belongs to a
  later planning decision.

Open questions:

- What is the intended first audience: commercial evaluators, open-source contributors,
  fast-mlx users, internal operators, or agents maintaining the project?
- Which benchmark rows are safe to publish, and what minimum evidence fields are mandatory
  before a row can appear publicly?
- Should the website publish only promoted dial tiers, or also dated shelved/negative results
  as credibility-building technical content?
- What review gate signs off public claims before publication?
- Which community surfaces are desired first: GitHub Discussions/issues, mailing list,
  Discord/Slack, blog/RSS, changelog, docs site, or benchmark dashboard?
- How should agent-facing engagement be exposed without turning private handoff data into
  public product copy?

## Next Planning Step

Assign a website/community planning owner to define the public information architecture,
benchmark publication contract, claim-review policy, and privacy boundary before any
implementation or hosting work begins. The first plan should map every public claim and
automated benchmark field back to a durable fast-mlx source.
