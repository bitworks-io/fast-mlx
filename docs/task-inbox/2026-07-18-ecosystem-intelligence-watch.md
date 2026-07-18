---
status: active-monitor / captured-intake-automation
type: research-ops-watch
priority: high
created: 2026-07-18
source: user
owner: Sol / fast-mlx
planning_ready: false
implementation_ready: false
---

# Ecosystem intelligence watch

## Raw Capture

Capture a lightweight future-work seed for a recurring public-only ecosystem/competitor
intelligence loop owned by Sol: official GitHub releases/commits/issues for MLX,
MLX Swift/LM, mlx-serve, oMLX, llama.cpp; HF models/papers/leaderboard; arXiv RSS; Apple
MLX signals; GitHub Trending/topics; TLDR AI as weak lead only. Require upstream
corroboration, private internal competitor litmus (performance/feature/moat gaps), dated
PROMOTE/SHELVE entries, novel Sol-proposed ideas, no public direct competitor comparison,
no private code/prompts/metrics sent out, monthly source pruning. Include raw user request
near-verbatim, status/type/priority, cadence as open or proposed, trust notes, and next
planning step; not a full implementation plan.

## Light Triage

The read-only monitoring slice is active as the Codex automation
`fast-mlx-ecosystem-and-competitor-watch`. The repository-side intake ledger, deduplication,
and durable PROMOTE/SHELVE automation remain captured future work, not an implementation plan.

User/operator: Sol / fast-mlx owner, using the existing technique-integration flywheel to
spot credible external signals early and turn only corroborated candidates into measured
fast-mlx gates.

Desired outcome: recurring public-source monitoring that lowers the cost of moving from
"interesting field signal" to a documented internal PROMOTE/SHELVE decision, while retaining
fast-mlx's measured, source-backed standard.

Priority: high as standing discovery hygiene. A signal does not displace the measured roadmap
unless primary evidence supports a concrete Apple/MLX performance, feature, reliability, or
product-surface delta.

Cadence: the active public-only Sol watch runs daily at 08:15 local time. It scans the core
primary sources daily, adds broader discovery feeds on Tuesday and Friday, and performs the
portfolio reconciliation on Friday. Source pruning remains monthly future workflow work.

## Scope Seed

Public sources to monitor:

- Official GitHub releases, commits, and issues for MLX, MLX Swift, MLX-LM, mlx-serve, oMLX,
  and llama.cpp.
- Hugging Face models, papers, and leaderboard movements relevant to local Apple-Silicon LLM
  serving.
- arXiv RSS for inference, speculative decoding, KV/cache compression, quantization, batching,
  structured generation, and Apple/Metal-relevant kernels.
- Apple MLX and Apple Machine Learning Research signals.
- GitHub Trending and topic pages as early weak signals.
- TLDR AI as a weak lead source only; never sufficient by itself.

Internal litmus for each candidate:

- Performance gap: could this change fast-mlx's measured throughput, TTFT, memory, power, or
  scaling frontier on target hardware and workloads?
- Feature gap: does it unlock a user-visible capability such as serving policy, tool/JSON
  reliability, long-context utility, model coverage, or operational resilience?
- Moat gap: does it threaten fast-mlx's quantified-loss dial, reproducibility, Apple fit, or
  measurement credibility?
- Novel Sol-proposed ideas are allowed, but must be labeled as internal hypotheses until
  backed by local evidence or upstream source review.

## Trust Notes

- Require upstream corroboration before opening a task: primary paper, official repository,
  release notes, maintainer artifact, or reproducible benchmark. Aggregators and social posts
  are leads, not evidence.
- Keep all competitor analysis private and internal. Do not publish direct public competitor
  comparison claims from this loop.
- Do not send private code, private prompts, unreleased metrics, customer data, or internal
  strategy to public services or third-party tools during monitoring.
- Preserve dated PROMOTE/SHELVE entries so negative results are not rediscovered as new work.
- Prune sources monthly so low-signal feeds do not become permanent obligations.

## Acceptance Signals

- New leads are captured with source links, date observed, confidence level, and the reason
  they matter to the current Sol/fast-mlx portfolio.
- Every promoted investigation has upstream corroboration and a fast-mlx-specific hypothesis.
- Every shelved lead records the reason and date, especially duplicates, unsupported Apple
  paths, vendor-only claims, or techniques already dominated by a measured local result.
- The loop explicitly separates private competitor assessment from any public-facing content.

## Open Questions

- Where should dated PROMOTE/SHELVE entries live: this inbox, `docs/reference/`, or the
  existing verification/verdict structure?
- What source list is small enough to maintain without crowding out active benchmark work?
- Should durable intake stay review-gated and manual, or become a small script-assisted workflow
  fed by the read-only automation?

## Next Step

Plan the durable repository intake boundary: a minimal schema, deduplication key, evidence
threshold, review gate, and storage location for dated intelligence entries. Reuse the active
read-only automation rather than building another watcher.
