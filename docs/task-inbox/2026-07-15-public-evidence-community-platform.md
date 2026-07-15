---
status: captured
type: product-platform
priority: parallel-product-lane
created: 2026-07-15
source: user-plus-sol-review
planning_ready: true
implementation_ready: false
---

# Public evidence, community, benchmark automation, and release platform

## Raw Capture

> Do a review of omlx.ai website, the community engagement and benchmarks is great, along with
> the auto update mechanism, be sure to roadmap those features. I will reserve a domain so
> fast-mlx can have a website.
>
> Use it as one of the references to determine how fast mlx is performing, relatively speaking.
>
> Record an artifact for another set of agents/subagents to spawn and develop the website,
> automated benchmarking collection, and any other engagement activities the Sol reviewer
> determines would add user or agent value.

## Decision context

The dated
[`oMLX public ecosystem review`](../reference/2026-07-15-omlx-public-ecosystem-review.md) is the
research input. The core decision is to reproduce oMLX's strongest loop—easy install, visible
proof, community measurement, shareable comparisons, active releases—while making fast-mlx's
measured speed-versus-loss dial and dated verdicts the differentiator.

This is a **parallel product/platform lane**, not a replacement for the performance flywheel in
`docs/agent-handoff.md`. Do not take the active bench Mac away from a promotion gate. A future M3
Ultra 256 GB node should expand the matrix as a separately labeled hardware tier; measurements
from unlike machines must never be pooled into one comparison cell.

The current repository has no site, public-ingest service, CI publication workflow, application
updater, or unified public evidence schema. `HarnessCore` contains strong specialized evidence,
but the generic benchmark row lacks an engine identifier, schema version, run count, and
dispersion; raw provenance may contain local model paths. Existing evidence formats are
heterogeneous. A public-safe projection is therefore a prerequisite, not a frontend detail.

## User and operator stories

### Prospective user

A Mac user should be able to discover fast-mlx, identify the useful model/dial point for their
hardware, understand the measured quality cost, compare reproducible performance, install a
trusted build, and configure an agent without reverse-engineering benchmark logs.

### Existing user

An existing user should be able to receive a verified stable/RC/dev update, see exactly what
changed, run a local benchmark, deliberately choose whether to publish its privacy-reviewed
payload, and manage or delete their submissions.

### Maintainer and research agent

A maintainer or agent should be able to schedule benchmark matrices across named hardware nodes,
validate clean-SHA artifacts, publish a signed/redacted projection, detect regressions, compare
the same workload against oMLX/MLX-LM, and consume the resulting catalog through a versioned API.

### Community participant

A participant should be able to ask questions, propose models or techniques, share a permalink,
report a reproducible failure, and receive credit in release/verdict notes without implying that
the engine source or contribution policy is necessarily open.

## Acceptance signals

1. A reserved fast-mlx domain serves a responsive, accessible site over TLS with Home, Download,
   Benchmarks, Compare, Verdicts, Models, Docs, Releases, Community, Methodology, Privacy, and
   Security routes; no third-party analytics or external asset CDN is enabled by default.
2. A clean-SHA benchmark artifact can be validated and projected into a versioned public schema
   without secrets, usernames, hostnames, absolute paths, model license material, or stable device
   identifiers, then published idempotently with an integrity signature.
3. Every displayed row has a visible trust level—VERIFIED LAB, REPRODUCED EXTERNAL, COMMUNITY, or
   EXTERNAL REFERENCE—and enough provenance to decide whether two rows are comparable.
4. The Compare surface plots throughput/latency **and** teacher-forced/task quality for lossy dial
   tiers, shows run count and dispersion, preserves the coherence/garbage floor, and refuses to
   declare a winner for mismatched protocols.
5. An automated runner schedules release, nightly/regression, model-onboarding, and explicit
   competitor-reference matrices without overlapping exclusive bench leases or mixing hardware
   tiers. A failed/dirty/stale run cannot publish as VERIFIED LAB.
6. oMLX and MLX-LM can run on the same physical box, exact model, prompt tokens, cache state, and
   trial protocol as fast-mlx. The first result is labeled REPRODUCED EXTERNAL and includes output
   correctness/quality boundaries; the public oMLX site remains EXTERNAL REFERENCE context.
7. Community upload is off until explicit informed consent. The UI previews the exact payload,
   describes retention and moderation, avoids a hardware UUID derivative, supports revocation and
   deletion, and treats upload failure as independent from benchmark success.
8. A stable/RC/dev release can be built from a reviewable pipeline, signed, notarized, accompanied
   by signed checksums/attestations and release notes, then installed atomically by an updater that
   rejects tampering, wrong Team ID, failed notarization, downgrade, and interrupted install while
   preserving the prior working version.
9. Versioned benchmark JSON, signed downloadable artifacts, RSS/Atom release and verdict feeds,
   `llms.txt`, and agent/client configuration snippets expose the same canonical facts as the
   human site.
10. GitHub Discussions or an equivalent community surface has Announcements, Q&A, Ideas/Models,
    Benchmark Reproduction, Show and Tell, and Security-routing guidance; release notes credit
    reporters, reproduction contributors, and benchmark donors where consent permits.
11. Every verified benchmark permalink provides a versioned workload manifest and copyable
    “reproduce this row” command. Packs cover cold start, warm model, multi-turn prefix/session
    reuse, and task correctness rather than only passive throughput rows.
12. A versioned client-compatibility manifest is the single source for API version, streaming,
    tools, structured output, session semantics, authentication mode, and supported client
    versions; the site, docs, config generator, and release pipeline consume it without drift.

## Happy and failure paths

**Happy path:** a release-tagged clean SHA enters an available hardware-node lease, runs its exact
and lossy matrices, signs a public-safe evidence bundle, publishes verified rows and a frontier,
generates release/verdict content, and becomes available through a verified update channel. A user
can inspect the proof, install, optionally submit a local result, and share its permalink.

**Required failure/recovery paths:**

- dirty SHA, missing provenance, stale protocol, incomplete trials, failed exactness predicate,
  missing teacher-forced quality, or coherence-floor failure quarantines the artifact;
- incomparable hardware/model/prompt/cache/runtime cells render side by side only with an explicit
  “not comparable” reason and never produce a winner badge;
- node disconnect, thermal limit, watchdog, cancellation, or expired lease preserves logs but
  cannot publish a partial row as verified;
- duplicate/replayed/community-abusive submissions are idempotently deduplicated or quarantined;
- deletion/revocation removes public ownership linkage while preserving only documented aggregate
  data consistent with the privacy policy;
- unavailable community services do not block local inference, benchmarks, or updates;
- signature, checksum, Team ID, notarization, channel, downgrade, disk-space, or atomic-swap
  failure leaves the installed application usable and provides a recovery path;
- a website/content mismatch fails publication because compatibility and release facts come from
  one canonical manifest.

## Trust and data contract

### Public evidence schema v1 minimum

The schema owner must define and test at least:

- record/schema/protocol versions and trust level;
- engine ID, semantic version, git SHA, dirty flag, build mode, runtime/MLX versions, feature flags,
  dial tier, cache/speculation/sampler/batching state;
- hardware family/variant/GPU cores/memory and OS, with no hostname or stable machine identifier;
- exact model repository/revision plus config/checkpoint/tokenizer/template hashes and quantization;
- prompt/corpus/token-ID digest, context/generation lengths, concurrency, cold/hot cache state;
- versioned agent-workload pack ID, reproduction-manifest digest, and public-safe reproduce command;
- trial/warmup counts, order/randomization, timestamps, thermal/power policy, and failure counts;
- PP/TG, TTFT/TPOT/E2E, aggregate throughput, memory, energy when available, median/dispersion and
  raw-trial attachment digest;
- exactness predicates for exact paths and teacher-forced KL/perplexity/task/coherence metrics for
  lossy paths, including the non-garbage floor and dominance decision;
- source artifact digest, validator version, signer/key ID, publication time, and supersession or
  retraction state.

Raw harness records remain internal evidence. A tested projection layer—not ad hoc string
replacement—must remove local paths and private fields before signing or upload.

### Trust levels

- **VERIFIED LAB:** clean-SHA fast-mlx run on an owned node, complete predicates, validated and
  signed public projection.
- **REPRODUCED EXTERNAL:** competitor engine run by fast-mlx under the same-box protocol.
- **COMMUNITY:** explicit opt-in self-report that passes schema, sanity, dedupe, and abuse checks;
  never silently upgraded to verified.
- **EXTERNAL REFERENCE:** upstream/site result with source URL and retrieval date; cannot generate
  a comparative winner.

## Sol-ranked workstreams and spawn packets

The integrating owner should spawn read-only research/review in parallel, then serialize shared
contracts before parallel frontend/backend implementation. Each worker must return a compact
file-referenced handoff and must not edit another workstream's files without coordination.

### P0 — unblock trustworthy public proof

#### W0: product, ownership, and architecture decision

- **Agent:** most capable product/architecture reviewer; read-only until decisions are accepted.
- **Owns:** domain/repository/hosting boundary, commercial-versus-public participation policy,
  threat model, data classification, service budget, rollout/rollback plan.
- **Inputs:** platform design, this task, oMLX review, existing external model-catalog ownership.
- **Output:** accepted design/spec and ADRs; site repo ownership; canonical release manifest; and
  the initial versioned client-compatibility manifest owned by the product/release authority.
- **Gate:** domain, host, code/community policy, privacy controller, and signing authority selected.

#### W1: public evidence contract and validator

- **Agent:** most capable HarnessCore/data-contract implementer.
- **Write scope:** versioned pure/Foundation schema, projection/validation/signing tools and tests;
  no MLX state.
- **Output:** schema v1, fixture corpus, redaction/property tests, artifact validator, migration and
  revocation model, CLI that emits deterministic public JSON.
- **Proof:** secret/path fixtures are rejected; dirty/incomplete evidence fails closed; valid
  projections reproduce byte-for-byte and verify signatures.
- **Depends on:** W0 data policy. **Unblocks:** W2a, W3, W4a, and W5.

#### W2a: comparison protocol and bounded same-box proof

- **Agent:** most capable harness/MLX-coupled owner; bench execution only on leased nodes.
- **Write scope:** protocol plus runner/adapters for fast-mlx, oMLX, and MLX-LM; output through W1
  only. The P0 execution is one bounded canonical same-box smoke, not the full context matrix.
- **Protocol:** exact checkpoint and prompt token IDs; one canonical 4K cell; fixed generation;
  temperature zero; cold/hot split; repeated trials in a seeded, counterbalanced three-engine
  order (for example, a Latin-square rotation that gives each engine each ordinal position);
  thermal policy; one common in-process or real-client boundary for every engine;
  TTFT/PP/TG/TPOT/E2E/RSS; output/quality checks. Never compare a direct harness timer with a
  network-service timer.
- **Proof:** a deliberately mismatched cell refuses comparison; a same-box smoke emits three
  distinct trust-labeled records with dispersion and a copyable reproduction manifest.
- **Depends on:** W1. End-to-end client comparison also depends on the production serving-route
  task; before then, publish only genuinely common-boundary adapter results. Keep M5 Max and future
  M3 Ultra 256 GB results separate.

#### W3: website and public evidence explorer

- **Agents:** UX/information-architecture worker plus frontend implementer with disjoint design and
  application write scopes.
- **Routes:** the acceptance route list plus benchmark permalink and status/retraction pages.
- **Key interaction:** user chooses hardware/model/workload and moves the optimization dial; the
  graph updates speed, memory/capacity, teacher-forced/task loss, coherence-floor status, evidence
  date, and provenance. Incomparable rows explain why.
- **Proof:** accessibility, responsive, visual, link, metadata/SEO, privacy, performance, empty,
  loading, stale, retracted, and API-failure checks. Self-host assets; strict CSP.
- **Depends on:** W0, W1. Static fixtures may unblock UI work before live ingestion.

#### W4a: validated verified-lab publisher

- **Agent:** backend/data-publication owner; separate security review required.
- **Owns:** validator invocation, public artifact store, signed/idempotent publisher, retention and
  retraction state, audit log, and one explicit/manual single-node publication flow.
- **Proof:** stale/partial/invalid artifacts quarantine; retry does not duplicate rows; a signed
  published digest matches its source; retraction is visible. Do not build a generalized fleet
  scheduler in P0.
- **Depends on:** W0, W1. Integrate W2a records after their contract passes.

#### W7a: minimum feedback and reproduction loop

- **Agent:** community/product-ops owner; coordinate public/commercial boundaries with W0.
- **Owns:** Discussions or equivalent categories for Announcements, Q&A, Ideas/Models, Benchmark
  Reproduction, and Show and Tell; support/reproduction templates; model requests; security
  routing; reporter/benchmark-donor credit consent; moderation owner and response expectations.
- **Proof:** a user can report one result or failure with the required reproduction fields, route a
  private security issue safely, and receive explicit credit consent. This surface does not imply
  an open-source license or code-contribution right.
- **Depends on:** W0 policy only; it can start before the site and evidence API.

### P1 — distribution and participation

#### W2b: complete competitor and agent-workload matrix

- Expand W2a to 1K/4K/16K/32K/64K, cold and hot cache, concurrency, and named agent-workload
  packs for cold start, warm model, multi-turn prefix/session reuse, and task correctness.
- Attach a versioned public-safe manifest and copyable “reproduce this row” command to every
  verified permalink; preserve the common-boundary and non-comparability rules.
- Run by explicit budget and available node lease. Do not delay the first trustworthy site on this
  full matrix. Keep every hardware tier separate.
- **Depends on:** W1, W2a; end-to-end cells depend on the serving route.

#### W4b: benchmark orchestration and regression collection

- **Agent:** backend/infrastructure owner; separate security review required.
- **Owns:** node registry, exclusive leases, job matrix, queues/retries/watchdogs, regression
  alerts, recovery, and scheduled collection through W4a's validator/publisher.
- **Triggers:** release candidate, model/dial onboarding, explicit competitive run, nightly bounded
  regression; expensive long-context/soak cells run by budget and gate, not on every commit.
- **Proof:** simultaneous jobs cannot claim one node; node loss recovers without publishing a
  partial row; retry remains idempotent; M5 Max and M3 Ultra records cannot enter one cell.
- **Depends on:** W2b, W4a. Generalized fleet behavior becomes valuable as the second node arrives.

#### W5: community submission and moderation

- **Agents:** privacy/security designer first, then API and app-UX workers with disjoint scopes.
- **Owns:** informed opt-in, payload preview, anonymous submission token, deletion/revocation,
  retention, rate limits, abuse/moderation/dedupe, COMMUNITY-only trust label.
- **Proof:** no network call before consent; decline persists; payload snapshot test excludes stable
  IDs/paths; delete/revoke works; replay and abuse quarantine; outage leaves local results intact.
- **Depends on:** W0, W1, W3.

#### W6: signed release and automatic update channel

- **Agent:** macOS release/security owner; use the most capable model and an independent reviewer.
- **Owns:** reproducible/reviewable build, stable/RC/dev feeds, Developer ID signing, notarization,
  Ed25519/Sparkle archive signatures, checksums/attestations, phased rollout, atomic install,
  rollback, release notes, Homebrew metadata.
- **Preferred baseline:** Sparkle 2 unless design review proves a safer maintained alternative.
- **Proof:** valid stable/RC/dev update; tampered archive, wrong key/Team ID, failed notarization,
  downgrade, disk exhaustion, interruption, and offline checks all fail safely.
- **Depends on:** W0 release/client-compatibility manifests and an actual app/distribution target.
  Do not copy the oMLX custom swap path or strip quarantine as a trust substitute.

#### W7b: generated community and agent surfaces

- **Agent:** docs/community/product-ops owner; extend W7a and coordinate policy with W0.
- **Owns:** benchmark donation guide, release/verdict template, public roadmap view, showcase, and
  generated compatibility/evidence surfaces.
- **Agent-facing outputs:** versioned JSON API, `llms.txt`, RSS/Atom for releases/verdicts/regressions,
  machine-readable model/dial catalog, copyable agent configuration generator, schema/examples,
  and a “latest verified evidence” endpoint.
- **Compatibility-manifest role:** consume the W0-owned versioned client-compatibility manifest for
  API version, streaming, tools, structured output, session semantics, authentication mode, and
  supported client versions; propose reviewed changes to its product/release owner. W3 consumes it
  for pages/config generation and W6 for release validation. No surface keeps a hand-written copy.
- **Proof:** a person and an automated agent can independently find the same current compatibility,
  release, benchmark, and verdict facts; stale pages fail generation rather than drift.
- **Depends on:** W0, W1, W3.

### P2 — expand after the proof loop works

#### W8: content automation and ecosystem growth

- Normalize front matter in `docs/content/`, generate the site index/whitepaper feed, add editorial
  review and source-status checks, and publish dated negative verdicts as first-class learning.
- Add localization only after canonical content is generated from shared metadata.
- Add case studies, benchmark-donor recognition, newsletter/social syndication, and public roadmap
  voting only after moderation, privacy, and evidence freshness are operational.
- Publish a monthly light competitive scan and quarterly deep Sol audit; a material upstream
  release or unexplained regression triggers an event-driven review.

## Dependency and integration order

```text
W0 architecture/policy
 ├─ W1 public evidence contract
 │   ├─ W2a bounded same-box proof ── W2b full matrix ─┐
 │   ├─ W3 site/explorer                              ├─ W4b orchestration
 │   ├─ W4a validated publisher ──────────────────────┘
 │   └─ W3 + W1 ── W5 community ingest
 ├─ W7a minimum feedback loop
 └─ W6 signed release/updater

W3 + W7a → W7b generated agent/community surfaces
W3 + W4b + W7b → W8 content/ecosystem expansion
```

W0 and W1 are the load-bearing contracts. W3 may build against signed fixtures while W2a/W4a run
in parallel, and W7a can begin once participation policy is explicit. The full W2b/W4b matrix and
fleet automation follow the first honest public proof. W5 and W6 cross privacy/supply-chain
boundaries and require focused independent review before any external rollout.

## What not to build or copy

- no automatic public upload after a benchmark and no hardware-derived stable owner identifier;
- no custom self-updater whose only trust assertion is a claimed notarized DMG;
- no “fastest” badge from mismatched models, contexts, cache state, or metric definitions;
- no leaderboard that mixes lab, competitor, community, and external rows without trust labels;
- no public projection of the current raw `modelPath` or machine-local provenance;
- no externally hosted fonts/scripts or behavioral analytics by default;
- no publication-count growth goal without dedupe, moderation, retention, and deletion;
- no community feature that implies an open-source license or code-contribution right unless W0
  explicitly changes the current product policy.

## Decisions required before implementation

1. Reserved domain and DNS owner; site repository and hosting provider.
2. Public/commercial licensing boundary and whether GitHub Discussions is available on a suitable
   public repository or needs a separate forum.
3. Privacy controller/contact, retention/deletion policy, moderation owner, and jurisdictions.
4. Apple Developer/Developer ID, notarization, update-signing, artifact-signing, and key-custody
   owners; secrets remain outside the repository.
5. Stable/RC/dev release policy, supported macOS/hardware matrix, and canonical manifest format.
6. Benchmark service budget, artifact retention, node names/credentials in the private overlay,
   and when the M3 Ultra 256 GB machine becomes available.
7. Whether the existing external model catalog remains authoritative or migrates behind a shared
   versioned API consumed by Concierge and the fast-mlx site.

## Next safe action

After the domain and ownership decisions are supplied, run W0 as a short design cycle and W1 as the
first implementation plan. In parallel, W3 may create low-fidelity information architecture and
fixture-backed dial/benchmark prototypes, while W7a opens the policy-bounded feedback loop. Do not
deploy ingestion, analytics, updater, or public benchmark data until their privacy/security gates
pass.
