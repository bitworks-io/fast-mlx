# oMLX public ecosystem review — patterns for fast-mlx

> **Private internal reference — owner policy updated 2026-07-18.** The external measurements and
> comparative analysis in this artifact are engineering litmus only and must not be published as
> fast-mlx product copy, benchmark rows, website comparisons, feeds, or release claims. Public
> surfaces report fast-mlx's own reproducible results. Any older “publishable comparison,”
> `REPRODUCED EXTERNAL`, or cross-engine site recommendation below is superseded; a same-box
> competitor lane may continue privately to inform priorities.

- **Date checked:** 2026-07-15
- **Scope:** public website, benchmark explorer and submission path, distribution/update path,
  release/community loop, and use of oMLX data as a fast-mlx performance reference
- **Method:** official oMLX website, repository, source, releases, and GitHub surfaces; official
  Sparkle documentation for updater guidance; local clean-SHA fast-mlx verdicts for fast-mlx
  measurements
- **Decision:** adopt the public proof and participation loop, strengthen its provenance and
  consent boundaries, and do not make a cross-engine winner claim until a same-box protocol runs

## Executive finding

oMLX has turned benchmarks into a product surface rather than leaving them in release notes. Its
homepage leads with a clear agent-workload problem, the community runner continuously feeds a
searchable leaderboard, the explorer converts submissions into context-length distributions, and
shareable result pages make the measurements useful in support and discussion. Releases are easy
to discover and install, while detailed notes credit reporters and contributors. That is the right
kind of flywheel for fast-mlx to learn from.

fast-mlx should make its own wedge more explicit: not merely “fast,” but **the fastest useful point
the user chooses on a measured speed-versus-loss frontier**. Its website should publish both sides
of every dial point, retain dated PROMOTE/SHELVE verdicts, and separate verified lab results from
community and external-reference data.

The current numbers place fast-mlx in the same single-request Qwen3-32B performance band as public
oMLX M5 Max rows. They do **not** establish that fast-mlx is faster. The prompt lengths, model
revision, runtime flags, cache state, run statistics, and metric boundaries are not aligned. oMLX
is therefore a useful reference corpus and direct-comparison target, not a substitute for the
same-box fast-mlx harness.

## What oMLX does especially well

### 1. A focused discovery-to-install funnel

**CONFIRMED.** The [oMLX homepage](https://omlx.ai/) moves from an outcome-led agent-use-case
headline and DMG call to action through product screenshots, capability cards, hardware-specific
benchmarks, FAQ, and installation. Navigation exposes the
[Performance Explorer](https://omlx.ai/compare),
[Community Benchmarks](https://omlx.ai/benchmarks), GitHub, and download instead of hiding proof
behind a generic documentation hierarchy.

The reusable pattern is:

1. say whose latency or capacity problem is being solved;
2. show the product and the shortest path to a first successful request;
3. prove claims on named hardware, model, context, and concurrency;
4. expose the full measurement corpus and methodology;
5. let a user install, compare, report, and return for updates.

### 2. Benchmarks as a community product

**CONFIRMED.** The [leaderboard](https://omlx.ai/benchmarks) showed approximately 332,000 rows
when checked. The count is volatile and should not be read as 332,000 independent machines or
users. Rows expose RAM, while the controls support chip/variant, model, quantization, context,
prompt-processing and token-generation filters; sortable results; pagination; and per-result
permalinks. The
[explorer](https://omlx.ai/compare) adds shareable filtered comparisons and distributions across
context lengths rather than presenting only a top score.

**CONFIRMED.** A result page such as
[this public example](https://omlx.ai/benchmarks/ar1t43c7) exposes chip, RAM, GPU cores, oMLX and
macOS versions, model label, quantization, context, TTFT, prompt processing, token generation, and
peak memory. That makes a result much easier to discuss than an image pasted into a chat.

**INFERRED LIMIT.** Public result pages do not visibly identify an exact model repository and
revision, checkpoint/config hashes, engine flags, custom-kernel state, cache state, benchmark
protocol version, raw trials, or verification/outlier disposition. The corpus is excellent for
discovery and weak-to-moderate comparative evidence; it is not enough on its own for a reproducible
winner claim.

### 3. Releases close the feedback loop

**CONFIRMED.** The [v0.5.1 release](https://github.com/jundot/omlx/releases/tag/v0.5.1) combines
upgrade-relevant fixes, before/after observations, linked issues and pull requests, reporter
credit, contributor credit, install artifacts, and a full changelog. GitHub
[Discussions](https://github.com/jundot/omlx/discussions) provides Announcements, Ideas, Q&A,
Show and Tell, Polls, and General categories, while the
[contribution guide](https://github.com/jundot/omlx/blob/main/docs/CONTRIBUTING.md) covers setup,
tests, contribution areas, and pull-request flow.

This is a high-value engagement loop even if fast-mlx remains commercially licensed: users still
need announcements, support/Q&A, benchmark-method discussion, model requests, reproducible bug
reports, and credit for useful evidence. Code-contribution permissions can remain a separate
policy decision.

### 4. Distribution meets users where they are

**CONFIRMED.** The [oMLX README](https://github.com/jundot/omlx/blob/main/README.md#install) offers
DMG, Homebrew, and source installation. Its updater checks GitHub Releases every 24 hours, supports
stable/RC/dev channels, selects an OS-compatible DMG, and displays release notes; see
[ReleasesChecker.swift](https://github.com/jundot/omlx/blob/main/apps/omlx-mac/Sources/Updater/ReleasesChecker.swift)
and
[UpdateController.swift](https://github.com/jundot/omlx/blob/main/apps/omlx-mac/Sources/Updater/UpdateController.swift).
The channel and release-note experience is worth adopting.

## What fast-mlx should not copy unchanged

### Automatic benchmark publication and stable device identity

**CONFIRMED.** oMLX uploads every eligible completed local benchmark after the run; experimental
feature and external-endpoint runs are skipped. The payload includes system/model/measurement
fields and an owner hash derived from the Mac's IOPlatform UUID and hardware attributes. See the
[upload implementation](https://github.com/jundot/omlx/blob/main/omlx/admin/benchmark.py#L677-L815)
and
[automatic post-run call](https://github.com/jundot/omlx/blob/main/omlx/admin/benchmark.py#L1128-L1131).

**INFERRED FROM THE REVIEWED SURFACES.** The public site does not visibly link a privacy policy,
stable-identifier explanation, retention policy, or deletion workflow. fast-mlx should require
explicit opt-in before the first submission, show the exact payload, avoid a raw or deterministic
hardware UUID derivative, support deletion and revocation, and keep local benchmark success
independent of publication success.

### Updater trust boundary

**CONFIRMED.** The current oMLX updater source says it performs no EdDSA archive-signature check,
uses Apple notarization stapled to the DMG as its trust boundary, mounts the DMG with
`hdiutil -noverify`, swaps the app, removes quarantine, and relaunches it. See
[AppUpdater.swift](https://github.com/jundot/omlx/blob/main/apps/omlx-mac/Sources/Updater/AppUpdater.swift#L1-L8)
and its
[mount/swap path](https://github.com/jundot/omlx/blob/main/apps/omlx-mac/Sources/Updater/AppUpdater.swift#L287-L374).
The repository also states that DMGs come from an off-tree maintainer pipeline in the
[packaging notes](https://github.com/jundot/omlx/blob/main/packaging/README.md#L69-L72).

**UNVERIFIED.** This review did not download the large DMGs to independently validate their code
signature or notarization ticket, so the website's “signed and notarized” statement remains
unverified here.

fast-mlx should use a maintained updater such as Sparkle 2, or an equivalently reviewed design,
with archive signatures, HTTPS, Developer ID signing, notarization, explicit Team ID/code-sign
verification, atomic installation, rollback, and failure-safe UX. The
[official Sparkle security guidance](https://sparkle-project.org/documentation/) recommends
HTTPS, Developer ID/notarization, and EdDSA archive signatures and supports channels, phased
rollouts, deltas, and safe installation.

### Documentation and benchmark drift

**CONTRADICTED.** The [oMLX homepage](https://omlx.ai/) says source installation requires Python
3.10+, while its [current README](https://github.com/jundot/omlx/blob/main/README.md#from-source)
says Python 3.11–3.13; the
[contribution guide](https://github.com/jundot/omlx/blob/main/docs/CONTRIBUTING.md) still says
3.10+. The README also describes the updater as Sparkle-driven while current
[updater source](https://github.com/jundot/omlx/blob/main/apps/omlx-mac/Sources/Updater/ReleasesChecker.swift#L1-L12)
documents a direct GitHub Releases replacement without Sparkle's appcast/EdDSA flow. fast-mlx
should generate website, download, compatibility, and update-channel facts from one versioned
release manifest.

### Mixed-trust numbers and absolute marketing language

The approximately 332,000 community rows are valuable, but volume does not prove independence,
reproducibility, or comparability. fast-mlx should never mix lab, reproduced competitor,
community, and externally sourced rows without visible trust labels. It should avoid universal
latency language such as “always under five seconds” unless the page binds it to an exact cache
hit, model, context, hardware, and percentile.

## Relative performance: what can be said now

The following is a **reference alignment**, not a head-to-head benchmark.

| Source | Hardware/model label | Workload and metric | Published result | Interpretation |
| --- | --- | --- | ---: | --- |
| oMLX community, 2026-07-01 | M5 Max 40-core, 128 GB; Qwen3-32B 4-bit | 4K context, TG | 26.3 tok/s | External self-report; exact checkpoint, flags, trial distribution, and cache state are not public on the row. |
| oMLX community, 2026-07-01 | same label | 16K / 64K context, TG | 21.4 / 12.7 tok/s | Useful context-scaling reference, not yet reproduced by fast-mlx under the same protocol. |
| fast-mlx PLD verdict | M5 Max, 128 GB; Qwen3-32B 4-bit, fp16 KV | 256-token repetitive shape, PLD off / on | 28.28 / 56.70 tok/s | Clean-SHA, three post-warmup runs; the +100.5% PLD result applies to a highly repetitive shape. |
| fast-mlx PLD verdict | same local model | structured code / low-repetition prose, PLD on | 29.31 / 28.66 tok/s | PLD gate preserves base-like speed where speculative reuse is weak. |
| fast-mlx service verdict | M5 Max, 128 GB; Qwen3-32B 4-bit, fp16 KV | common short workload, batch-no-spec C=1 / 2 / 4 / 8 aggregate service throughput | 26.72 / 42.70 / 51.44 / 56.56 tok/s | Includes service queue, prefill, and makespan; not comparable to oMLX TG-only rows. |

The public oMLX rows are available through the
[M5 Max/Qwen3-32B filter](https://omlx.ai/benchmarks?chip_full=M5%7CMax%7C40&model=Qwen3-32B).
The fast-mlx sources are the
[PLD verdict](../superpowers/verdicts/2026-07-09-pld-firstrun.md) and
[continuous-batching verdict](../superpowers/verdicts/2026-07-14-continuous-batching-chunked-prefill.md).

The honest present-tense statement is:

> On the same hardware and model-size label, fast-mlx's approximately 28 tok/s single-request
> rows are in the same band as oMLX's public 4K Qwen3-32B rows at 23.9–26.3 tok/s. The numeric
> lead is promising but methodologically inconclusive. No cross-engine winner has been measured.

### Required direct-comparison lane

A publishable comparison must run fast-mlx, oMLX, and a useful upstream control such as MLX-LM on
the same physical machine with:

- the same exact model directory, repository/revision, config and checkpoint hashes;
- the same prompt token IDs at 1K, 4K, 16K, 32K, and 64K, plus the same generation length;
- one common measurement boundary—either in-process adapters for every engine or a real client/API
  for every engine; never compare a direct fast-mlx harness timer with an oMLX HTTP timer;
- temperature zero, thinking/tool settings fixed, and output bytes or tokens checked where engine
  contracts permit;
- cold-prefix and hot-prefix cells labeled separately; no hidden cache carry-over;
- the same warmup count, measured trial count, thermal/power policy, and run ordering;
- client-observed TTFT, prompt processing, token generation/TPOT, end-to-end latency, aggregate
  service throughput, peak/steady RSS, and failures;
- engine/version/SHA, MLX/runtime, OS, hardware, dial tier, flags, and raw-trial provenance;
- medians, dispersion/error bars, sample counts, and an explicit “not comparable” state when any
  required field differs.

This follows the reproducibility intent in the
[MLCommons inference rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
and the rich provenance practice in
[llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md), while
retaining fast-mlx's stricter exactness and teacher-forced quality contracts.

## Recommended fast-mlx public experience

The first site should expose:

- `/` — the optimization-dial wedge, product screenshot, verified proof, and install CTA;
- `/download` — stable/RC/dev channels, compatibility, signatures, checksums, release notes;
- `/benchmarks` — verified lab results first, then reproduced external and community rows;
- `/compare` — hardware/model/context/concurrency filters and a speed-versus-quality frontier;
- `/verdicts` — dated PROMOTE/SHELVE decisions, including useful negative results;
- `/models` — system-aware capacity and context guidance;
- `/docs`, `/releases`, `/community`, `/security`, `/privacy`, and `/methodology`;
- a versioned benchmark JSON API, downloadable signed artifacts, RSS/Atom feeds, and `llms.txt` so
  people and agents can consume the same facts;
- a versioned agent-workload pack and copyable “reproduce this row” command on every verified
  permalink, including cold start, warm model, multi-turn prefix/session reuse, and task checks;
- a client/config generator for supported agent tools, with commands derived from versioned
  compatibility data—API version, streaming, tools, structured output, session semantics,
  authentication mode, and supported client versions—rather than hand-copied prose.

Every benchmark row should carry one of four visible trust levels:

| Trust level | Meaning |
| --- | --- |
| **VERIFIED LAB** | fast-mlx clean-SHA run, complete provenance, validated predicates, signed public projection. |
| **REPRODUCED EXTERNAL** | another engine run by fast-mlx under the same-box comparison protocol. |
| **COMMUNITY** | explicit opt-in submission that passes schema/abuse checks but is not independently reproduced. |
| **EXTERNAL REFERENCE** | sourced upstream/community evidence with retrieval date and no winner claim. |

The implementation backlog and agent spawn packets are captured in
[`2026-07-15-public-evidence-community-platform.md`](../task-inbox/2026-07-15-public-evidence-community-platform.md).

## Source-review disposition

| Claim family | Status | Publication rule |
| --- | --- | --- |
| oMLX site structure, public filters, explorer, result fields | **CONFIRMED** from official live surfaces | May inform product design; count must retain retrieval date. |
| oMLX automatic upload, owner hash, updater mechanics | **CONFIRMED** from official source | May support design/risk conclusions; re-check before implementation because upstream changes quickly. |
| oMLX signed/notarized release artifacts | **UNVERIFIED** in this review | Do not repeat as independently validated. |
| oMLX community rows as reproducible engine comparisons | **UNVERIFIED / insufficient provenance** | Reference only; do not derive a winner. |
| fast-mlx PLD and service numbers | **CONFIRMED** from local dated clean-SHA verdicts | Publish only with their workload and metric boundaries. |
| “fast-mlx beats oMLX” | **UNVERIFIED** | Prohibited until the direct-comparison lane passes. |
