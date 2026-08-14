# fast-mlx

`fast-mlx` is an experimental Swift 6 inference engine and evaluation system for Apple Silicon.
It uses Apple's MLX framework to test inference ideas, measure their real quality and performance
effects, and keep only the capabilities that survive explicit correctness and operational gates.

**Website:** [improvement loop](https://bitworks-io.github.io/fast-mlx/) ·
[operator quickstart](https://bitworks-io.github.io/fast-mlx/quickstart/) ·
[current status](https://bitworks-io.github.io/fast-mlx/status/) ·
[capabilities and evidence](https://bitworks-io.github.io/fast-mlx/capabilities/) ·
[reviewed benchmark explorer](https://bitworks-io.github.io/fast-mlx/benchmarks/) ·
[reviewed releases](https://bitworks-io.github.io/fast-mlx/releases/) ·
[research notes](https://bitworks-io.github.io/fast-mlx/research/)

The project is built around a guarded improvement loop:

1. research a concrete inference technique;
2. state the user outcome and failure boundaries;
3. add a failing contract or bounded experiment;
4. implement the smallest candidate;
5. measure correctness, quality, capacity, and performance;
6. independently review the evidence; and
7. promote the capability, shelve it, or publish the negative result.

"Self-improving" here does **not** mean unrestricted self-modifying software. Research and agent
work remain reviewable, tests are fail-closed, and only verified source and public-safe evidence
are published.

## What exists today

- an OpenAI-compatible chat-completions HTTP/SSE server;
- an explicit continuous-batching route for supported dense models;
- exact prefix/session-cache and serving lifecycle controls;
- a measurement harness for exactness, quality drift, throughput, capacity, and soak behavior;
- capacity planning and proof-control command-line tools; and
- dated technical notes that publish both successful and negative experiments.

The engine is still experimental. A capability appearing in source does not make it a supported
default: the project distinguishes implemented, verified, promoted, shelved, and diagnostic-only
states.

The generated [current-status dashboard](https://bitworks-io.github.io/fast-mlx/status/) collects
those reviewed feature states, measured proof points, release records, research counts, and the
unchanged runtime/model boundary in one static page. It is a present-state reader, not a roadmap,
live telemetry surface, benchmark runner, or second source of publication authority.
Each capability card links to a canonical detail permalink that repeats only the reviewed state,
scope, evidence paths, and claim boundary for that one record.

## First run

Requirements:

- Apple Silicon Mac;
- macOS 14 or newer; and
- Swift 6.

The transport-only backend exercises the public HTTP surface without loading a model:

```sh
swift run --package-path spike fastmlx-serve --scripted
```

Then, in another terminal:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"fastmlx-scripted","messages":[{"role":"user","content":"hello"}],"temperature":0,"n":1,"stream":false}'
```

Loaded-model serving requires a source-locked local model directory and explicit memory, cache,
and reserved-KV limits. Inspect the exact contract before attempting it:

```sh
swift run --package-path spike fastmlx-serve --help
```

No model weights are bundled with this repository.

## Research notes and evidence

The [research-note library](docs/content/README.md) records the investigation arc, including wrong
hypotheses and useful failures. The public website is generated from an explicit reviewed manifest;
unreviewed operator evidence, machine-local paths, private competitor analysis, and partial runs are
excluded from both the site and the public repository projection.

The release page,
[`/releases/index.json`](https://bitworks-io.github.io/fast-mlx/releases/index.json), and
[`/releases/feed.atom`](https://bitworks-io.github.io/fast-mlx/releases/feed.atom) are generated
from `site/releases.json`, a reviewed public release ledger. The Atom feed is a static subscription
surface for the same newest-first entries; it performs no network fetch or external ingestion while
building. These surfaces are discoverability metadata for public milestones and unchanged
boundaries; they do not grant runtime authority, publish new benchmark claims, or replace the
capability/evidence review gates.

[`/research/index.json`](https://bitworks-io.github.io/fast-mlx/research/index.json) and
[`/research/feed.atom`](https://bitworks-io.github.io/fast-mlx/research/feed.atom) are generated
from the seven explicitly reviewed notes in `site/publications.json`. Each research-feed entry
contains only pinned titles, dates, themes, summaries, and canonical article links—never article
bodies, external intake, scripts, trackers, or a build-time network request.

The [research archive](https://bitworks-io.github.io/fast-mlx/research/) progressively adds local
title/summary/theme search and an exact-theme filter while leaving every reviewed note visible
without JavaScript. Filter state is bounded and shareable in the URL; it never fetches content,
reorders notes, stores user data, admits a new article, or creates publication authority.

[`/feed.atom`](https://bitworks-io.github.io/fast-mlx/feed.atom) combines those two reviewed
streams into one newest-first subscription without making the generated feed a source of truth.
Entries retain their stable release-commit or canonical-article IDs, carry an explicit release or
research category, and remain text-only. The combined feed performs no external fetch, automatic
publication, benchmark recomputation, ranking, or authority transition.

[`/sitemap.xml`](https://bitworks-io.github.io/fast-mlx/sitemap.xml) inventories only the reviewed
human-facing pages, and [`/robots.txt`](https://bitworks-io.github.io/fast-mlx/robots.txt) points
crawlers to that canonical map. They are deterministic discovery hints—not an indexing guarantee,
an external-content intake path, or a second publication authority.

Each of the 38 reviewed HTML pages also publishes one self-referential absolute canonical URL and a
reviewed Open Graph description. The seven research notes alone use article metadata; the nine
product/index pages, six capability-detail pages, three benchmark-detail pages, and thirteen
release-detail pages remain website objects, while `404.html` and machine-readable endpoints publish
neither. Detail pages are immutable views of already-reviewed capability records, benchmark
evidence, or release-ledger entries; they do not create new evidence, support, measurement, runtime,
model, acquisition, admission, authority, ranking, or recomputation. A single same-origin 1200×630
preview image is retained as an exact hash-pinned static asset: no remote image, analytics request,
live lookup, new benchmark claim, or runtime authority is introduced by a shared-link preview.

Read [PUBLICATION.md](PUBLICATION.md) for the public boundary and
[CONTRIBUTING.md](CONTRIBUTING.md) for the research-to-release workflow.

## Repository status

This checkout remains the engineering source of truth. A fail-closed export creates the public
repository from a reviewed allowlist instead of publishing the entire operator workspace or its
history. GitHub Pages deployment and public-source checks run from the exported repository.

## License

fast-mlx is licensed under the [Apache License 2.0](LICENSE). Commercial use and proprietary
extensions are permitted subject to that license and the notices in [NOTICE](NOTICE). Third-party,
vendored, and dataset-derived material retains its own license and provenance; see
`spike/Vendor/mlx-swift-lm/FAST_MLX_UPSTREAM.md` and
`spike/Tests/HarnessCoreTests/Fixtures/GSM8K-LICENSE`.
