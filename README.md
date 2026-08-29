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

Requirements: an Apple Silicon Mac, macOS 14 or newer, and Xcode or the Xcode command-line tools
(for the Swift 6 toolchain). Nothing else — the MLX Metal library ships prebuilt in this
repository (`spike/prebuilt/mlx.metallib`), so you do **not** need to install Apple's Metal
Toolchain component or build through Xcode. (SwiftPM cannot compile MLX's Metal kernels itself, and
on macOS 26 the Metal compiler is a separate download; shipping the prebuilt metallib keeps a fresh
checkout runnable with one command.)

### Serve a model (one command)

```sh
# fast-mlx fetches the model for you — just name a Hugging Face repo:
./scripts/serve.sh --model mlx-community/Qwen3-8B-4bit
```

That builds `fastmlx-serve`, downloads the model on first run (cached afterward), colocates the
shipped metallib, and serves on `127.0.0.1:8080`. Already have the weights locally? Pass
`--model-path` instead — it wins over auto-fetch:

```sh
./scripts/serve.sh --model-path ./my-model-dir --model my-name
```

`serve.sh` derives sensible MLX memory limits from your machine's RAM — pass explicit
`--memory-limit-bytes` / `--host` / `--port` to override. Set `FASTMLX_API_KEY` to require Bearer
auth (mandatory for a non-loopback `--host`).

Call it with the standard OpenAI shape, including tools:

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model":"qwen3-8b",
  "messages":[{"role":"user","content":"Do you have the RTX 6000 Ada in stock?"}],
  "tools":[{"type":"function","function":{"name":"get_product","description":"Look up a product",
    "parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}],
  "enable_thinking":false
}'
```

The assistant replies with an OpenAI `tool_calls` message (`finish_reason: "tool_calls"`); send the
tool result back as a `{"role":"tool","tool_call_id":…,"content":…}` message to continue.

Completion length is model- and host-fit-aware. The default request budget is 4,096 tokens, but it
is not a global maximum: when `--max-completion-tokens` is omitted, the loaded model's authenticated
context and the pre-load host-fit decision determine the ceiling. Use
`--default-completion-tokens`, `--max-completion-tokens`,
`--max-non-streaming-completion-tokens`, and `--completion-limit-policy reject|clamp` to narrow or
shape the policy. Long completions above the non-streaming ceiling remain available with
`"stream":true`. The request-body ceiling scales with the admitted context (1–64 MiB by default),
and `--max-request-body-bytes` plus `--max-non-streaming-response-bytes` provide explicit transport
overrides. Authenticated clients can inspect every effective token and byte limit at
`GET /v1/models` instead of guessing from a model name.

### Transport-only (no model)

```sh
swift run --package-path spike fastmlx-serve --scripted
```

No model weights are bundled with this repository. If you change the `mlx-swift` pin in
`spike/Package.swift`, regenerate the shipped metallib with `scripts/build-metallib.sh` (it
installs Apple's Metal Toolchain component on first use, no sudo required).

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
from the 24 explicitly reviewed notes in `site/publications.json`. Each research-feed entry
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

Each of the 45 reviewed HTML pages also publishes one self-referential absolute canonical URL and a
reviewed Open Graph description. The 24 research notes alone use article metadata; the ten
product/index pages, seven capability-detail pages, three benchmark-detail pages, and fifteen
release-detail pages remain website objects, while `404.html` and machine-readable endpoints publish
neither. Detail pages are immutable views of already-reviewed capability records, benchmark
evidence, or release-ledger entries; they do not create new evidence, support, measurement, runtime,
model, acquisition, admission, authority, ranking, or recomputation. A single same-origin 1200×630
preview image is retained as an exact hash-pinned static asset: no remote image, analytics request,
live lookup, new benchmark claim, or runtime authority is introduced by a shared-link preview.

Read [PUBLICATION.md](PUBLICATION.md) for the public boundary and
[CONTRIBUTING.md](CONTRIBUTING.md) for the research-to-release workflow.

## Repository status

fast-mlx is published as an Apache-2.0 source distribution through a reviewed, fail-closed
allowlist. In the public repository, the checkout includes that manifest, exporter, and their
tests, so a committed clone whose index matches the reviewed manifest can reproduce and validate
the same public boundary locally. The public identity manifest pins the complete tracked path/mode
set, so a newly tracked file cannot silently expand a whole-tree allowlist:

```sh
python3 scripts/export_public_repository.py --output /fresh/path/fast-mlx-public
python3 scripts/validate_public_repository.py /fresh/path/fast-mlx-public
```

The destination must be absent or empty and outside this checkout. The exporter reads only Git's
index; unstaged files and content outside the allowlist cannot enter the candidate. A broader
development workspace can contain private or operator-only material and must not be published
wholesale. GitHub Pages deployment and public-source checks run from the projected public
distribution.

## License

fast-mlx is licensed under the [Apache License 2.0](LICENSE). Commercial use and proprietary
extensions are permitted subject to that license and the notices in [NOTICE](NOTICE). Third-party,
vendored, and dataset-derived material retains its own license and provenance; see
`spike/Vendor/mlx-swift-lm/FAST_MLX_UPSTREAM.md` and
`spike/Tests/HarnessCoreTests/Fixtures/GSM8K-LICENSE`.
