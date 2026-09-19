# fast-mlx

`fast-mlx` measures what a speed or memory setting costs in output quality for LLMs on Apple
Silicon (MLX). It sizes a model against your machine before loading and refuses rather than
degrades, publishes per-model quality cards (teacher-forced KL mean/p95, top-1 agreement, task
checks) that can gate serving admission as an opt-in, and keeps exactness evidence for every
claim. It includes an experimental Swift 6 serving engine that is a research platform, not a
production or fastest-available server.

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

- a pre-load fit-check that sizes a model against the host and refuses rather than degrades;
- per-model quality cards (teacher-forced KL mean/p95, top-1 agreement, task checks) that can gate
  serving admission as an opt-in;
- a `fastmlx` command that pulls a pinned model, ranks the packs that fit this Mac by their measured
  quality, and fit-checks and quality-gates a serve in front of any OpenAI-compatible engine;
- a research OpenAI-compatible chat-completions HTTP/SSE server;
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

A pre-load fit-check sizes the model against your machine and derives the MLX memory and cache
limits from that, so you don't have to supply them. Passing `--memory-limit-bytes` (or
`--cache-limit-bytes`) is optional and means something specific: it's a budget that bounds planning
*below* what this machine would otherwise allow — useful when something else on the box needs the
headroom. It can only lower the ceiling, never raise it, and startup tells you which ceiling
actually bound, including when your budget turned out to be the loose one. Use `--host` / `--port`
to change where it listens, and set `FASTMLX_API_KEY` to require Bearer auth (mandatory for a
non-loopback `--host`).

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

Set `"logprobs":true` (with optional `"top_logprobs":0…20`) to get real per-token log-probabilities
on `choices[0].logprobs.content` (streaming: the same shape as a sibling of `delta` on each chunk
that carries new tokens). Each value is `log_softmax` of the model's own RAW output logits for that
generation step — computed BEFORE temperature, top-p/top-k, min-p, and any penalty/logit processor,
matching vLLM's default logprobs semantics rather than the post-sampling distribution actually drawn
from. Real logprobs are served only by the single-stream scalar route, and only when that route is
serving a plain (native-cache, non-speculative) decoder: a scalar serve running the compiled-fp16
path or in-checkpoint MTP speculative decoding, the continuous-batch route, the separate draft-model
speculative route, and evidence-recording mode all fail closed with 400 `logprobs_unsupported`
rather than silently omitting the values a caller asked for, or reporting logprobs for tokens a
speculative decoder didn't itself score. In practice, hybrid-attention checkpoints served without
MTP compute logprobs. Dense-attention checkpoints, which take the compiled fp16 fast path by
default, refuse them. `top_logprobs` without `logprobs:true` is rejected
(`top_logprobs` param), matching OpenAI's own validation. `logprobs:false`/absent is unchanged — no
`logprobs` key at all, byte-identical to before this feature existed.

Structured output: `"response_format":{"type":"json_object"}` and
`"response_format":{"type":"json_schema","json_schema":{"name":…,"schema":…,"strict":…}}` are enforced
by masking tokens during decoding, so the reply always parses (and, for `json_schema`, validates
against the schema). This covers the OpenAI SDK's `client.chat.completions.parse(response_format=
<Pydantic model>)`, streaming or not. `json_schema` accepts a JSON Schema subset:
- `type` is `object`, `array`, `string`, `number`, `integer`, `boolean` or `null`, or `[T,"null"]`.
- Supported keywords: `properties`, `required`, `additionalProperties:false`, `items`, `enum` of
  scalar literals, `anyOf` whose branches differ in their first character, and local non-recursive
  `$ref` into `$defs` or `definitions`.
- `title`, `description`, `$schema`, `default` and `examples` are ignored.
- Anything else gets a 400 whose message names the JSON-pointer path of the unsupported keyword.
  That includes `pattern`, `format`, length and range bounds, `oneOf`, `allOf`, `not`, `const`,
  recursive `$ref`, and open `additionalProperties`.
- `strict:true` also requires every object to set `additionalProperties:false` and list every
  property in `required`.
- Properties are generated in declared order, and optional ones may be omitted.
- With thinking on, reasoning is generated first and the constraint applies to the reply.

Both formats are served only by the single-stream scalar route and its non-speculative decoders.
Speculative and continuous-batch routes return a 400, and so does combining either format with
`tools`, `logprobs` or `stop`.

`POST /v1/completions` (the legacy OpenAI text-completions shape) is also served, for older clients
such as `openai-python`'s `client.completions.create`, LangChain's `OpenAI` LLM, or lm-eval-style
harnesses. It shares the same sampling, completion-budget, and admission behavior as
`/v1/chat/completions`, but applies no chat template: `prompt` (a non-empty string, or an array of
exactly one string) is tokenized as-is and completed verbatim, with `choices[0].text` in the
response. `n`, `best_of`, and `echo` are only accepted at their neutral single-choice/no-echo values,
and `suffix` (insertion mode) is not supported. Integer `logprobs` (0…5) is a real, honored request —
`0` means "the sampled token's own logprob, no alternatives", a meaningful value distinct from
"logprobs off" — returning `choices[0].logprobs` with `tokens`/`token_logprobs`/`top_logprobs`
(`null` per-token when `logprobs:0`)/`text_offset` (the code-point offset of each token into the
full completion text), same raw-logits semantics as the chat route above; a value outside 0…5 is
rejected. Only the single-stream scalar serving route serves this route today (including
in-checkpoint MTP on that route); the continuous-batch route, the separate draft-model speculative
route, and evidence-recording mode all fail closed with `completions_unsupported` (or
`logprobs_unsupported` for a logprobs request) rather than silently mistreating raw text as an
already-templated chat prompt or omitting values a caller asked for. As with the chat route, a
`logprobs` request against the scalar route ITSELF still refuses with `logprobs_unsupported` when
that scalar serve is running the compiled-fp16 path or in-checkpoint MTP speculative decoding — only
a plain (native-cache, non-speculative) scalar decoder computes real values. An omitted `max_tokens` does
not fall back to the legacy API's 16-token default — it uses the server's own completion-budget
policy, the same one `/v1/chat/completions` applies when `max_completion_tokens` is omitted.

Completion length is model- and host-fit-aware. The default request budget is 4,096 tokens, but it
is not a global maximum: when `--max-completion-tokens` is omitted, the loaded model's authenticated
context and the pre-load host-fit decision determine the ceiling. Use
`--default-completion-tokens`, `--max-completion-tokens`,
`--max-non-streaming-completion-tokens`, and `--completion-limit-policy reject|clamp` to narrow or
shape the policy. Long completions above the non-streaming ceiling remain available with
`"stream":true`. The request-body ceiling scales with the admitted context (1–64 MiB by default),
and `--max-request-body-bytes` plus `--max-non-streaming-response-bytes` provide explicit transport
overrides. Prompt length carries a separate, optional host bound: `--max-prefill-tokens N` rejects a
prompt longer than N tokens at admission, so a prompt a given host cannot prefill fails that request
instead of the process. It is independent of the model's context window, unset (disabled) by
default, and set per host — an interim protection expected to lift once prefill memory is bounded.
Authenticated clients can inspect every effective token and byte limit at
`GET /v1/models` instead of guessing from a model name.

`GET /metrics` (Prometheus text, bearer-authenticated like the API routes) reports resource and
fit gauges plus three HTTP series that are always recorded, whether or not `--request-log json` is
on: `fastmlx_http_requests_total` (labeled by route template, status class and outcome),
`fastmlx_http_request_duration_seconds`, and `fastmlx_http_time_to_first_token_seconds` (streaming
successes only). The histograms use fixed buckets from 5 ms to 300 s. Route labels come from the same
bounded set the access log uses, and any unmatched path is labeled `other`, so label cardinality
stays bounded.

### The `fastmlx` command: pull, recommend, serve

`scripts/fastmlx.py` puts the fit check and the quality cards in front of whichever
OpenAI-compatible engine you serve with. It needs only the Python 3 standard library. The fit check
itself runs `fastmlx-serve --fit-check-only`, so build that first
(`swift build -c release --package-path spike --product fastmlx-serve`). Then put
`spike/.build/release` on `PATH`, or pass `--fit-check-bin`.

```sh
# Pull an exact Hugging Face revision: every file is hash-checked, and an interrupted pull resumes.
python3 scripts/fastmlx.py pull mlx-community/Qwen3-8B-4bit@<40-hex-commit> --dest ./models/qwen3-8b

# Adopt a directory that was staged by hand (rsync, a copy from another host, ...) instead of by
# `fastmlx pull`: verifies every manifest file already there against the pinned revision (size +
# content hash) and writes the same receipt a real pull would, instead of downloading anything.
python3 scripts/fastmlx.py pull mlx-community/Qwen3-8B-4bit@<40-hex-commit> --dest ./models/qwen3-8b --adopt

# Rank the local packs that fit this Mac, with each one's measured quality card.
python3 scripts/fastmlx.py recommend --models-dir ./models

# Fit check, then quality-card admission, then start the engine.
python3 scripts/fastmlx.py serve --model-path ./models/qwen3-8b
```

`pull` writes a receipt next to the model directory that records the exact revision. `serve` and
`recommend` read it to find the model's quality card. A pack whose card says NO_GO is refused until
you opt in with `--accept-quality <card-id>`, so a measured quality cost is never applied silently. A
pack without a card is listed as uncarded and is never recommended over a carded one. `serve` refuses
a model that does not fit unless you pass `--force`, and it refuses outright if the fit check cannot
run. `--engine-profile` points `serve` at a different OpenAI-compatible engine; the default is this
repository's `fastmlx-serve`. A model directory with no receipt and no `--model-revision` has no
resolved identity, so no quality card can ever match it; `serve` prints one stderr line saying so
(admission still proceeds as unmeasured) -- `pull ... --adopt` is how you give a hand-staged directory
that identity without re-downloading it. `--adopt` also walks the whole directory tree (not just the
top level): any file that is not an exact manifest entry, at any depth, is a refusal, and so is an
unlisted symlink; only a dot-prefixed path (`.cache/`, a dotfile, ...) is ignored, so a hand-staged
directory must not carry any extra file the pinned revision does not itself name.

`serve` and `recommend` also accept a GGUF pack directory (one or more top-level `*.gguf` shards, no
`config.json`) as a model path. `scripts/fastmlx_gguf_fit.py` sizes it from its GGUF headers alone
(no weight bytes are read) against the GPU wired-memory limit — read live from `iogpu.wired_limit_mb`,
or 75% of RAM if that sysctl is unavailable — minus an 8 GiB margin by default, plus a KV-cache
reserve you must supply. `--residency expert-stream` sizes only the non-expert tensors and labels the
result a lower bound, for an engine that streams experts from SSD instead of holding them resident.
It does not derive a context ceiling, so pass `--context` yourself and size the KV reserve for it:

```sh
python3 scripts/fastmlx.py serve --model-path ./models/some-gguf-pack --context 32768 \
  --fit-check-bin scripts/fastmlx_gguf_fit.py --fit-check-arg=--kv-reserve-gib --fit-check-arg=16 \
  --engine-profile <your engine profile>
```

For an MLX safetensors pack served by another engine, `scripts/fastmlx_safetensors_fit.py` applies
the same ceiling rule (wired limit minus margin, plus your KV reserve) to the pack's `*.safetensors`
bytes, after checking each shard's header against its file size; shards are counted recursively (any
subdirectory, skipping only a dot-prefixed one), and if a top-level `model.safetensors.index.json` is
present, every shard it names must actually be among the counted files. The default fit check models
this repository's own engine; use this one when the engine you serve with holds only the safetensors
resident. `--wired-limit-mib`, `--wired-margin-gib`, and `--kv-reserve-gib` also accept the same
`FASTMLX_WIRED_LIMIT_MIB`, `FASTMLX_WIRED_MARGIN_GIB`, and `FASTMLX_GGUF_KV_RESERVE_GIB` environment
fallbacks as the GGUF fit check above. A large non-safetensors file in the pack (for example an
n-gram table the engine memory-maps) must be named explicitly with a repeatable `--mmap-side-file`
flag to be excluded from resident bytes and reported instead of counted; an unnamed file of at least
1 GiB is refused as a configuration error rather than silently excluded, since that is an assumption
about the engine you should confirm against a measured peak before relying on it:

```sh
python3 scripts/fastmlx.py serve --model-path ./models/some-mlx-pack --context 262144 \
  --fit-check-bin scripts/fastmlx_safetensors_fit.py --fit-check-arg=--kv-reserve-gib --fit-check-arg=8 \
  --fit-check-arg=--mmap-side-file --fit-check-arg=ngram_table.bin \
  --engine-profile <your engine profile>
```

### Install a prebuilt release (v0.1.0)

Apple Silicon (arm64) macOS only — there is no Intel or Linux build. Download the tarball and its
checksum file, verify, then extract:

```sh
curl -LO https://github.com/bitworks-io/fast-mlx/releases/download/v0.1.0/fastmlx-0.1.0-arm64-macos.tar.gz
curl -LO https://github.com/bitworks-io/fast-mlx/releases/download/v0.1.0/fastmlx-0.1.0-arm64-macos.tar.gz.sha256
shasum -a 256 -c fastmlx-0.1.0-arm64-macos.tar.gz.sha256
tar -xzf fastmlx-0.1.0-arm64-macos.tar.gz
```

The binaries are unsigned and not notarized. A `curl` download carries no quarantine attribute, so
nothing further is needed; a browser download does, and macOS will refuse to run anything inside
the tarball until you clear it:

```sh
xattr -dr com.apple.quarantine fastmlx-0.1.0-arm64-macos
```

Add the extracted `bin` directory to `PATH` (or symlink `bin/fastmlx` into a directory already on
it), then run it:

```sh
export PATH="$PWD/fastmlx-0.1.0-arm64-macos/bin:$PATH"
fastmlx --help
fastmlx capacity --help
```

`fastmlx` is a Python 3 dispatcher and needs `python3` on `PATH` — the `python3` that ships with
Apple's Command Line Tools is enough; nothing further to install. The tarball also ships
`fastmlx-serve`, this repository's bundled fit checker and research engine: it is what `fastmlx`'s
pre-load fit check and default `serve` target run against, exactly as described above.
`fastmlx serve --engine-profile <path>` can front a different OpenAI-compatible engine instead of
the bundled one, by pointing at a profile document that names that engine's binary and argv
template (see `scripts/fastmlx_launch.py` for the profile schema).

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
