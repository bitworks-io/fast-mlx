# fast-mlx

`fast-mlx` is an experimental Swift 6 inference engine and evaluation system for Apple Silicon.
It uses Apple's MLX framework to test inference ideas, measure their real quality and performance
effects, and keep only the capabilities that survive explicit correctness and operational gates.

**Website:** [improvement loop](https://bitworks-io.github.io/fast-mlx/) ·
[capabilities and evidence](https://bitworks-io.github.io/fast-mlx/capabilities/) ·
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
  -d '{"model":"fastmlx-scripted","messages":[{"role":"user","content":"hello"}],"stream":false}'
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
