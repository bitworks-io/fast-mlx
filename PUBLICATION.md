# Public repository and publication boundary

The engineering checkout contains application source alongside machine-local operator material,
private research, host topology, failed evidence, and agent configuration. It must never be made
public wholesale.

The public `fast-mlx` repository is a fail-closed projection created by
`scripts/export_public_repository.py` from `public/public-repository.json`. The projection includes:

- the project Apache-2.0 `LICENSE` and `NOTICE`;
- Swift application, library, and test source;
- the pinned vendored MLX Swift LM fork with its upstream license and provenance;
- public repository documentation and contribution controls;
- the static website source and its deterministic builder;
- the status-aware capability inventory in `site/capabilities.json`, whose evidence slugs must
  resolve to reviewed published notes;
- the reviewed public release ledger in `site/releases.json`, which generates `/releases/`,
  `/releases/index.json`, and the static `/releases/feed.atom` subscription surface for milestone
  discovery only;
- deterministic `/sitemap.xml` and `/robots.txt` outputs derived only from reviewed human-facing
  routes, with no machine endpoint or unpublished note admitted;
- only the research notes listed in `site/publications.json`; and
- GitHub Actions that validate source/site boundaries and deploy GitHub Pages.

It excludes:

- durable operator handoffs and raw verification logs;
- machine-local paths, hostnames, IP addresses, credentials, and reservations;
- private competitor analysis and predecessor-engine comparisons;
- acquired models, tokenizers, build products, caches, and benchmark roots;
- partial, diagnostic, failed, or stale evidence not explicitly approved for public explanation;
- machine-owned agent configuration and temporary files; and
- local deployment and hardware-control scripts.

## The publishing loop

Every public update follows the same sequence:

```text
research → design → test-first implementation → verification → review
        → public projection → secret/link/build checks → commit and push
        → Pages deployment → live verification → feedback into research
```

The exporter refuses files outside its manifest. The site builder refuses unlisted articles and
known private markers. Capability cards require an explicit state and scope. Numeric highlights also
require the exact model, hardware, workload, date, decision, caveat, and one published evidence slug;
arbitrary source or local artifact paths are not accepted. A release owner reviews the generated diff
and secret scan before creating a public commit. Failed or incomplete measurements may be retained
privately, but they cannot become published benchmark claims.

The benchmark explorer is a read-only view of those same reviewed numeric highlights. Its filters
change visibility only: they do not execute benchmarks, convert units, aggregate results, rank
configurations, or introduce a second machine-readable contract. Site validation requires every
rendered result to keep its exact metric, model, hardware, workload, date, decision, caveat, and
published evidence link attached.

The release ledger is also read-only site data. Its entries point readers to public commits and
reviewed surfaces, and its current-boundary entry remains explicitly gated. The deterministic Atom
feed is generated from those same entries without a build-time network request, external content
ingestion, or publication authority. Neither surface approves a model, launches a runtime, changes
public Apache-2.0 source boundaries, or creates benchmark evidence beyond the reviewed note and
capability manifests.

The sitemap and robots outputs are crawl-discovery hints only. Their exact route set contains the
seven reviewed product/index pages and the seven explicitly published research notes; it excludes
JSON contracts, the Atom feed, LLM text surfaces, assets, and the 404 page. The builder performs no
submission or network request, and validation rejects route drift, non-UTF-8 or oversized content,
DTD/entity declarations, symlinks, and non-regular files.

Canonical and Open Graph metadata are a discovery and presentation surface for that same exact
fourteen-page set. The validator independently pins every page's route, title, description, object
type, and research-note section instead of accepting generated HTML or `research/index.json` as
authority. Canonical and `og:url` values must be identical absolute same-origin URLs; only reviewed
notes may carry article properties. The shared preview PNG is size-, signature-, dimension-, and
SHA-256-pinned, and `404.html`, JSON, Atom, sitemap, robots, LLM text, and asset endpoints cannot
become metadata objects. These tags neither create a search-indexing guarantee nor widen benchmark,
model, acquisition, runtime, or publication authority.

GitHub Pages is downstream of the complete `Public source quality` workflow. It accepts only a
successful `push` run for `main`, checks out the same validated commit, and exposes no independent
manual deployment path. The reusable workflow repeats the event and branch check before building.
A site build therefore cannot publish while the public-boundary or Swift quality jobs for the same
commit are failing or still in progress.

## Creating a candidate projection

From a committed engineering checkpoint:

```sh
python3 scripts/export_public_repository.py --output /fresh/path/fast-mlx-public
python3 scripts/validate_public_repository.py /fresh/path/fast-mlx-public
```

The output path must be absent or empty. The exporter never initializes Git, commits, pushes, or
deletes an existing checkout. Validation requires the Apache-2.0 `LICENSE` and `NOTICE` without a
bypass. Repository creation, GitHub ownership, Pages enablement, and branch protection remain
explicit owner decisions.
