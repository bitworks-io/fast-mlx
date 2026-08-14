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
- the reviewed research manifest in `site/publications.json`, which generates `/research/`,
  `/research/index.json`, and the static text-only `/research/feed.atom` subscription surface;
- the static `/quickstart/` operator path for the model-free HTTP/JSON and HTTP/SSE smoke,
  capacity planning entry point, and explicit loaded-serving boundary;
- the static `/status/` current-state dashboard derived from the same reviewed capability,
  release, and research manifests, with no future-roadmap or authority effect;
- six static `/capabilities/<id>/` detail permalinks, one per reviewed capability record, with
  exact state, scope, evidence paths, review dates, and unchanged claim boundary;
- the static text-only `/feed.atom` subscription surface, which combines only those reviewed
  release and research records without becoming another source manifest;
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

The current-status dashboard is a static reader over those already-reviewed records. It presents
capability state counts, the complete scoped feature set, all measured highlights, the latest
reviewed release record, research/release inventory counts, and the current gated runtime/model
boundary. It performs no live lookup, recomputation, ranking, ingestion, future commitment, or
authority transition; exact output sealing prevents jointly drifted rendered claims from becoming
their own source of truth.

Capability detail permalinks are static readers over the exact six reviewed capability records.
Validation reconstructs those records independently from generated `capabilities/index.json`, joins
them to pinned research-note metadata, and seals every generated detail page byte-for-byte. A
capability permalink creates no broader support, measurement, runtime, model, acquisition,
publication, admission, launchability, containment, ranking, or recomputation authority.

The research archive is another read-only progressive-enhancement surface. Search covers only the
reviewed title, summary, and exact published theme; theme selection uses the exact catalog value,
and article order remains date/slug descending. All cards stay visible without JavaScript. The
bounded local script drops overlong query state, ignores unknown themes, updates only visibility
and same-page URL state, and performs no fetch, storage, tracking, ingestion, ranking, publication,
or authority transition. Validation independently pins the card set, order, text, search fields,
theme options, no-JavaScript state, accessibility controls, and exact script bytes.

The release ledger is also read-only site data. Its entries point readers to public commits and
reviewed surfaces, and its current-boundary entry remains explicitly gated. The deterministic Atom
feed is generated from those same entries without a build-time network request, external content
ingestion, or publication authority. Neither surface approves a model, launches a runtime, changes
public Apache-2.0 source boundaries, or creates benchmark evidence beyond the reviewed note and
capability manifests.

The reviewed research feed follows [Atom 1.0](https://datatracker.ietf.org/doc/html/rfc4287) and is
derived independently from the exact publication manifest and pinned note metadata. Stable entry
IDs are the same-origin canonical article URLs; publication and review dates are normalized to UTC
midnight; summaries are XML-escaped text; and article bodies are never embedded. Validation
reconstructs both the research JSON and feed from its own reviewed catalog, so jointly edited output
files cannot authorize a new note, identity, link, ordering, or claim.

The combined reviewed-updates feed is reconstructed from the exact reviewed release index and the
independently pinned research catalog. Its entries retain their release-commit or canonical-article
identities, use explicit `release` and `research` categories, contain text summaries only, and are
ordered by their timezone-aware reviewed timestamps. It introduces no external intake, automatic
publication, benchmark recomputation, ranking, runtime action, or authority transition.

The sitemap and robots outputs are crawl-discovery hints only. Their exact route set contains the
nine reviewed product/index pages, six capability-detail pages, three benchmark-detail pages,
thirteen release-detail pages, and seven explicitly published research notes; it excludes JSON
contracts, all three Atom feeds, LLM text surfaces, assets, and the 404 page. Detail pages are
immutable views of already-reviewed capability records, capability highlights, or release-ledger
entries, not new evidence, support, measurement, runtime, model, acquisition, admission, authority,
ranking, or recomputation. The builder performs no submission or network request, and validation
rejects route drift, non-UTF-8 or oversized content, DTD/entity declarations, symlinks, and
non-regular files.

Canonical and Open Graph metadata are a discovery and presentation surface for that same exact
38-page set. The validator independently pins every page's route, title, description, object
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

After Pages deploys, the same reusable workflow checks out the exact triggering SHA again, rebuilds
and validates the publication subject, and compares bounded HTTPS observations from the fixed
`https://bitworks-io.github.io/fast-mlx/` origin with those exact bytes. Two consecutive complete
matching cohorts are required. The verifier writes one canonical typed `PASS` or `FAIL` receipt;
an independent offline validator must accept the matching result before the workflow can retain the
single receipt artifact. A valid `FAIL` receipt keeps the workflow red and is diagnostic only.

The receipt records content equality for its bounded observation cohorts, not permanent serving
state or GitHub's internal deployment identity. It contains ten literal false authority flags and
cannot approve a model, admit absorbed MLA, authorize execution or containment, ingest evidence,
publish another change, promote a benchmark, or roll back an already deployed site. Cancellation
and concurrency are liveness controls only. Release sealing remains a separate authenticated,
read-only audit of the exact run attempt, complete seven-job set, deployment status, artifact
digest, receipt identity, and public commit.

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
