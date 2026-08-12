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
