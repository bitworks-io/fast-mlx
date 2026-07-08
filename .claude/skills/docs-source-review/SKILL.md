---
name: docs-source-review
description: Verify that documented claims or agent instructions are supported by official sources. Use before publishing docs that cite external APIs, specs, or services; labels each claim CONFIRMED, CONTRADICTED, or UNVERIFIED with citations.
---

# Docs Source Review

Verify that documented claims or agent instructions are supported by official sources. Use before publishing docs or agent instructions that cite external APIs, specs, or services. Do not make up citations or paste long excerpts.

## Inputs

- **Claims or documentation to review** — the claims or doc text you were given.
- **Official source URLs to check against** — the source list you were given.

## Procedure

For each claim:

1. Confirm or flag it with a direct quote or citation (source URL + section/heading).
2. Separate sourced facts from inference: label each as SOURCED or INFERRED.
3. If a claim cannot be verified from the provided sources, mark it UNVERIFIED and state what source would be needed to confirm it.

## Constraints

- Do not introduce new claims not present in the claims or documentation you were given.
- Do not paste excerpts longer than two sentences; a citation is sufficient evidence.
- Do not make up citations.

## Output

One entry per claim:

- Label: CONFIRMED / CONTRADICTED / UNVERIFIED.
- Evidence: quote or citation (max two sentences) or reason it is unverifiable.

Keep output to: a per-claim verdict list with CONFIRMED / CONTRADICTED / UNVERIFIED labels and brief citations.
