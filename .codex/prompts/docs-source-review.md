# Docs Source Review Prompt Packet

Purpose: verify that documented claims are supported by official sources.
When to use: before publishing docs or agent instructions that cite external APIs, specs, or services.

```
Review the following claims or documentation against the listed official sources.
Do NOT make up citations or paste long excerpts.

Claims or documentation to review:
{claims_or_doc}

Official source URLs to check against:
{official_source_urls}

For each claim:
1. Confirm or flag it with a direct quote or citation (source URL + section/heading).
2. Separate sourced facts from inference: label each as SOURCED or INFERRED.
3. If a claim cannot be verified from the provided sources, mark it UNVERIFIED and
   state what source would be needed to confirm it.
4. Do not introduce new claims not present in {claims_or_doc}.
5. Do not paste excerpts longer than two sentences; a citation is sufficient evidence.

Output format:
- One entry per claim.
- Label: CONFIRMED / CONTRADICTED / UNVERIFIED.
- Evidence: quote or citation (max two sentences) or reason it is unverifiable.
```

Keep output to: a per-claim verdict list with CONFIRMED / CONTRADICTED / UNVERIFIED labels and brief citations.
