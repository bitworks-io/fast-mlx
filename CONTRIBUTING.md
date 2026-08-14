# Contributing to fast-mlx

fast-mlx accepts changes through an evidence-first research loop. A fast implementation without a
comparable baseline, a correctness boundary, and a failure story is an experiment—not a feature.

## Before implementation

Open a research or feature issue that states:

- the target user or operator outcome;
- the source paper, implementation, or observed bottleneck;
- the representative happy path;
- correctness, quality, permission, resource, and recovery failures;
- the baseline and exact workload that make results comparable; and
- the proof commands or artifacts that will decide promotion.

External engine work may inform private prioritization, but public claims and public benchmark rows
must describe fast-mlx's own reproducible results only.

## Change loop

1. **Research:** identify the technique and record source/version identity.
2. **Design:** bound the contract, threat model, blast radius, and rollback.
3. **Test first:** add the narrowest meaningful failure or regression contract.
4. **Implement:** make the smallest change that satisfies the contract.
5. **Verify:** run focused tests, relevant regression suites, and acceptance-oriented checks.
6. **Measure:** compare identical workloads and retain failures without promoting partial data.
7. **Review:** inspect correctness, security, public claims, and the final diff.
8. **Publish:** update source, tests, documentation, and a dated research note together.

A candidate is explicitly **promoted**, **shelved**, or **rejected**. Negative results are valuable
and should be published when their evidence is public-safe.

## Pull requests

Pull requests should include:

- the user story and observable acceptance criteria;
- changed behavior and intentionally excluded paths;
- the exact verification commands and mapped results;
- benchmark protocol and provenance when performance is discussed;
- security, privacy, and operational impact; and
- rollback or disablement instructions for risky changes.

Do not commit model weights, tokens, credentials, local absolute paths, host identifiers, raw
benchmark roots, partial evidence, or machine-owned agent state.

## Contribution license

Unless explicitly stated otherwise, a contribution intentionally submitted for inclusion in
fast-mlx is provided under the project's [Apache License 2.0](LICENSE), consistent with section 5
of that license. By submitting a contribution, you represent that you have the right to do so.

## Public claims

Research notes are not automatically published. Additions to the website require an explicit entry
in `site/publications.json` and must pass the public-site validator. Performance claims must be
dated, model/workload/hardware scoped, and clear about whether a result was promoted, negative, or
diagnostic-only.

## Security reports

Do not open a public issue for a suspected vulnerability. Follow [SECURITY.md](SECURITY.md).
